"""
Data augmentation for the Sharp CV-P09FX Proxy-KD task.
=======================================================

Why: the distilled student memorises the Q->A mapping for *seen* phrasings but fails the SAME fact
in an unseen phrasing (it aces "Can PLASMACLUSTER be used in VENTILATION?" but confabulates "What
happens if you press PLASMACLUSTER in VENTILATION?"). The ablations showed this is a data-coverage
problem, not a teacher- or KD-hyperparameter problem. Fix: express each fact in MANY question forms.

This reuses the original generator's design (chunk -> sample dimensions -> grounded LLM gen ->
LLM-judge -> filter) but:
  * adds a PHRASING_STYLE dimension (the actual fix) so each fact gets asked many ways,
  * calls DeepInfra directly (OpenAI-compatible) instead of the heavy `data-designer` dep,
  * writes straight to our ChatML {"text": ...} format with <think>,
  * KEEPS the 32-question val set untouched (eval stays comparable) and only grows TRAIN.

Answers are authored by Llama-3.3-70B (grounded + judge-filtered). In Proxy-KD these are just the
hard-label token sequence; the grounded proxy still supplies the soft-labels at train time.

Run (Colab):  set DEEPINFRA_API_KEY in Colab Secrets, then `!python augment_data.py`
Run (local):  set env var DEEPINFRA_API_KEY, then `python augment_data.py`
Deps:         pip install openai
"""

import os, re, json, random, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ------------------------------- Config -----------------------------------------------------
GEN_MODEL   = "meta-llama/Meta-Llama-3.3-70B-Instruct"
JUDGE_MODEL = "meta-llama/Meta-Llama-3.3-70B-Instruct"
BASE_URL    = "https://api.deepinfra.com/v1/openai"

MANUAL_PATH = "sharp_cv_p09fx_manual_en_cleaned.md"
TRAIN_IN    = "ac_manual_synth_tra.jsonl"   # existing 185 Claude pairs (kept)
VAL_IN      = "ac_manual_synth_val.jsonl"   # 32 held-out (used only for anti-leakage dedup)
TRAIN_OUT   = "ac_manual_synth_tra_aug.jsonl"

N_NEW       = 500     # target NEW kept examples (on top of the existing 185)
MAX_WORKERS = 6
SEED        = 0

SYSTEM = ("You are a helpful assistant for the Sharp CV-P09FX portable air conditioner. "
          "Answer questions accurately based on the product manual.")

QUESTION_TYPES = [
    "safety and warnings", "installation procedure", "operation and controls",
    "troubleshooting", "maintenance and cleaning", "energy efficiency tips",
    "warranty and specifications",
]

# The fix: the SAME fact asked in many forms, so the student generalises past one wording.
PHRASING_STYLES = {
    "yes_no":       "Phrase it as a direct yes/no capability question (e.g. 'Can I...?', 'Is it possible to...?').",
    "what_happens": "Phrase it as 'What happens if I ...?' about a specific action or setting.",
    "why":          "Phrase it as a 'Why ...?' question asking for the reason/cause behind a behaviour.",
    "troubleshoot": "Phrase it as a troubleshooting report: the user describes a symptom and asks what to check or do.",
    "how_to":       "Phrase it as a step-by-step 'How do I ...?' request.",
    "lookup":       "Phrase it as a short factual lookup of a specific value, spec, number, or limit.",
    "compare":      "Phrase it as a comparison between two modes, settings, or options.",
}
DIFFICULTIES = ["basic", "intermediate", "advanced"]

random.seed(SEED)

# ------------------------------- Manual chunking --------------------------------------------
def chunk_manual(text, target=1200, overlap=100):
    """Split on markdown headers first, then pack paragraphs to ~target chars."""
    sections, cur = [], []
    for line in text.splitlines():
        if re.match(r"^#{1,3}\s", line) and cur:
            sections.append("\n".join(cur)); cur = [line]
        else:
            cur.append(line)
    if cur:
        sections.append("\n".join(cur))
    chunks = []
    for sec in sections:
        if len(sec) <= target:
            if len(sec.strip()) > 100:
                chunks.append(sec.strip())
            continue
        paras = re.split(r"\n\s*\n", sec)
        buf = ""
        for p in paras:
            if len(buf) + len(p) > target and buf:
                chunks.append(buf.strip()); buf = buf[-overlap:] + "\n\n" + p
            else:
                buf += ("\n\n" if buf else "") + p
        if len(buf.strip()) > 100:
            chunks.append(buf.strip())
    # merge consecutive small chunks so each has enough context to ground a question
    merged, buf = [], ""
    for c in chunks:
        if len(buf) + len(c) <= target:
            buf = (buf + "\n\n" + c) if buf else c
        else:
            if buf:
                merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)
    return merged

# ------------------------------- Dedup helpers ----------------------------------------------
_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)

def norm_q(q):
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()

def load_questions(path):
    qs = []
    if not os.path.exists(path):
        return qs
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        m = _USER_RE.search(json.loads(line)["text"])
        if m:
            qs.append(m.group(1).strip())
    return qs

# ------------------------------- LLM calls --------------------------------------------------
client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("DEEPINFRA_API_KEY", ""))

def _json_call(model, prompt, temperature=0.7, max_tokens=900):
    r = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, max_tokens=max_tokens,
    )
    txt = r.choices[0].message.content
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group()) if m else None

def gen_qa(chunk, qtype, phrasing, difficulty):
    prompt = (
        "You are a technical documentation expert for the Sharp CV-P09FX portable air conditioner.\n"
        f"Using ONLY the information in the context, create a {difficulty} question about {qtype}.\n"
        f"QUESTION STYLE: {PHRASING_STYLES[phrasing]}\n\n"
        f"<context>\n{chunk}\n</context>\n\n"
        "Return JSON with keys:\n"
        '  "question": a self-contained question a real user would ask, answerable from the context.\n'
        '  "reasoning": step-by-step reasoning that locates the relevant facts in the context.\n'
        '  "answer": the final, clean, accurate answer grounded strictly in the context (no invented facts).\n'
    )
    return _json_call(GEN_MODEL, prompt, temperature=0.8)

def judge(chunk, q, a):
    prompt = (
        "Evaluate a Q&A pair against the context from the Sharp CV-P09FX manual.\n\n"
        f"<context>\n{chunk}\n</context>\n\n"
        f"Question: {q}\nAnswer: {a}\n\n"
        "Return JSON with keys 'groundedness' (Grounded|PartiallyGrounded|Ungrounded), "
        "'accuracy' (Accurate|PartiallyAccurate|Inaccurate), "
        "'relevance' (Relevant|PartiallyRelevant|Irrelevant)."
    )
    return _json_call(JUDGE_MODEL, prompt, temperature=0.1, max_tokens=200)

# ------------------------------- ChatML writer ----------------------------------------------
def to_chatml(question, reasoning, answer):
    assistant = f"<think>\n{reasoning.strip()}\n</think>\n{answer.strip()}"
    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>")
    return {"text": text}

# ------------------------------- Main -------------------------------------------------------
def one_record(chunks, seen):
    for _ in range(3):  # a few tries to pass the judge / dedup
        chunk = random.choice(chunks)
        qa = gen_qa(chunk, random.choice(QUESTION_TYPES),
                    random.choice(list(PHRASING_STYLES)), random.choice(DIFFICULTIES))
        if not qa or not qa.get("question") or not qa.get("answer"):
            continue
        q = qa["question"].strip()
        if norm_q(q) in seen:
            continue
        v = judge(chunk, q, qa["answer"])
        if not v:
            continue
        if (v.get("groundedness") == "Grounded" and v.get("accuracy") == "Accurate"
                and v.get("relevance") == "Relevant"):
            return to_chatml(q, qa.get("reasoning", ""), qa["answer"]), norm_q(q)
    return None, None

def main():
    assert os.environ.get("DEEPINFRA_API_KEY"), \
        "Set DEEPINFRA_API_KEY (Colab Secrets -> os.environ, or an env var)."
    manual = open(MANUAL_PATH, encoding="utf-8").read()
    chunks = chunk_manual(manual)
    print(f"Manual chunks: {len(chunks)}")

    seen = set(norm_q(q) for q in load_questions(TRAIN_IN) + load_questions(VAL_IN))
    print(f"Existing questions (train+val, for dedup): {len(seen)}")

    kept, lock_seen = [], set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(one_record, chunks, seen) for _ in range(int(N_NEW * 1.6))]
        for fut in as_completed(futures):
            rec, nq = fut.result()
            if rec and nq not in lock_seen:
                lock_seen.add(nq); seen.add(nq); kept.append(rec)
                if len(kept) % 25 == 0:
                    print(f"  kept {len(kept)}")
            if len(kept) >= N_NEW:
                break
    print(f"Generated {len(kept)} new grounded, judge-passed examples.")

    # existing 185 Claude pairs are kept verbatim, new ones appended
    existing = [json.loads(l) for l in open(TRAIN_IN, encoding="utf-8") if l.strip()]
    out = existing + kept
    random.shuffle(out)
    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {TRAIN_OUT}: {len(existing)} Claude + {len(kept)} new = {len(out)} train examples "
          f"(val untouched: {VAL_IN}).")

if __name__ == "__main__":
    main()
