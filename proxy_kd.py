"""
Proxy-KD for the Sharp CV-P09FX AC-manual task  (from-scratch, Colab-T4 friendly)
=================================================================================

Paper: "Knowledge Distillation of Black-Box Large Language Models" (arXiv:2401.07013)

The problem Proxy-KD solves
---------------------------
Classic (white-box) KD needs the TEACHER's full next-token probability vector so the
student can match it with a KL loss. But the best teachers (Claude / GPT-4 / Fable-5)
are API-only: you get TEXT, not logits. Proxy-KD bridges that gap with an open-weight
PROXY whose logits you *can* read:

    BLACK-BOX TEACHER (Claude)  --text-->  PROXY (LFM2-1.2B)  --logits/KL-->  STUDENT (LFM2.5-350M)
            (align proxy to teacher)             (white-box KD to student)

In OUR setup the teacher is Claude, and we already have its text answers offline in
`ac_manual_synth_tra.jsonl` (each line is a ChatML conversation whose assistant turn is
Claude's answer, including a <think> block). So we never call an API here -- the
"black-box teacher" is that fixed corpus of Claude outputs.

Proxy and student are both LFM2-family => they share ONE tokenizer (vocab 65536), so the
token-level KL in Stage B is valid (we assert this below).

NOTE vs the earlier on-policy `onpolicy_distill.py`:
  * There, the "teacher" (LFM2-1.2B) only answered well because we pasted the MANUAL into
    its context (context-distillation), and the student learned via *reverse* KL on its own
    rollouts. There is NO manual in Proxy-KD: the proxy INTERNALISES Claude's behaviour via
    alignment (Stage A), then hands dense soft-labels to the student (Stage B).
  * The student stage here is FORWARD KL  D_KL(proxy || student), teacher-forced on Claude's
    answer tokens (off-policy) -- faithful to the paper. This tends to transfer *facts*
    better than the per-token reverse-KL on rollouts did (a lesson from the on-policy run).

Three stages
------------
  A1. Proxy SFT     : fine-tune LFM2-1.2B (LoRA) on Claude's answers            (NLL)
  A2. Proxy DPO     : push the proxy toward Claude vs its own samples           (preference)
  B.  Student distil: LFM2.5-350M learns from proxy soft-labels + hard labels,
                      with a PER-SAMPLE adaptive weight that trusts the proxy only
                      where the proxy is confident.

Run on a free Colab T4 (16 GB). Microbatch = 1 + gradient accumulation keeps it simple.
Upload `ac_manual_synth_tra.jsonl` and `ac_manual_synth_val.jsonl` next to this file first.
"""

# --------------------------------------------------------------------------------------------
# 0. Install (Colab). Comment out locally.
# --------------------------------------------------------------------------------------------
# !pip install -q "transformers>=4.56" "peft>=0.13" accelerate
# Colab ships torchao 0.10, which the fresh peft rejects (wants >0.16) at get_peft_model.
# We don't use torchao -> remove it so peft's is_torchao_available() returns False.
# !pip uninstall -y torchao

import os
import json
import math
import copy
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

# --------------------------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------------------------
PROXY_ID   = "LiquidAI/LFM2-1.2B"     # white-box proxy: aligned to Claude, then teaches student
STUDENT_ID = "LiquidAI/LFM2.5-350M"   # final small model we ship

DATA_TRA = "ac_manual_synth_tra.jsonl"   # Claude answers (the offline black-box teacher corpus)
DATA_VAL = "ac_manual_synth_val.jsonl"   # held out for eval only

SYSTEM = ("You are a helpful assistant for the Sharp CV-P09FX portable air conditioner. "
          "Answer questions accurately based on the product manual.")

# --- Stage A1: proxy SFT ---
A1_EPOCHS = 3
A1_LR     = 1e-4

# --- Stage A2: proxy DPO ---
A2_ROUNDS   = 1        # iterations of (sample rejected -> DPO). 1 is enough for practice.
A2_EPOCHS   = 2        # passes over the data per round
A2_LR       = 5e-6     # DPO wants a small LR
DPO_BETA    = 0.1
GEN_TEMP    = 1.0      # sampling temp for the proxy's "rejected" rollouts
GEN_TOP_P   = 0.95
MAX_NEW_TOK = 128      # cap the DPO rejected-sample length -> A2 sampling is far faster

# --- Stage B: student distillation ---
B_EPOCHS   = 15        # think-stripped answers are short -> 15 is plenty and less overfit
B_LR       = 2e-4      # match the LR the reference SFT used; small LoRA tolerates it.
STRIP_THINK = True     # train on Claude's FINAL answer only (drop <think>..</think>): a 350M
                       # student rambles / can't stop if it must imitate the long CoT (see load_qa)
KD_TEMP    = 1.0       # temperature inside the KL
ALPHA_KD   = 1.0       # weight on the soft-label term. Paper uses 100 under a different
                       # normalisation; with per-token forward-KL, ~1 balances against the
                       # per-token NLL. Tune up if soft labels look under-used.
ACCUM      = 8         # gradient accumulation (all stages, microbatch=1)
SEED       = 0

PROXY_SFT_DIR = "./lfm2_1.2b_proxy_sft_lora"
PROXY_DPO_DIR = "./lfm2_1.2b_proxy_dpo_lora"
STUDENT_DIR   = "./lfm25_350m_ac_proxykd_lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32   # T4 has no real bf16
random.seed(SEED); torch.manual_seed(SEED)

# --------------------------------------------------------------------------------------------
# 2. Data: each JSONL line is {"text": "<ChatML>"}; pull (question, claude_answer).
#    Claude's answers are "<think>{reasoning}</think>{final answer}". A 350M student can't
#    imitate the long chain-of-thought and stop cleanly, so STRIP_THINK keeps only the final
#    answer (much cleaner, stoppable generations). Set STRIP_THINK=False to distill the CoT too.
# --------------------------------------------------------------------------------------------
import re
_USER_RE  = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
_ASST_RE  = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

def load_qa(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            u = _USER_RE.search(text); a = _ASST_RE.search(text)
            if u and a:
                ans = a.group(1).strip()
                if STRIP_THINK:
                    ans = _THINK_RE.sub("", ans).strip()   # drop the <think>..</think> block
                pairs.append((u.group(1).strip(), ans))
    return pairs

# --------------------------------------------------------------------------------------------
# 3. Tokenizer + shared-vocab check (token-level KL is only valid if the maps agree)
# --------------------------------------------------------------------------------------------
print("Loading tokenizer ...")
tok = AutoTokenizer.from_pretrained(STUDENT_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_t = AutoTokenizer.from_pretrained(PROXY_ID)
_probe = "Window panel width 22 inches (559mm). Drain the condensate."
assert tok(_probe)["input_ids"] == _t(_probe)["input_ids"], \
    "Proxy and student tokenizers disagree -- token-level KL would be meaningless."
del _t
print("Tokenizer alignment OK.")

# --------------------------------------------------------------------------------------------
# 4. Core teacher-forcing helpers (shared by every stage)
#
#    We tokenize the PROMPT (system+user+generation-prompt) and the ANSWER separately, but via
#    the chat template so the LFM2 special tokens (<|im_start|>/<|im_end|>) are exactly right.
#    Trick: build the full conversation WITH the assistant turn, and take answer_ids as the
#    suffix after the prompt -- guarantees the answer's trailing <|im_end|> is included.
# --------------------------------------------------------------------------------------------
def build_prompt_answer_ids(question, answer):
    prompt_msgs = [{"role": "system", "content": SYSTEM},
                   {"role": "user",   "content": question}]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": answer}]
    prompt_ids = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)["input_ids"]
    full_ids = tok.apply_chat_template(full_msgs, add_generation_prompt=False,
                                       return_tensors="pt", return_dict=True)["input_ids"]
    answer_ids = full_ids[:, prompt_ids.shape[1]:]
    return prompt_ids.to(DEVICE), answer_ids.to(DEVICE)

def build_prompt_ids(question):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"]
    return ids.to(DEVICE)

def logits_on_answer(model, prompt_ids, answer_ids):
    """Logits that PREDICT each answer token: shape [answer_len, V].
    Logit at position p predicts token p+1, so the block predicting the answer starts at
    index (prompt_len - 1) of the concatenated [prompt ++ answer] sequence."""
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = model(full).logits[0]                    # [seq, V]
    start = prompt_ids.shape[1] - 1
    return logits[start : start + answer_ids.shape[1]]  # [answer_len, V]

def token_logprobs(model, prompt_ids, answer_ids):
    """Per-token log-prob the model assigns to the *actual* answer tokens: [answer_len]."""
    logp = F.log_softmax(logits_on_answer(model, prompt_ids, answer_ids).float(), dim=-1)
    return logp.gather(-1, answer_ids[0].unsqueeze(-1)).squeeze(-1)   # [answer_len]

def nll_loss(model, prompt_ids, answer_ids):
    """Mean negative log-likelihood over the answer tokens (the SFT / hard-label loss)."""
    return -token_logprobs(model, prompt_ids, answer_ids).mean()

# --------------------------------------------------------------------------------------------
# 5. Stage A1 -- Proxy SFT: teach LFM2-1.2B to imitate Claude's answers (hard labels only).
# --------------------------------------------------------------------------------------------
PROXY_LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0,
                  target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                  task_type="CAUSAL_LM")

def load_proxy(adapter_dir=None, merge=False, trainable=False):
    base = AutoModelForCausalLM.from_pretrained(PROXY_ID, torch_dtype=DTYPE).to(DEVICE)
    if adapter_dir is None:
        return get_peft_model(base, LoraConfig(**PROXY_LORA))     # fresh trainable adapter
    # NOTE: from_pretrained loads adapters in INFERENCE mode (is_trainable=False) by default,
    # which freezes the LoRA params. Pass trainable=True to keep fine-tuning them (Stage A2).
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=trainable)
    if merge:
        model = model.merge_and_unload()                         # bake in -> fast frozen fwd
    return model

def train_proxy_sft(pairs):
    print("\n===== Stage A1: Proxy SFT =====")
    proxy = load_proxy()
    proxy.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in proxy.parameters() if p.requires_grad], lr=A1_LR)
    proxy.train()
    step = 0
    for epoch in range(A1_EPOCHS):
        random.shuffle(pairs)
        opt.zero_grad(); running = 0.0; counted = 0
        for i, (q, a) in enumerate(pairs):
            prompt, ans = build_prompt_answer_ids(q, a)
            if ans.shape[1] == 0:
                continue
            loss = nll_loss(proxy, prompt, ans)
            (loss / ACCUM).backward()
            running += loss.item(); counted += 1
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in proxy.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"  A1 epoch {epoch} step {step}  nll={running/max(counted,1):.4f}")
                    running = 0.0; counted = 0
        opt.step(); opt.zero_grad()
    proxy.save_pretrained(PROXY_SFT_DIR)
    print(f"  saved proxy-SFT adapter -> {PROXY_SFT_DIR}")
    del proxy, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------------
# 6. Stage A2 -- Proxy DPO: prefer Claude's answer y over the proxy's own sample y_hat.
#
#    Iterative & on-policy in spirit: each round the proxy samples its CURRENT best guess
#    (the "rejected" response), and DPO pushes probability mass from y_hat toward Claude's y.
#    The DPO reference distribution pi_ref is just this same model with the LoRA DISABLED,
#    so we need no second copy of the 1.2B weights in memory.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def proxy_sample(proxy, question):
    prompt = build_prompt_ids(question)
    out = proxy.generate(prompt, max_new_tokens=MAX_NEW_TOK, do_sample=True,
                         temperature=GEN_TEMP, top_p=GEN_TOP_P,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, prompt.shape[1]:]
    eos = tok.eos_token_id
    if eos is not None and (gen[0] == eos).any():
        gen = gen[:, : (gen[0] == eos).nonzero()[0, 0].item() + 1]
    return gen

def seq_logprob(model, prompt_ids, answer_ids):
    """Sum of per-token log-probs over the answer = log pi(answer | prompt)."""
    return token_logprobs(model, prompt_ids, answer_ids).sum()

def dpo_pair_logps(proxy, prompt, ans):
    """(current-policy logprob, reference logprob) for one answer, sharing the base weights."""
    lp_cur = seq_logprob(proxy, prompt, ans)                 # LoRA enabled  (has grad)
    with torch.no_grad(), proxy.disable_adapter():
        lp_ref = seq_logprob(proxy, prompt, ans)             # base model    (reference)
    return lp_cur, lp_ref

def train_proxy_dpo(pairs):
    print("\n===== Stage A2: Proxy DPO =====")
    proxy = load_proxy(PROXY_SFT_DIR, trainable=True)        # start from the SFT'd proxy
    proxy.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in proxy.parameters() if p.requires_grad], lr=A2_LR)
    step = 0
    for rnd in range(A2_ROUNDS):
        # (a) Collect fresh "rejected" samples from the CURRENT proxy (on-policy).
        #     Unbatched generation -> this is the slowest part of A2 (no logs until DPO starts).
        proxy.eval()
        print(f"  A2 round {rnd}: sampling {len(pairs)} rejected responses ...")
        rejected = [proxy_sample(proxy, q) for q, _ in pairs]
        # (b) DPO passes: chosen = Claude's answer, rejected = proxy's own sample.
        proxy.train()
        for epoch in range(A2_EPOCHS):
            order = list(range(len(pairs)))
            random.shuffle(order)
            opt.zero_grad(); running = 0.0; counted = 0
            for k, idx in enumerate(order):
                q, a = pairs[idx]
                yhat = rejected[idx]
                if yhat.shape[1] == 0:
                    continue
                prompt, chosen = build_prompt_answer_ids(q, a)
                lp_cur_w, lp_ref_w = dpo_pair_logps(proxy, prompt, chosen)   # winner (Claude)
                lp_cur_l, lp_ref_l = dpo_pair_logps(proxy, prompt, yhat)     # loser  (proxy)
                # DPO loss: -log sigmoid( beta * [ (cur_w - ref_w) - (cur_l - ref_l) ] )
                margin = DPO_BETA * ((lp_cur_w - lp_ref_w) - (lp_cur_l - lp_ref_l))
                loss = -F.logsigmoid(margin)
                (loss / ACCUM).backward()
                running += loss.item(); counted += 1
                if (k + 1) % ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in proxy.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad(); step += 1
                    if step % 5 == 0:
                        print(f"  A2 round {rnd} epoch {epoch} step {step} "
                              f"dpo={running/max(counted,1):.4f}")
                        running = 0.0; counted = 0
            opt.step(); opt.zero_grad()
    proxy.save_pretrained(PROXY_DPO_DIR)
    print(f"  saved proxy-DPO adapter -> {PROXY_DPO_DIR}")
    del proxy, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------------
# 7. Stage B -- Student distillation with per-sample adaptive weighting.
#
#    Loss(x, y) = NLL_student(y|x)              # hard labels (Claude's answer tokens)
#               + ALPHA * w(x,y) * KL(proxy || student)   # soft labels from the aligned proxy
#
#    w(x,y) = sigmoid( (mean-logprob_proxy(y|x) - mu) / gamma )
#      -> down-weights the soft term where the proxy is UNSURE about y (so a shaky proxy can't
#         mislead the student there; the student leans on the hard NLL instead).
#      -> mu, gamma are the mean / std of the proxy's per-token log-likelihood across the data,
#         computed ONCE up front. (We use per-token mean, not the raw sum, to remove length
#         bias so w reflects confidence rather than answer length.)
# --------------------------------------------------------------------------------------------
STUDENT_LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                    task_type="CAUSAL_LM")   # bigger than the proxy's: the 350M student needs
                                             # more capacity to absorb the facts + English style.

@torch.no_grad()
def proxy_confidence(proxy, pairs):
    """Per-example mean log-prob the aligned proxy gives Claude's answer; plus mu, gamma."""
    scores = []
    for q, a in pairs:
        prompt, ans = build_prompt_answer_ids(q, a)
        if ans.shape[1] == 0:
            scores.append(float("nan")); continue
        scores.append(token_logprobs(proxy, prompt, ans).mean().item())
    s = torch.tensor([x for x in scores if not math.isnan(x)])
    mu, gamma = s.mean().item(), (s.std().item() + 1e-6)
    print(f"  proxy confidence: mu={mu:.3f}  gamma={gamma:.3f}")
    return scores, mu, gamma

def train_student(pairs):
    print("\n===== Stage B: Student distillation =====")
    # Aligned proxy: load the DPO adapter (fall back to SFT) and MERGE -> a fast frozen teacher.
    proxy_dir = PROXY_DPO_DIR if os.path.isdir(PROXY_DPO_DIR) else PROXY_SFT_DIR
    print(f"  using aligned proxy from {proxy_dir}")
    proxy = load_proxy(proxy_dir, merge=True).eval()
    for p in proxy.parameters():
        p.requires_grad_(False)

    scores, mu, gamma = proxy_confidence(proxy, pairs)

    student = get_peft_model(
        AutoModelForCausalLM.from_pretrained(STUDENT_ID, torch_dtype=DTYPE).to(DEVICE),
        LoraConfig(**STUDENT_LORA))
    student.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=B_LR)
    student.train()

    step = 0
    for epoch in range(B_EPOCHS):
        order = list(range(len(pairs)))
        random.shuffle(order)
        opt.zero_grad(); run_nll = 0.0; run_kl = 0.0; counted = 0
        for k, idx in enumerate(order):
            q, a = pairs[idx]
            if math.isnan(scores[idx]):
                continue
            prompt, ans = build_prompt_answer_ids(q, a)
            if ans.shape[1] == 0:
                continue

            # Frozen proxy soft labels over Claude's answer tokens.
            with torch.no_grad():
                p_logp = F.log_softmax(
                    logits_on_answer(proxy, prompt, ans).float() / KD_TEMP, dim=-1)  # [L,V]
            # Student distribution over the same tokens (with grad).
            s_logits = logits_on_answer(student, prompt, ans).float()
            s_logp = F.log_softmax(s_logits / KD_TEMP, dim=-1)                       # [L,V]

            # Forward KL  D_KL(proxy || student), per token then mean.
            kl = (p_logp.exp() * (p_logp - s_logp)).sum(-1).mean()
            # Hard-label NLL from the same student logits (reuse s_logp, no second forward).
            nll = -s_logp.gather(-1, ans[0].unsqueeze(-1)).squeeze(-1).mean()

            w = 1.0 / (1.0 + math.exp(-(scores[idx] - mu) / gamma))   # sigmoid confidence
            loss = nll + ALPHA_KD * w * kl

            (loss / ACCUM).backward()
            run_nll += nll.item(); run_kl += (w * kl).item(); counted += 1
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"  B epoch {epoch} step {step}  nll={run_nll/max(counted,1):.4f} "
                          f"w*kl={run_kl/max(counted,1):.4f}")
                    run_nll = run_kl = 0.0; counted = 0
        opt.step(); opt.zero_grad()

    student.save_pretrained(STUDENT_DIR)
    print(f"  saved student adapter -> {STUDENT_DIR}")
    return student, proxy

# --------------------------------------------------------------------------------------------
# 8. Lightweight eval: greedy answers + mean forward-KL(proxy||student) on held-out questions.
#    (For the full rubric-based reward, plug in ropd_rubric.ipynb's judge on these answers.)
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_answer(model, question, max_new=256):
    prompt = build_prompt_ids(question)
    # Plain greedy is best here. repetition_penalty / no_repeat_ngram tip an overfit 350M
    # multilingual model into other-language garbage. Stop on <|im_end|> so it doesn't ramble
    # past the answer (that token is the assistant-turn terminator in the LFM2 chat format).
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out = model.generate(prompt, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=im_end, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def eval_student(student, proxy, val_pairs, n_show=3):
    print("\n===== Eval  (Proxy = quality ceiling, Student = what we shipped) =====")
    for q, _ in val_pairs[:n_show]:
        print(f"Q: {q}\nProxy  : {greedy_answer(proxy, q)}\nStudent: {greedy_answer(student, q)}\n")

# --------------------------------------------------------------------------------------------
# 9. Orchestration
# --------------------------------------------------------------------------------------------
def main(run_a1=True, run_a2=True, run_b=True):
    train_pairs = load_qa(DATA_TRA)
    val_pairs   = load_qa(DATA_VAL)
    print(f"Loaded {len(train_pairs)} train QA pairs, {len(val_pairs)} val QA pairs.")

    if run_a1:
        train_proxy_sft(train_pairs)
    if run_a2:
        train_proxy_dpo(train_pairs)
    if run_b:
        student, proxy = train_student(train_pairs)
        eval_student(student, proxy, val_pairs)

if __name__ == "__main__":
    main()
