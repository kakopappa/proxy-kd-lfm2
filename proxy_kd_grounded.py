"""
Grounded Proxy-KD  (proxy = LFM2-2.6B WITH the manual in its context)
====================================================================

An improved variant of `proxy_kd.py`. Two changes fix the factual-accuracy ceiling we hit with
the small ungrounded proxy:

  1. Bigger proxy: LFM2-2.6B instead of LFM2-1.2B (still shares the student's tokenizer,
     vocab 65536, so token-level KL stays valid).
  2. GROUNDED proxy: the product manual is placed in the proxy's system context, so the proxy
     stops confabulating and gives *accurate* soft labels. The student never sees the manual
     (knowledge moves into its weights) -> this is Proxy-KD fused with context-distillation.

Efficiency: the proxy is frozen and teacher-forced on fixed answer tokens, so we CACHE its
soft-labels ONCE (one forward per example) instead of every epoch. That makes the long
(~11.5k-token) manual context essentially free, and lets the student train fast.

By default the proxy is used ZERO-SHOT (no fine-tune): a capable grounded 2.6B instruct model
is already a strong teacher. Set run_a1=True to also SFT the proxy on the teacher's answers.

Result on the Sharp CV-P09FX task (vs the ungrounded 1.2B proxy): clean/structured/stoppable
answers, proxy confidence mu improved -1.68 -> -0.98, and ~4/6 val questions factually correct
(the student sometimes beats its own proxy, because it also sees Claude's correct answers via
the NLL term). Remaining errors are on subtle facts / cross-phrasing generalization, bounded by
the 350M student and 185-example coverage.

Run on Colab L4 (bf16). Upload/download the two JSONL files and the manual .md next to this file.
"""

# --------------------------------------------------------------------------------------------
# Install (Colab): pip install -q "transformers>=4.56" "peft>=0.13" accelerate ; pip uninstall -y torchao
# --------------------------------------------------------------------------------------------
import os, json, math, re, random
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------------- Config --------------------------------------------------
PROXY_ID   = "LiquidAI/LFM2-2.6B"     # bigger + GROUNDED (manual in its context)
STUDENT_ID = "LiquidAI/LFM2.5-350M"   # blind student (never sees the manual)

DATA_TRA    = "ac_manual_synth_tra.jsonl"
DATA_VAL    = "ac_manual_synth_val.jsonl"
MANUAL_PATH = "sharp_cv_p09fx_manual_en_cleaned.md"

SYSTEM = ("You are a helpful assistant for the Sharp CV-P09FX portable air conditioner. "
          "Answer questions accurately based on the product manual.")
STRIP_THINK    = True     # train on Claude's final answers, not the <think> chain-of-thought
MAX_MANUAL_TOK = 12000    # full manual (~11.5k tok); cached once so the long context is cheap

A1_EPOCHS = 3;  A1_LR = 1e-4          # optional proxy SFT (run_a1=True)
B_EPOCHS  = 15; B_LR = 2e-4           # student distillation
KD_TEMP = 1.0;  ALPHA_KD = 1.0;  ACCUM = 8;  SEED = 0

PROXY_SFT_DIR = "./lfm2_2.6b_proxy_grounded_lora"
STUDENT_DIR   = "./lfm25_350m_proxykd_grounded_lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------- Data ----------------------------------------------------
_USER_RE  = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
_ASST_RE  = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

def load_qa(path):
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)["text"]
        u = _USER_RE.search(t); a = _ASST_RE.search(t)
        if u and a:
            ans = a.group(1).strip()
            if STRIP_THINK:
                ans = _THINK_RE.sub("", ans).strip()
            pairs.append((u.group(1).strip(), ans))
    return pairs

# ---------------------------- Tokenizer + grounded prompt -----------------------------------
tok = AutoTokenizer.from_pretrained(STUDENT_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
_t = AutoTokenizer.from_pretrained(PROXY_ID)
assert tok("Window panel width 22 inches (559mm).")["input_ids"] == \
       _t("Window panel width 22 inches (559mm).")["input_ids"], "tokenizer mismatch -> token KL invalid"
del _t
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

_mids = tok(open(MANUAL_PATH, encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
MANUAL = tok.decode(_mids[:MAX_MANUAL_TOK])
PROXY_SYSTEM = SYSTEM + "\n\n=== PRODUCT MANUAL (answer strictly from this) ===\n" + MANUAL

def _ids(msgs, gen):
    return tok.apply_chat_template(msgs, add_generation_prompt=gen,
                                   return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)

def student_prompt_answer(q, a):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    p = _ids(m, True); full = _ids(m + [{"role": "assistant", "content": a}], False)
    return p, full[:, p.shape[1]:]

def student_prompt(q):
    return _ids([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}], True)

def proxy_prompt(q):   # GROUNDED: manual in the system message
    return _ids([{"role": "system", "content": PROXY_SYSTEM}, {"role": "user", "content": q}], True)

def logits_on_answer(model, prompt_ids, answer_ids):
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = model(full).logits[0]
    s = prompt_ids.shape[1] - 1
    return logits[s : s + answer_ids.shape[1]]

def token_logprobs(model, p, a):
    lp = F.log_softmax(logits_on_answer(model, p, a).float(), dim=-1)
    return lp.gather(-1, a[0].unsqueeze(-1)).squeeze(-1)

# ---------------------------------- Models --------------------------------------------------
PROXY_LORA   = dict(r=16, lora_alpha=32, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], task_type="CAUSAL_LM")
STUDENT_LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], task_type="CAUSAL_LM")

def load_proxy(adapter=None, merge=False, trainable=False):
    base = AutoModelForCausalLM.from_pretrained(PROXY_ID, torch_dtype=DTYPE).to(DEVICE)
    if adapter is None:
        return base                                          # zero-shot grounded proxy
    m = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    return m.merge_and_unload() if merge else m

# --------- Optional Stage A: SFT the grounded proxy on the teacher's answers ----------------
def train_proxy_sft(pairs):
    print("\n===== Stage A: GROUNDED Proxy SFT (optional) =====")
    base = load_proxy()
    base.config.use_cache = False
    base.gradient_checkpointing_enable()                     # 11.5k-token backward is memory-heavy
    base.enable_input_require_grads()                        # frozen base + checkpointing -> let grads flow
    proxy = get_peft_model(base, LoraConfig(**PROXY_LORA))
    proxy.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in proxy.parameters() if p.requires_grad], lr=A1_LR)
    proxy.train(); step = 0
    for e in range(A1_EPOCHS):
        random.shuffle(pairs); opt.zero_grad(); run = 0.0; c = 0
        for i, (q, a) in enumerate(pairs):
            _, ans = student_prompt_answer(q, a)
            if ans.shape[1] == 0:
                continue
            loss = -token_logprobs(proxy, proxy_prompt(q), ans).mean()
            (loss / ACCUM).backward(); run += loss.item(); c += 1
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_([p for p in proxy.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"  A e{e} s{step} nll={run/max(c,1):.4f}"); run = 0.0; c = 0
        opt.step(); opt.zero_grad()
    proxy.save_pretrained(PROXY_SFT_DIR); print("  saved", PROXY_SFT_DIR)
    del proxy, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ------------------------ Stage B: cache soft-labels, train student -------------------------
def train_student(pairs, proxy):
    print("\n===== Stage B: caching grounded proxy soft-labels =====")
    cache, scores = [], []
    with torch.no_grad():
        for q, a in pairs:
            _, ans = student_prompt_answer(q, a)
            if ans.shape[1] == 0:
                cache.append(None); scores.append(float("nan")); continue
            lp = F.log_softmax(logits_on_answer(proxy, proxy_prompt(q), ans).float() / KD_TEMP, dim=-1)
            cache.append((ans.detach().cpu(), lp.half().cpu()))
            scores.append(lp.gather(-1, ans[0].unsqueeze(-1)).squeeze(-1).mean().item())
    s = torch.tensor([x for x in scores if not math.isnan(x)]); mu = s.mean().item(); gamma = s.std().item() + 1e-6
    print(f"  cached {sum(c is not None for c in cache)} examples. confidence mu={mu:.3f} gamma={gamma:.3f}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    student = get_peft_model(AutoModelForCausalLM.from_pretrained(STUDENT_ID, torch_dtype=DTYPE).to(DEVICE),
                             LoraConfig(**STUDENT_LORA))
    student.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=B_LR)
    student.train(); step = 0
    print("===== Stage B: training student (blind, no manual) =====")
    for e in range(B_EPOCHS):
        order = list(range(len(pairs))); random.shuffle(order)
        opt.zero_grad(); rn = 0.0; rk = 0.0; c = 0
        for k, idx in enumerate(order):
            if cache[idx] is None:
                continue
            q, _ = pairs[idx]
            ans = cache[idx][0].to(DEVICE); p_logp = cache[idx][1].to(DEVICE).float()
            s_logp = F.log_softmax(logits_on_answer(student, student_prompt(q), ans).float() / KD_TEMP, dim=-1)
            kl = (p_logp.exp() * (p_logp - s_logp)).sum(-1).mean()
            nll = -s_logp.gather(-1, ans[0].unsqueeze(-1)).squeeze(-1).mean()
            w = 1.0 / (1.0 + math.exp(-(scores[idx] - mu) / gamma))
            loss = nll + ALPHA_KD * w * kl
            (loss / ACCUM).backward(); rn += nll.item(); rk += (w * kl).item(); c += 1
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    print(f"  B e{e} s{step} nll={rn/max(c,1):.4f} w*kl={rk/max(c,1):.4f}"); rn = rk = 0.0; c = 0
        opt.step(); opt.zero_grad()
    student.save_pretrained(STUDENT_DIR); print("  saved", STUDENT_DIR)
    return student

# ---------------------------------- Eval ----------------------------------------------------
@torch.no_grad()
def gen(model, prompt_ids, max_new=200):
    out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                         eos_token_id=IM_END, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()

def evaluate(student, proxy, val, n=6):
    print("\n===== Eval  (REF=teacher | Proxy=grounded 2.6B | Student=blind 350M) =====")
    for q, ref in val[:n]:
        print(f"\nQ: {q}")
        print(f"  REF    : {ref[:280]}")
        print(f"  Proxy  : {gen(proxy, proxy_prompt(q))[:280]}")
        print(f"  Student: {gen(student, student_prompt(q))[:280]}")

def main(run_a1=False):
    tr = load_qa(DATA_TRA); va = load_qa(DATA_VAL)
    print(f"Loaded {len(tr)} train, {len(va)} val pairs. Manual: {len(_mids)} tokens.")
    if run_a1:
        train_proxy_sft(tr)
        proxy = load_proxy(PROXY_SFT_DIR, merge=True).eval()   # SFT'd grounded proxy
    else:
        proxy = load_proxy().eval()                            # zero-shot grounded proxy
    for p in proxy.parameters():
        p.requires_grad_(False)
    student = train_student(tr, proxy)
    evaluate(student, proxy, va)

if __name__ == "__main__":
    main(run_a1=False)
