# Proxy-KD from scratch: distilling a black-box LLM into a 350M model

A minimal, heavily-commented, **from-scratch PyTorch** implementation of **Proxy-KD**
([Knowledge Distillation of Black-Box Large Language Models](https://arxiv.org/abs/2401.07013),
Chen et al., 2024) — applied to a real task: turning **LiquidAI LFM2.5-350M** into a Q&A
assistant for the *Sharp CV-P09FX* portable air-conditioner manual, using **Claude** as the
(black-box) teacher.

This repo is a learning artifact. The code is deliberately a ~450-line single file you can read
top-to-bottom, not a framework. It also documents the **things that actually went wrong** and how
we fixed them — which turned out to be more instructive than the method itself.

---

## The problem Proxy-KD solves

Classic (white-box) knowledge distillation needs the teacher's **full next-token probability
distribution** so the student can match it with a KL loss. But the strongest teachers
(Claude, GPT-4, Gemini) are **API-only**: you get *text*, not logits. Naive "black-box KD" then
degenerates into plain SFT on the teacher's text, throwing away the soft-label signal that makes
KD work.

**Proxy-KD** inserts an open-weight **proxy** model whose logits you *do* control. You first align
the proxy to imitate the black-box teacher, then do ordinary white-box KD **from the proxy** to
your student.

```
  Black-box teacher (Claude)  ──text──►  Proxy (LFM2-1.2B)  ──logits / KL──►  Student (LFM2.5-350M)
          align the proxy to the teacher                    white-box KD to the student
```

| Role | In the paper | In this repo |
|------|--------------|--------------|
| Black-box teacher | GPT-4 | **Claude** — its answers are pre-generated offline in `ac_manual_synth_*.jsonl` (no API calls at train time) |
| Proxy (has logits) | Llama-2-70B | **LFM2-1.2B** |
| Student | Llama-2-7B | **LFM2.5-350M** |

Proxy and student are both LFM2-family, so they **share one tokenizer** (vocab 65536) — a
prerequisite for the token-level KL in Stage B (the code asserts this).

---

## Method: three stages

### Stage A1 — Proxy SFT
Fine-tune the proxy (LoRA) on the teacher's answers with plain next-token NLL:

$$\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(x,y)}\big[-\log \pi_p(y \mid x)\big]$$

### Stage A2 — Proxy DPO
Push the proxy toward the teacher's answer $y$ and away from the proxy's *own* sample $\hat y$
(on-policy preference alignment). The reference policy is the same model with the LoRA adapter
disabled — so no second copy of the weights is held in memory:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big[\beta \log\tfrac{\pi_p(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \beta \log\tfrac{\pi_p(\hat y\mid x)}{\pi_{\text{ref}}(\hat y\mid x)}\Big]$$

### Stage B — Student distillation with adaptive weighting
The student learns from the aligned proxy's soft labels **plus** the hard labels, teacher-forced
on the teacher's answer tokens. Each example is weighted by how *confident* the proxy is on it, so
a shaky proxy can't mislead the student where it's unreliable:

$$\mathcal{L}_{\text{student}} = \underbrace{-\log\pi_s(y\mid x)}_{\text{hard labels}} \; + \; \alpha \, \underbrace{w(x,y)\, D_{\text{KL}}\big(\pi_p(y\mid x)\,\|\,\pi_s(y\mid x)\big)}_{\text{weighted soft labels}}$$

$$w(x,y) = \sigma\!\Big(\frac{\overline{\log \pi_p}(y\mid x) - \mu}{\gamma}\Big)$$

where $\mu, \gamma$ are the mean/std of the proxy's **per-token** log-likelihood over the dataset
(per-token, not summed, to remove length bias). Computed once up front.

---

## Repo layout

| File | What it is |
|------|-----------|
| `proxy_kd.py` | The whole base implementation — imports, config, data, the three stages, eval, `main()`. Read this first. |
| `proxy_kd_grounded.py` | The **grounded variant** (LFM2-2.6B proxy + manual in-context + cached soft-labels). The stronger version — see *Grounded variant* below. |
| `proxy_kd_colab.ipynb` | The base code split into Colab cells (install → upload data → config → code → run). |
| `ac_manual_synth_tra.jsonl` | 185 training Q&A pairs (Claude answers, ChatML, with `<think>` blocks). |
| `ac_manual_synth_val.jsonl` | 32 held-out validation Q&A pairs. |
| `sharp_cv_p09fx_manual_en_cleaned.md` | Cleaned English manual — the grounding source for `proxy_kd_grounded.py`. |
| `requirements.txt` | `transformers`, `peft`, `accelerate`, `torch`. |

### Data format
Each line is `{"text": "<ChatML conversation>"}`. The assistant turn is Claude's answer, shaped as
`<think>{reasoning}</think>{final answer}`. By default the loader **strips the `<think>` block**
and trains on the final answer only (see *Findings* — this matters a lot for a 350M model).

---

## How to run

### On Google Colab (recommended — needs a GPU)
1. Open `proxy_kd_colab.ipynb` in Colab and set the runtime to a **GPU** (T4 works; L4 is faster).
2. Run the **install** cell (it also `pip uninstall -y torchao` — see *Gotchas*).
3. Run the **upload** cell and select both `ac_manual_synth_*.jsonl` files.
4. Run the config + code cells, then the final cell:
   ```python
   main(run_a1=True, run_a2=True, run_b=True)
   ```

### As a script
```bash
pip install -r requirements.txt
# place the two .jsonl files next to proxy_kd.py, then:
python proxy_kd.py
```

Each stage **saves its LoRA adapter to disk**, so you can re-run a single stage by flipping the
flags (e.g. resume at Stage B after a disconnect):
```python
main(run_a1=False, run_a2=False, run_b=True)
```

Runtime on an L4: A1 ≈ 1–2 min, A2 ≈ a few min (the rejected-sampling loop is the slow part),
Stage B ≈ 7–10 min.

### Key knobs (`proxy_kd.py`, section 1)
| Knob | Default | Note |
|------|---------|------|
| `STRIP_THINK` | `True` | drop Claude's `<think>` CoT, train on final answers only |
| `ALPHA_KD` | `1.0` | weight on the soft-label (KD) term; `0.0` = pure SFT baseline |
| `B_EPOCHS` | `15` | student epochs |
| `MAX_NEW_TOK` | `128` | cap for the DPO rejected-sample length (speeds up A2) |

---

## Findings

The interesting part. Training the pipeline was easy; getting *coherent generations* out of a 350M
model took real debugging. In rough order of impact:

### 1. Training loss ≠ generation quality
Every stage's loss fell beautifully (student NLL → ~0.2), yet the first generations were
multilingual word-salad. Low teacher-forced loss says "predicts the next token well *given the
correct prefix*"; free-running generation is a different, harder test (exposure bias), especially
for a small model.

### 2. Decoding config was the single biggest cause of "garbage"
We initially evaluated with `repetition_penalty=1.3` + `no_repeat_ngram_size=3`. On an overfit,
*multilingual* 350M base, once those suppress the common English tokens a factual answer needs, the
distribution tips into other-language tokens and cascades. **Plain greedy + stop on `<|im_end|>`**
fixed most of the "garbage" with zero retraining.

```
before (rep_penalty 1.3):  "urodzeni focus target distance ... återstwahrsogenverbesserung ..."
after  (plain greedy):     "After a power failure, the air conditioner will not restart.
                            Check the following: 1. Check the power supply ... 2. Check the fuse ..."
```

### 3. Chain-of-thought traces hurt a tiny student
Claude's answers are `<think>{long reasoning}</think>{answer}`. A 350M model spends its whole budget
imitating the reasoning and never converges to a clean final answer or a stop token. **Stripping
`<think>` to answer-only** produced concise, correctly-terminating answers. (`<think>` is a special
token; because LoRA only touches attention and the `lm_head` stays frozen, the model emits a
cosmetic junk token — `" pau"` — in its place, then recovers.)

### 4. The KD term acted as a regularizer
The pure-SFT baseline (`ALPHA_KD=0`) overfit *harder* (NLL → 0.10) and generated *worse* than the
Proxy-KD student (`ALPHA_KD=1`, NLL → 0.24). So the soft-label KL wasn't just "extra signal" — it
kept the student's distribution from collapsing.

### 5. The proxy's fidelity is the ceiling — this is Proxy-KD's whole point
Even after everything above, the student's **factual accuracy** stays weak: it hallucinates
buttons, phone numbers, and steps. Why? The 1.2B proxy — with no manual in context and only 185
examples — **confabulates**, and a student can't be more correct than the teacher it distills from.
The paper used a **70B** proxy for exactly this reason. Our takeaway matches its thesis in the
negative: **Proxy-KD is only as good as how faithfully the proxy reproduces the black-box teacher.**
With a small proxy you get fluent, well-structured, on-topic answers but shaky facts.

### Gotchas (environment)
- **torchao/peft clash on Colab.** Colab preinstalls `torchao 0.10`, which recent `peft` rejects at
  `get_peft_model` (wants >0.16). We don't use it → `pip uninstall -y torchao`.
- **`PeftModel.from_pretrained` loads adapters frozen** (`is_trainable=False`). To *continue*
  training a saved adapter (Stage A2 resuming from A1), pass `is_trainable=True`.

---

## Results (qualitative)

With `STRIP_THINK=True` + plain-greedy decoding, the Proxy-KD student produces clean, structured,
correctly-stopping answers:

> **Q: After a power failure, the air conditioner won't restart. What should I check?**
> After a power failure, the air conditioner will not restart. Check the following:
> 1. **Check the power supply**: ensure the unit is plugged in and the power plug is functioning.
> 2. **Check the fuse**: locate the power plug and check if the fuse has blown; if so, replace it.
> …

Fluent and on-topic; factual precision is bounded by the 1.2B proxy (see Finding #5).

---

## Grounded variant — raising the ceiling (`proxy_kd_grounded.py`)

Finding #5 says the proxy's fidelity is the ceiling. So we attacked the proxy directly, two ways at
once (`proxy_kd_grounded.py`):

1. **Bigger proxy:** `LFM2-2.6B` instead of `LFM2-1.2B` (still vocab 65536 → token-level KL intact).
2. **Grounded proxy:** the **full manual (~11.5k tokens) goes in the proxy's system context**, so it
   answers from the text instead of confabulating. The student still never sees the manual — the
   grounded knowledge moves into its weights. This fuses Proxy-KD with *context-distillation*.

Efficiency: the proxy is frozen and teacher-forced on fixed answer tokens, so its soft-labels are
**cached once** (one forward per example) rather than recomputed every epoch — the long manual
context is paid a single time. The proxy is used **zero-shot** by default (a grounded 2.6B is
already a strong teacher); `main(run_a1=True)` optionally SFTs it first.

**Results (vs the ungrounded 1.2B proxy), on the same 32-question val set:**

| | Ungrounded 1.2B | **Grounded 2.6B** |
|---|---|---|
| Fluency / format / stopping | broke into multilingual salad | clean, structured, stops ✅ |
| Proxy confidence `mu` (mean logprob of gold answer) | −1.68 | **−0.98** |
| Factual accuracy (qualitative) | hallucinated freely | ~4/6 correct, quotes the manual ✅ |

Example — the student now matches the teacher *and quotes the source*:

> **Q: Can PLASMACLUSTER be used during VENTILATION mode?**
> **Student:** "No, the PLASMACLUSTER function cannot be used during VENTILATION mode. The manual
> states: *'The PLASMACLUSTER function cannot be used during VENTILATION mode.'*"

**A surprising result: the student sometimes beats its own proxy teacher.** On "can I run COOL mode
at 98°F?" the zero-shot proxy wrongly said *yes*; the student correctly said *no*. The student gets
the best of both signals — the teacher's *correct* answers via the hard-label NLL **and** the
grounded soft-labels via KD — so it stays anchored on truth even where the proxy drifts.

**What's still imperfect:** subtle facts and cross-phrasing generalization (it nailed the
PLASMACLUSTER fact in one phrasing but confabulated it in another). That residual is now bounded by
the **350M student's capacity** and the **185-example coverage** — no longer by the teacher. The
next levers are proxy SFT (`run_a1=True`), more/paraphrased data, or a bigger student.

---

## When does each knob help? (Stage-A and KD-temperature ablations)

We pushed two more knobs to find out where the remaining errors actually come from. The short
answer: **once the proxy is grounded, neither knob buys factual accuracy** — the bottleneck has
moved off the teacher.

### Stage A: SFT the grounded proxy (`main(run_a1=True)`)

SFT-ing the grounded 2.6B on the teacher's answers made it an almost-perfect Claude replica:

| Proxy confidence `mu` (mean logprob of Claude's answer) | ungrounded 1.2B | grounded 2.6B zero-shot | **grounded 2.6B + Stage-A** |
|---|---|---|---|
| | −1.68 | −0.98 | **−0.28** (≈ 75%/token) |

The proxy's own wrong facts got corrected (e.g. the "set both timers" question flipped wrong →
right). **But the student's held-out accuracy did not improve (~3/6, same as zero-shot grounded).**
Two reasons:

1. **The bottleneck moved.** Q2/Q4 fail on *held-out phrasings* — a 350M-capacity + 185-example
   limit that a better teacher can't fix.
2. **A too-confident teacher regularizes less.** The peaked SFT'd soft-labels dropped `w·kl` from
   ~0.19 → ~0.08, so the student overfit *harder* (NLL 0.14 → 0.07). Stage A improved the teacher
   but quietly weakened the KD regularizer that was helping the student.

> Note on cost: Stage A back-props through the ~11.5k-token grounded context on the 2.6B, so it needs
> `gradient_checkpointing_enable()` **and** `enable_input_require_grads()` (frozen base + checkpointing
> won't propagate gradients otherwise). ~50 min on an L4.

### KD temperature (`T = 2`, soften the KL only, keep NLL at `T = 1`)

To restore the regularization Stage A removed, we re-distilled with the KL softened at `T=2` (×T² to
keep its weight) while pinning the hard-label NLL at `T=1`. Regularization came back (`w·kl` ~0.4–2.2,
slower NLL descent) — and the effect on outputs was revealing:

- **Formatting generalization improved** — the louvers answer became a near-perfect step list.
- **Factual accuracy did not** — and one question *regressed*: the 98°F/COOL answer flipped from a
  correct "No" to a self-contradiction ("Yes… since 98°F is above the upper limit, you can still
  operate").

**Why:** a fact lives in the distribution's **peak** (the yes/no token). Temperature deliberately
flattens peaks — great for transferring open-ended *style*, bad for *decisiveness* on factual tokens.
So KD temperature is a **form-vs-fact dial, not an accuracy dial**.

### Takeaway

| Knob | What it fixes | What it can't fix |
|---|---|---|
| Ground + enlarge the proxy | teacher fidelity (the ceiling) | — this was the big win |
| Stage-A proxy SFT | teacher fidelity even further | student accuracy (bottleneck moved) |
| KD temperature | formatting / style transfer | facts (can even hurt decisiveness) |

The remaining errors are **data-coverage / student-capacity bound**. The smoking gun: the student
*aces* "Can PLASMACLUSTER be used in VENTILATION?" but *confabulates* "What happens if you press
PLASMACLUSTER in VENTILATION?" — the **same fact, reworded**. The one lever that targets this is
**data augmentation**: express each fact in several phrasings so the student generalizes past
memorizing one wording. (A bigger student would also help.)

---

## Limitations & next steps
- **Raise the proxy's fidelity** (the real lever): give the proxy the manual *in context* during
  Stage A (RAG the teacher) so it stops confabulating, or use a larger/stronger proxy.
- **Scale is small on purpose** — 185 examples, a 1.2B proxy, a 350M student. This is a
  method-learning exercise, not a SOTA result.
- **Modern alternative:** for the black-box *on-policy* setting, see **GAD** (Generative Adversarial
  Distillation, [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)) — it replaces the proxy with a
  co-evolving discriminator reward model.

---

## References
- Chen et al., *Knowledge Distillation of Black-Box Large Language Models*, [arXiv:2401.07013](https://arxiv.org/abs/2401.07013)
- Ye et al., *Black-Box On-Policy Distillation of Large Language Models* (GAD), [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
- Base models: [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B), [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)

## License
MIT (code). The manual-derived Q&A data is for research/educational use.
