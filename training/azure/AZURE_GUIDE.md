# 7adethni — Training v3 on Azure (result-trusted)

The Kaggle notebooks were built to *survive* free hardware. This suite is built to **produce a result
you can defend**: validation loss, best-checkpoint selection, deterministic eval, ablation, and
confidence intervals — plus a `manifest.json` that ties every model back to the exact data that made it.

## What's different from the Kaggle version

| | Kaggle v2 | **Azure v3** |
|---|---|---|
| Validation split | ❌ none | ✅ held-out, stratified by task |
| Overfitting visible? | ❌ no | ✅ `eval_loss` every 100 steps |
| Which checkpoint ships | last one | ✅ **best val_loss** (+ early stopping) |
| Eval decoding | sampling (noisy) | ✅ **greedy** (reproducible) |
| "Did CPT help?" | unanswerable | ✅ **ablation** (base / cpt / cpt+sft / sft-only) |
| "Is the gap real?" | unanswerable | ✅ **paired bootstrap**, 95% CI |
| Reproducible? | ❌ | ✅ seed + data SHA + config in `manifest.json` |
| Precision | 4-bit only | ✅ bf16 on A100/H100, 4-bit auto-fallback |
| Corpus | 9.4M tokens | ✅ 17.3M (balanced — see the note below) |

---

## 0. Machine
Any single NVIDIA GPU works; the scripts auto-configure. Recommended Azure SKUs:

| SKU | GPU | What you get |
|---|---|---|
| **NC24ads A100 v4** | 1× A100 80GB | ⭐ best value — bf16, seq 2048, batch 8. CPT ≈ 2–3 h |
| NC A100 v4 (40GB) | 1× A100 40GB | bf16, batch 4. CPT ≈ 4–5 h |
| NCasT4_v3 | 1× T4 16GB | falls back to 4-bit automatically (like Kaggle) |

```bash
git clone <your repo> && cd 7adethni
pip install -r training/azure/requirements.txt
# optional but ~30% faster on A100/H100:
pip install flash-attn --no-build-isolation
```

## 1. Sanity-check without burning GPU time
```bash
python training/azure/train_cpt.py --dry-run
python training/azure/train_sft.py --dry-run
```
Prints the detected GPU, chosen precision/batch, token count and step estimate, and writes a manifest.
**Always run this first** — it catches data problems in seconds instead of an hour in.

## 2. Stage 1 — CPT
```bash
python training/azure/train_cpt.py \
    --data training/cpt.jsonl \
    --out outputs/tunisian_cpt \
    --seq 2048 --epochs 1 --lora-r 64 \
    --report-to mlflow          # Azure ML picks this up automatically
```
Watch **`eval_loss`**: it should fall then flatten. If it starts *rising*, the run is overfitting —
early stopping and best-checkpoint selection protect you, but that's the signal to use less data repetition.

## 3. Stage 2 — SFT (+ the ablation that proves CPT's value)
```bash
# the real model
python training/azure/train_sft.py --cpt outputs/tunisian_cpt --out outputs/tunisian_final --epochs 2

# ablation: same SFT, WITHOUT the CPT stage  (run this — it is the control group)
python training/azure/train_sft.py --cpt none --out outputs/sft_only --epochs 2
```

## 4. Judge it — the part that makes results trustworthy
```bash
python training/azure/evaluate.py \
    --adapters base=none cpt=outputs/tunisian_cpt final=outputs/tunisian_final sft_only=outputs/sft_only \
    --out outputs/eval
```
You get a table like:
```
model            real_word          number_rule       comprehension
base           88% [84%-92%]       94% [90%-97%]      41% [37%-45%]
final          94% [91%-97%]       98% [96%-99%]      63% [59%-67%]

Is the gap real?  P(model > base) by paired bootstrap
  final      real_word 99%  |  comprehension 100%  |  number_rule 96%
```
**Decision rule:**
- **P(>base) ≥ 90%** on comprehension **and** real_word → the gain is real. Ship it.
- **P between 40–60%** → indistinguishable from base. **Do not ship**; more GPU won't fix it, more real data will.
- `final` vs `sft_only` tells you whether **CPT was worth it** — keep or drop the stage next round.

Then the one thing statistics can't replace: **your native blind A/B** on 50 prompts.

## 5. Ship
```powershell
modal volume put 7adethni-adapter ./outputs/tunisian_final /tunisian_final
modal app stop 7adethni-serving; modal deploy serving/modal_app.py
```

---

## Honest limitations (read before trusting a number)

1. **`eval_set.jsonl` has only 101 prompts → ±5.9 points of noise.** A 5-point gain is *invisible*
   there. The comprehension test (600 items, ±2.4 pts) is the metric with real statistical power —
   weight your decision on it.
2. **Real Arabizi is the bottleneck, not compute.** We only have ~240k tokens of natively-written
   Arabizi (TUNIZI + lexicon). It is upsampled 10× to reach 11% of the corpus. Scaling the corpus
   past ~20M tokens *lowers* that share and biases the model toward transliterated, vowel-poor style
   (`wallh nfs al7kaia`) — the builder now warns you when this happens.
   **The highest-value work is not a bigger GPU — it is scraping more natively-Arabizi text**
   (`dataset/tools/yt_comments.py` → `build_real_conv.py`).
3. **CPT sees held-out test words** in raw text (deliberate — that's language exposure, not answer
   leakage; the test maps EN gloss → Arabizi word, a mapping CPT never sees). SFT is fully clean:
   1,584 lexicon entries / 5,245 word forms excluded, verified 0 leaks.
4. **bf16 training, 4-bit serving** is normal QLoRA practice; expect a small, usually favorable delta.

## Troubleshooting
- **OOM**: lower `--seq 1024`, then `--batch`, then `--quant 4bit`.
- **flash-attn build fails**: skip it, the scripts fall back to PyTorch SDPA automatically.
- **`eval_loss` rising from step 1**: LR too high — try `--lr 1e-4` (CPT) / `--lr 5e-5` (SFT).
- **Resume after a crash**: `--resume outputs/tunisian_cpt/checkpoint-XXXX`.
