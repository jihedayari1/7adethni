# Kaggle fine-tune notebook — `tunisian_finetune_kaggle.ipynb`

**Standard HuggingFace QLoRA** (transformers + peft + bitsandbytes). No Unsloth — avoids the
`'int' object has no attribute 'mean'` training-step bug that Unsloth hits on Kaggle's image.

This is the **clean retrain** after the last run overfit (loss started at 0.13) and produced
fluent-looking **gibberish** that the old "dialect rate" metric scored 96%.

## What changed vs the last run
- **Real-data anchor:** mixes `real_pairs.jsonl` (real human Arabizi) with your conversational
  pairs. Only the **English→Derja** subset is used (Arabizi *output*) so the model learns correct
  words/spelling — it can't invent them. Comprehension/vocab pairs (English output) are filtered out.
- **Anti-overfit:** 1 epoch · LR 1e-4 · LoRA r=8 · **no** conversation upsampling.
- **Honest metric:** headline is **real-word rate** (vs the 17k lexicon). Dialect rate is kept only
  as a surface check.

## Steps
1. kaggle.com → **Create → New Notebook**.
2. **+ Add Data → New Dataset** → upload these **4** files:
   - `aigenerateddataset/cs_pairs.jsonl`
   - `aigenerateddataset/real_pairs.jsonl`
   - `dataset/eval_set.jsonl`
   - `rag/lexicon.jsonl`  ← needed for the real-word metric
3. **File → Import Notebook** → upload `training/tunisian_finetune_kaggle.ipynb`.
4. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON**.
5. **Run All**. Qwen2.5-7B on a T4 ≈ 1.5–2.5 h.

## What you'll see
- **BASELINE** real-word rate + dialect rate (base model, no adapter).
- Training loss going down (it should *not* start near 0.1 this time).
- **AFTER** real-word rate + dialect rate, with the before→after delta.
- 8 before/after samples — judge **coherence**, not just "looks like Arabizi".
- LoRA adapter saved to `/kaggle/working/tunisian_lora` (download from the **Output** tab).

### How to read it
- **AFTER ≥ BEFORE real-word rate _and_ coherent samples** → the data helped, ship the adapter.
- **AFTER drops below BEFORE** → the fine-tune is hurting. Ship the **base model + `serving/`
  grounding** instead (that was the whole finding). The base 7B already produces good Tunisian.

## Config (cell 2)
- `MODEL_NAME` — default `Qwen/Qwen2.5-7B-Instruct`. Faster test: `Qwen/Qwen2.5-3B-Instruct`
  (then `BATCH=2`, `GRAD_ACCUM=4`).
- `CONV_UPSAMPLE` (1) · `REAL_SAMPLE` (4000) — conversation-vs-real-anchor mix.
- `EPOCHS` (1) · `LR` (1e-4) · `LORA_R` (8) — the anti-overfit knobs.

## Troubleshooting
- **OOM** (GPU memory): lower `MAX_SEQ_LEN` to 768, or use the 3B model.
- **`paged_adamw_8bit` error**: change `optim` to `'adamw_torch'` in the train cell.
- **File not found**: confirm the dataset is attached and all 4 filenames match.
- **real-word metric says "disabled"**: you didn't upload `lexicon.jsonl`.
