# Kaggle fine-tune notebook — `tunisian_finetune_kaggle.ipynb`

**Standard HuggingFace QLoRA** (transformers + peft + bitsandbytes). No Unsloth — avoids the
`'int' object has no attribute 'mean'` training-step bug that Unsloth hits on Kaggle's current image.

Trains a Tunisian-Arabizi assistant, **measures dialect-rate before vs after** (your
dataset-efficiency result), and lets you chat with the model.

## Steps
1. kaggle.com → **Create → New Notebook**.
2. **+ Add Data → New Dataset** → upload these 3 files:
   - `aigenerateddataset/cs_pairs.jsonl`
   - `aigenerateddataset/parallel_pairs.jsonl`
   - `dataset/eval_set.jsonl`
3. **File → Import Notebook** → upload `training/tunisian_finetune_kaggle.ipynb`.
4. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON**.
5. **Run All**. Default model **Qwen2.5-3B** → ~30–60 min on a T4.

## What you'll see
- **BASELINE** dialect rate (untrained model — usually ~0–5%, answers in Arabic/MSA).
- Training loss going down.
- **AFTER** dialect rate + the **before→after jump** = how well your data teaches Tunisian.
- 8 before/after sample answers, then an interactive chat.
- LoRA adapter saved to `/kaggle/working/tunisian_lora` (download from the **Output** tab).

## Config (cell 2)
- `MODEL_NAME` — default `Qwen/Qwen2.5-3B-Instruct` (fast + reliable on T4). For the real v1:
  `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen3-8B` **and set `BATCH=1`, `GRAD_ACCUM=16`**.
- `CONV_UPSAMPLE` (3) / `PARALLEL_SAMPLE` (5000) — conversation-vs-translation mix.
- `EPOCHS`, `LR`, `BATCH`, `GRAD_ACCUM`.

## Troubleshooting
- **OOM** (GPU memory): lower `MAX_SEQ_LEN` to 768, set `BATCH=1`, or stay on the 3B model.
- **`paged_adamw_8bit` error**: change `optim` to `'adamw_torch'` in cell 8.
- **File not found**: confirm the dataset is attached and the 3 filenames match.
- This notebook does **not** use Unsloth, so the earlier `'int'.mean` crash cannot occur.
