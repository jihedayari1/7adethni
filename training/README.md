# Kaggle fine-tune notebook — `tunisian_finetune_kaggle.ipynb`

Trains a Tunisian-Arabizi assistant on your data, **measures dialect-rate before vs after**
(your dataset-efficiency result), and lets you chat with the model. Free Kaggle GPU.

## Steps
1. Go to **kaggle.com → Create → New Notebook**.
2. **+ Add Data → New Dataset** → upload these 3 files (from this repo):
   - `aigenerateddataset/cs_pairs.jsonl`
   - `aigenerateddataset/parallel_pairs.jsonl`
   - `dataset/eval_set.jsonl`
3. **File → Import Notebook** → upload `training/tunisian_finetune_kaggle.ipynb`.
4. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON**.
5. **Run All**. ~30–60 min on a T4.

## What you'll see
- **BASELINE** dialect rate (untrained model on the eval prompts — usually low, leans MSA/English).
- Training loss going down.
- **AFTER** dialect rate + the **before→after jump** = how well your data teaches Tunisian.
- 8 before/after sample answers, then an interactive chat.
- The LoRA adapter saved to `/kaggle/working/tunisian_lora` (download from the **Output** tab).

## Config (cell 2)
- `MODEL_NAME` — default `Qwen2.5-7B` (safe on T4). Switch to `unsloth/Qwen3-8B-bnb-4bit` for the
  real v1, or `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` for a faster run.
- `CONV_UPSAMPLE` (3) / `PARALLEL_SAMPLE` (8000) — the conversation-vs-translation mix.
- `EPOCHS`, `LR`, `BATCH`, `GRAD_ACCUM`.

## Troubleshooting
- **`SFTTrainer` complains about `dataset_text_field` / `max_seq_length`**: trl version drift —
  replace `TrainingArguments(...)` with `trl.SFTConfig(...)` and move those two args into it.
- **OOM on a single T4**: lower `MAX_SEQ_LEN` to 768, `BATCH` to 1, or use the 3B model.
- **File not found**: make sure the dataset is attached and the 3 filenames match.
