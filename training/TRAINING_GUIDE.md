# 7adethni — Training v2, step by step (CPT → SFT → TunBench)

The recipe that actually teaches Arabizi + Tunisian slang, on free Kaggle GPUs.
**Why staged:** CPT teaches the *language* (natural Arabizi, slang) from real text; SFT teaches it
to *follow instructions*; TunBench tells you honestly if it worked. Skipping CPT is what failed before.

```
 LOCAL (laptop CPU)                         KAGGLE (free T4)                 SHIP
 process_corpus ─┐                                                          
 build_cpt_corpus├─ cpt.jsonl ───────────▶  Stage 1: cpt_kaggle.ipynb  ──▶ tunisian_cpt
 build_sft_real ─┴─ sft_real.jsonl ──┐                                        │
 tunbench build ── comprehension ────┴────▶ Stage 2: sft_kaggle.ipynb ──▶ tunisian_final ─▶ serving/
```

Everything below is copy-paste. Run local steps from the repo root.

---

## Prerequisites (local, one time)
```bash
pip install pandas pyarrow            # read the TUNIZI parquet
pip install datasketch                # OPTIONAL: near-dedup for the corpus (recommended)
```

## STEP 1 — Process the 3GB corpus  *(local, runs overnight, CPU only)*
Turns `chunk_*.json.zip` into clean, dialect-bucketed text.
```bash
python corpus/process_corpus.py                 # all chunks  (or --limit 5000 to test first)
```
→ `corpus/corpus_clean.jsonl`. Look at the printed buckets: **tn_arabic** should dominate (that's
your transliteration fuel). *If you skip this, CPT still runs on TUNIZI + lexicon, just smaller.*

## STEP 2 — Build the CPT corpus  *(local, seconds–minutes)*
```bash
python training/build_cpt_corpus.py --tokens 8000000    # ~1 Kaggle session
```
→ `training/cpt.jsonl` (~7M tokens, 41 MB). Bigger needs >1 Kaggle session — 8M is the sweet spot.

## STEP 3 — Build the real SFT set  *(local)*
```bash
python training/build_sft_real.py
```
→ `training/sft_real.jsonl` (~22k real pairs: translate / use-word / meaning / conversation / CS).

## STEP 4 — Build the held-out comprehension test  *(local)*
```bash
python dataset/tools/tunbench.py build --n 300
```
→ `dataset/tunbench_comprehension.jsonl` (slang test the model never trains on).

---

## STEP 5 — Stage 1: CPT on Kaggle  *(~9–10 h — measured; fits the 12 h limit)*
1. kaggle.com → **New Notebook** → **File → Import** `training/cpt_kaggle.ipynb`.
2. **+ Add Data → New Dataset** → upload **`training/cpt.jsonl`**.
3. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON**.
4. **Run All.** Watch the smell-test in the last cell — it should emit *natural Arabizi*.
5. **Output tab → download `tunisian_cpt`** (the adapter). Keep it.

## STEP 6 — Stage 2: SFT on Kaggle  *(~4–6 h)*
1. **+ Add Data**: upload **`tunisian_cpt`** (the folder from Step 5) as a new Dataset, **and**
   `training/sft_real.jsonl`, `dataset/eval_set.jsonl`, `rag/lexicon.jsonl`.
2. Import `training/sft_kaggle.ipynb`, GPU + Internet ON, **Run All.**
3. The eval cell prints **REAL-WORD RATE base → yours** + 8 base-vs-yours samples.
4. **Output tab → download `tunisian_final`.**

---

## STEP 7 — Judge it honestly (the gate)
The SFT notebook already prints **real-word / number-rule / comprehension (base vs yours)** and
writes `/kaggle/working/preds.jsonl` for you. Download it, then locally:
```bash
python dataset/tools/tunbench.py score preds.jsonl
```
Plus the **native A/B you can't skip**: put base vs `tunisian_final` outputs side by side for
50 prompts, blind, and pick which sounds Tunisian.

**SHIP `tunisian_final` only if ALL hold:**
- real_word ≥ base, and ≥ ~90%
- msa_leak ≤ 5%
- number_rule ≥ 95%
- comprehension up vs base
- you win the native A/B ≥ 55%

**If it loses** → don't ship it. Keep base + grounding (already live), scrape more real
conversation (`yt_comments.py` → `build_real_conv.py`), and re-run. You lost only Kaggle hours.

## STEP 8 — Deploy the winner
Point serving at the adapter (same as base, just with weights):
```powershell
modal volume put 7adethni-adapter ./tunisian_final /tunisian_final
modal app stop 7adethni-serving; modal deploy serving/modal_app.py
```
`serving/modal_app.py` auto-detects the adapter (present → base+LoRA, absent → AWQ base).
Grounding (best-of-N + lexicon) stays ON regardless — it protects any model.

---

## Compute budget (all free)
| Stage | Tokens/rows | T4 time | Kaggle sessions |
|---|---|---|---|
| CPT | ~9.4M tokens | ~9–10 h (measured, ~280 tok/s) | 1 (weekly limit is ~30 GPU-h) |
| SFT | ~24k rows | ~4–6 h | ½ |
| Eval | 300–600 prompts | minutes | — |

## Troubleshooting
- **`No module named 'transformers.models.audioflamingo3'`** (or any missing transformers module):
  Kaggle's preinstalled transformers is broken/mismatched. Cell 1 now uninstalls it and pins
  **4.51.3**, then **auto-restarts the kernel** — when it stops, just click **Run All again**.
  (If 4.51.3 ever fails, change `TARGET_TF` in cell 1 to `'4.46.3'`.)
- **`Expected all tensors to be on the same device (cuda:1 and cuda:0)`**: the T4 x2 split the
  model across both GPUs. Fixed — cell 2 now pins `CUDA_VISIBLE_DEVICES='0'` and loads with
  `device_map={'': 0}`. 7B QLoRA fits on one 16 GB T4; the 2nd GPU simply idles.
- **OOM *while loading* the model** (message says ~13 GB already in use): a model from a previous
  failed run is still on the GPU. **Run → Factory reset**, then **Run All**. The notebook now also
  frees leftovers automatically and asserts the GPU is empty before loading.
- **`CUDA out of memory`** *during training*: Qwen2.5's vocab is 151,936, so the loss tensor is
  `seq x batch x 152k x 4 bytes` — that is the memory hog, not the model. Already fixed:
  `MAX_SEQ_LEN=1024`, `BATCH=1`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  **Still OOM?** in order: (1) `MAX_SEQ_LEN=512`, (2) `LORA_R=16`,
  (3) switch `MODEL_NAME` to `'Qwen/Qwen2.5-3B-Instruct'`.
  Note: halving seq length does NOT slow training — same tokens, just more (smaller) steps.
- **`paged_adamw_8bit` error**: set `optim='adamw_torch'`.
- **Loss starts near 0.1** (SFT): you're overfitting — drop to `EPOCHS=1`, `LR=5e-5`, or add data.
- **CPT smell-test is gibberish**: your `cpt.jsonl` was too small/noisy — process more corpus, re-run Step 2.
- **Retention cell fails**: Internet must be ON (it pulls the alpaca sample); otherwise it's skipped
  and you lose English retention (acceptable, but reasoning may dip).

## What NOT to do (past traps)
- ❌ Don't grow `cs_pairs.jsonl` (synthetic) — real data only from here.
- ❌ Don't use `clean_darija_english.csv` — it's **Moroccan**, it poisons the dialect.
- ❌ Don't trust the old "dialect rate" — use real-word + comprehension + native A/B.
