<div align="center">

# 7adethni · حدّثني

**An AI writing assistant for Tunisian Derja — written the way Tunisians actually write it: in Arabizi.**

`chnowa el 7keya?` · `3aslema, kifech n3awnek?` · `barcha 7lewa el fekra hedhi 🇹🇳`

</div>

---

## The problem

Around **12 million Tunisians** write their dialect every day in **Arabizi** — Latin letters plus
numerals that stand in for Arabic sounds that have no Latin equivalent:

| 7 | 9 | 3 | 5 | 2 |
|---|---|---|---|---|
| ح | ق | ع | خ | ء |

So *"what's the story?"* is written **`chnowa el 7keya`**, not `شنوة الحكاية`.

Frontier models handle this badly. Ask one to write Tunisian and it returns Modern Standard Arabic
(which Tunisians don't speak casually), Moroccan Darija (a different dialect), or fluent-looking
**invented words**. Tunisian Derja is a genuinely low-resource language: little clean data, no
standard orthography, heavy French/Arabic code-switching, and two scripts in daily use.

## What this repository is

A complete, working pipeline for that problem — data, training, honest evaluation, serving, and a
shipped product. Built solo, on free-tier compute.

```
 raw Tunisian text ──▶ cleaning / dialect filtering ──▶ CPT ──▶ SFT ──▶ TunBench ──▶ serving ──▶ users
       68M tokens          Moroccan rejected,          language  instruction   honest    grounding    corrections
                           PII stripped, deduped        learning   following   metrics    layer       flow back
```

---

## What's actually built

### 📚 Data assets
| Asset | Size | What it is |
|---|---|---|
| **Tunisian lexicon** | **17,307 entries** | Arabizi spelling variants + Arabic script + English/French gloss + real example sentence per entry |
| **Cleaned corpus** | **553k lines / ~68M tokens** | Real Tunisian social text: deduplicated (MinHash), dialect-bucketed, PII-stripped |
| **Instruction pairs** | **21,943** | Built from *real* sources — translation, slang comprehension, conversation |
| **TunBench** | **600 items** | Held-out slang benchmark, contamination-verified (0 leaks) |

### 🔤 Arabizi language tooling
- **Deterministic transliterator** (Arabic ⇄ Arabizi) built from a published Tunisian Arabizi
  convention table — the number↔letter rules are enforced in *code*, not hoped for in weights.
- **Spelling-variant normalizer**: `barcha` = `barsha` = `barchaaa` collapse to one key. Arabizi has
  no standard orthography, so every lookup goes through this.
- **Hybrid retriever**: exact + fuzzy match with a consonant-skeleton gate (Arabizi vowels are
  unstable — `flouss`/`flus`/`floos` are the same word).

### 🎓 Training (`training/azure`, runs on any GPU)
QLoRA **CPT → SFT** on Qwen2.5-7B, following the dual-script recipe that works for Arabizi
(train on Arabic script *and* Arabizi *and* explicit transliteration pairs).

Built for results you can defend:
- held-out **validation split** → `eval_loss` (overfitting is visible, not guessed)
- **best-checkpoint selection** + early stopping
- **greedy decoding** at eval time → reproducible numbers
- **ablation mode** (`--cpt none`) → proves whether the CPT stage actually earned its cost
- **paired bootstrap confidence intervals** → answers *"is this gap real or is it noise?"*
- `manifest.json` per run: config + data SHA + metrics + hardware

### 📊 TunBench — honest evaluation
The project's first metric was a **vanity metric**: it scored 96% on output that was, in the
author's own words, *"arabizi but many words have no meaning."* It counted Tunisian function words
and never checked whether the words existed. It was replaced:

| Metric | What it catches |
|---|---|
| **real-word rate** | invented words, scored against the 17k lexicon |
| **comprehension** | does it actually know the slang (600 held-out items) |
| **number-rule** | valid Arabizi digits only (`2 3 5 7 8 9`) |
| **MSA leakage** | drifting out of dialect |

Plus a **blind native A/B** — the one thing statistics can't replace.

### 🔌 Serving with a grounding layer (`serving/`)
Model output is never trusted blindly:
- **best-of-N**: generate candidates, keep the one with the highest real-word rate
- **lexicon check**: flag out-of-vocabulary (invented) words at inference time
- **input glossing**: inject confident meanings for slang in the user's message
- **number-rule canonicalization**

This runs on *any* model — a fine-tune or the base — so quality never depends on training alone.

### 🧩 Product (`extension/`, `website/`, `backend/`)
A Chrome extension (translate · reply · rewrite, with tone selection), a landing site, and a
FastAPI gateway with quota + a **data flywheel**: when a user edits a suggestion before copying it,
that edit is captured as a native-speaker correction and becomes future training data.

---

## The finding I'm most proud of

The first fine-tune **made the model worse**, and the metrics said it got better.

Training loss started at **0.13** — a fresh fine-tune should start near 2.0–3.0. The model was
memorizing synthetic data, not learning. Meanwhile the "dialect rate" metric read 96% before *and*
after training, because the base model already passed its shallow test.

The response was to throw the metric out, build TunBench, verify contamination (1,584 lexicon
entries / 5,245 word forms excluded from training, 0 leaks), and rebuild the data pipeline around
*real* text. The current CPT run starts at **loss 3.09 → 2.16**, which is what learning looks like.

**Negative results are kept in this repo on purpose.** A pipeline that can't detect its own failures
isn't a pipeline.

---

## Repository map

```
rag/            Arabizi normalizer, transliterator, 17k lexicon, hybrid retriever
corpus/         3GB raw corpus -> cleaned, deduped, dialect-bucketed text
dataset/tools/  scrapers (YouTube API), PII scrubbing, toxicity filter, TunBench
training/       CPT + SFT data builders
  azure/        result-trusted training suite (validation, ablation, bootstrap CIs)
  modal/        serverless-GPU runner for the same suite
serving/        model server + grounding layer (best-of-N, lexicon QC)
backend/        gateway: quota, usage, the correction flywheel
extension/      Chrome extension (MV3)
website/        landing page
```

## Quick start

```bash
pip install -r training/azure/requirements.txt

# 1. build the training data from the raw corpus
python corpus/process_corpus.py
python training/build_cpt_corpus.py --tokens 20000000
python training/build_sft_real.py
python dataset/tools/tunbench.py build --n 600

# 2. validate the config without spending GPU time
python training/azure/train_cpt.py --dry-run

# 3. train
python training/azure/train_cpt.py --out outputs/tunisian_cpt
python training/azure/train_sft.py --cpt outputs/tunisian_cpt --out outputs/tunisian_final
python training/azure/train_sft.py --cpt none --out outputs/sft_only        # ablation

# 4. judge it honestly
python training/azure/evaluate.py \
    --adapters base=none final=outputs/tunisian_final sft_only=outputs/sft_only
```

## Honest limitations

- **Natively-written Arabizi is the bottleneck, not compute.** Only ~240k tokens exist in the
  corpus; it is upsampled to ~11% of the CPT mix. Scaling the corpus *past* ~20M tokens lowers that
  share and biases output toward transliterated, vowel-poor style — the builder warns when this happens.
- The 101-prompt generation eval has ±5.9 points of noise. Decisions weight the 600-item
  comprehension benchmark instead.
- Scraped conversation data is PII-stripped and **kept out of this repository**.
- No trained weights are published yet — training is in progress.

## License & data ethics

Code: MIT. Scraped text is used for model training only, PII-stripped
(`@mentions`, phone numbers, emails, URLs), never redistributed. Moroccan/Algerian Darija is
explicitly filtered out — this is a **Tunisian** model, and dialect contamination was an early,
costly mistake.

---

<div align="center">

**Built in Tunisia 🇹🇳 — because the way we write deserves a model that understands it.**

</div>
