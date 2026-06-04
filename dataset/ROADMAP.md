# Dataset Build Roadmap — Tunisian Arabizi storytelling model (v1)

**Goal of v1:** a model that *writes* storytelling / content (jokes, short stories,
social-media-style text, captions) in **Tunisian Arabizi**, reviewed by natives.

**Locked decisions**
- Spelling convention: the normalizer in [../rag/normalizer.py](../rag/normalizer.py) + [../rag/arabizi_rules.json](../rag/arabizi_rules.json) (`7 5 3 9 gh ch th dh`, French vowels).
- Generator: **Claude** (paid). Anchored on real Arabizi few-shot so output stays Tunisian, not MSA.
- Reviewers: you + a few friends (batch review).
- Headline metric: **MSA-leakage rate** (track on every batch).

---

## What we are building (the missing pieces, in priority order)

| # | Artifact | Target v1 | Why |
|---|----------|-----------|-----|
| A | **Instruction/conversation pairs**, storytelling-weighted | **3,000–5,000** accepted | the core that makes it *talk* (report §2.2) |
| B | **Evaluation set** (held-out, never trained on) | **300** gold | honest quality + MSA-leakage tracking (report §2.5) |
| C | **MSA-leakage detector** (lightweight) | classifier/heuristic | the headline metric (report §3) |
| D | **Lexicon variant enrichment** (real Arabizi spellings) | +native spellings | lifts RAG hit-rate above 73% (already scoped) |

Datasets you already have cover the rest (raw corpus ✓, parallel ✓, lexicon seed ✓).

---

## Phases & acceptance gates

### Phase 1 — Foundations  *(no API key needed; uses existing data)*
1. **Style seed bank** — sample clean, real Arabizi short texts from `clean_darija_english.csv`
   (35k) + TUNIZI parquet (50k). These become (a) few-shot exemplars for Claude and
   (b) a spelling reference. → `seed_bank.jsonl`
2. **Generation spec** — the storytelling instruction taxonomy + the Claude prompt template
   (anti-MSA rules, convention-enforced, few-shot from the seed bank). → `generation_spec.md`, `prompt_template.txt`
3. **Pilot 50** — hand-generate 50 pairs, you+friends review, measure accept-rate & MSA-leakage.
   **GATE:** ≥80% accept-rate before scaling. If lower, fix the prompt, not the volume.

### Phase 2 — Bootstrap generation  *(Claude API)*
4. **Generation harness** — batched Claude calls with prompt caching (system+few-shot cached),
   dedup, convention-normalization, auto-filters (MSA-leakage heuristic, length, script). → `generate.py`
5. Generate in **batches of ~300**; auto-filter; queue survivors for review.

### Phase 3 — Native review loop
6. **Review tool** — CLI that shows a pair, accepts `a`/fix `f`/reject `r`, logs reviewer + reason. → `review.py`
7. You + friends review each batch. **GATE per batch:** keep only accepted/fixed; log MSA-leakage %.
8. Repeat 5–7 until **3,000–5,000 accepted**.

### Phase 4 — Evaluation set + metric
9. Hand-build **300 gold** storytelling prompts with native reference answers (separate from A). → `eval_set.jsonl`
10. **MSA-leakage detector** — start heuristic (function-word + script ratio), upgrade to a
    small classifier when there's labeled data. → `msa_leakage.py`

### Phase 5 — Package for training
11. Convert accepted pairs to training formats (chat JSONL), make **train/val/test splits**,
    confirm eval set is fully held out. → `final/`

---

## Directory layout
```
dataset/
  ROADMAP.md            <- this file
  seed_bank.jsonl       <- Phase 1: real Arabizi exemplars
  generation_spec.md    <- Phase 1: storytelling taxonomy + rules
  prompt_template.txt   <- Phase 1: the Claude prompt
  raw_generated/        <- Phase 2: Claude output, unreviewed
  reviewed/             <- Phase 3: accepted + fixed pairs
  eval_set.jsonl        <- Phase 4: held-out gold
  final/                <- Phase 5: train/val/test splits
  tools/                <- generate.py, review.py, msa_leakage.py, build_seed_bank.py
```

## Cross-cutting rules (every phase)
- Run every output through the normalizer convention before storing.
- Track MSA-leakage % per batch; it must trend **down**, never up.
- Strip personal data; keep a `source`/`license` field on every row.
- Never let eval items leak into training.
