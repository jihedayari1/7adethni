# Tunisian Arabizi RAG layer

Implements report §3 (normalizer) and §4 (RAG) over the data already in this folder.
Build order followed: **lexicon → normalizer → hybrid retriever → prompt injection → eval.**

## Files
| File | Role |
|------|------|
| `arabizi_rules.json` | The number→character rules extracted from the PDF (Table 1), native-corrected. |
| `normalizer.py` | Canonical matching key (`barcha`=`barsha`=`barchaaa`) + Arabic→Arabizi transliterator + variant generator. **Reused at training time and at RAG query time.** |
| `build_lexicon.py` | Builds `lexicon.jsonl` from `derja-english.csv` (18k) + `chunk_16.json`. |
| `lexicon.jsonl` | 17,307 entries in the report's §2.4 schema, with generated `arabizi_variants`. |
| `retriever.py` | Hybrid retriever: exact-normalized + fuzzy (consonant-skeleton gated) + optional semantic backstop. Detects slang tokens, builds the `Meanings:` block. |
| `demo_prompt.py` | End-to-end §4.4 pipeline: assembles the augmented prompt to send the base model. |
| `eval_retrieval.py` | Retrieval hit-rate on a held-out slang set (§4.5). |

## Run
```bash
python rag/build_lexicon.py      # (re)build lexicon.jsonl
python rag/retriever.py          # smoke test
python rag/eval_retrieval.py     # hit-rate
python rag/demo_prompt.py        # see the injected prompt
python rag/retriever.py --semantic   # enable embedding backstop (downloads a model)
```

## The Arabizi spelling convention (locked — report §3, "Decision 1")
Number↔letter rules are in `arabizi_rules.json`. Canonical (Tunisian / French-background):

| sound | canonical | also accepted |
|-------|-----------|---------------|
| ح | `7` | |
| خ | `5` | `kh`, `7'` |
| ع | `3` | |
| ق | `9` | `k`, `q` |
| غ | `gh` | `4'`, `8` |
| ش | `ch` | `sh`, `$` |
| ث | `th` | |
| ذ | `dh` | |
| ء (hamza) | `2` | |

Vowels: French-style — `ou`→/u/, `ch` not `sh`. The normalizer folds the "also accepted"
column into the canonical form so retrieval matches regardless of how the user typed it.

## Honest limitations (so the next step is clear)
1. **Variants are generated from Arabic script, which omits short vowels** → keys like
   `برشة`→`brcha` (missing the `a` in `barcha`). Skeleton-gated fuzzy matching covers most of
   this, but it is the main source of the ~25% miss rate. **Fix:** harvest real Arabizi
   spellings from `clean_darija_english.csv` (35k native sentences) and the TUNIZI parquet
   (50k) and add them to `arabizi_variants` — this is the single biggest quality lever here.
2. **Homograph collisions** (حافلة bus vs حفلة party both → `7afla`). Needs native review.
3. RAG supplies *meanings only*. It does **not** make the model fluent — that still requires
   the instruction/conversation fine-tuning set (report §2.2), which is still missing.
