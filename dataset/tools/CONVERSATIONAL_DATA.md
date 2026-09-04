# Real conversational data — the coherence path

Synthetic Claude-Arabizi hit a ceiling (fluent-looking gibberish). Real human comment→reply
threads are what teach **coherence**. This is the pipeline that turns them into training pairs.

```
  source (real threads)                build_real_conv.py                 training
  ─────────────────────   ───────────────────────────────────────────   ──────────
  YouTube  (yt_comments)  PII-strip → dialect-gate → transliterate    →  real_conv_pairs.jsonl
  Facebook (clean_facebook)  → quality-filter (lexicon) → structure       (folded into the
  Reddit / any JSON          into "Jaweb 3al message: …" reply pairs       conversational pool)
```

## Pick a source (Facebook is walled — don't bother)
| Source | Tool | Why |
|---|---|---|
| **YouTube comments** ⭐ | `yt_comments.py` | Real, conversational, usually **already in Arabizi** (naturally vowelled). Official API, free quota. Best targets. |
| Facebook export | `clean_facebook.py` → feed its pairs | If you obtain an export (Apify/official API). PII-stripped. |
| Reddit r/Tunisia, any dump | generic JSON → `build_real_conv.py` | `ingest()` accepts flat pairs or nested threads. |

## Run it
```bash
# 1) scrape (YouTube example) — needs an API key, see yt_comments.py header
export YOUTUBE_API_KEY=...
python dataset/tools/yt_comments.py --channel UCxxxx --max 25      # -> aigenerateddataset/yt_threads.jsonl

# 2) clean + transliterate + structure into training pairs
python dataset/tools/build_real_conv.py aigenerateddataset/yt_threads.jsonl
#    -> aigenerateddataset/real_conv_pairs.jsonl

# 3) upload real_conv_pairs.jsonl with your Kaggle dataset; the notebook folds it in automatically.
```
`python dataset/tools/build_real_conv.py` with no args runs a built-in DEMO so you can see the
transform (Tunisian kept, Moroccan rejected, PII stripped, Arabic→Arabizi) before scraping anything.

## What the pipeline guarantees
- **PII removed**: @mentions, emails, URLs, phone numbers (prices/emojis kept). *Never push raw scrapes to public git.*
- **Tunisian only**: rejects Moroccan/Algerian markers (`bghit`, `daba`, `dyal`, `wakha`, `bzaf`…).
- **Always Arabizi output**: Arabic-script → deterministic transliteration (`rag/normalizer.py`).
- **Real words**: each kept reply must score ≥ 60% real-word rate vs the 17k lexicon.

## One quality note
Prefer sources **already written in Arabizi** (YouTube/Reddit). Transliterating *Arabic-script*
text gives correct-but-**raw** Arabizi (vowels dropped: `wallh nfs al7kaia`), because Arabic
omits short vowels. Natural Arabizi is more readable — so it makes better generation targets.
