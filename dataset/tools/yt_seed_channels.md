# YouTube seed list — Tunisian Arabizi comment mining

Goal: feed `yt_comments.py` sources whose **comment sections are dominated by Tunisian Arabizi**
(Latin letters + numbers), then pipe through `build_real_conv.py`. You're Tunisian — treat the
named channels as starting points to **recognize & verify**; the ready-to-run `--search` queries
need no IDs (the API resolves them live).

## Fastest start — search queries (no IDs needed)
```bash
export YOUTUBE_API_KEY=...
python dataset/tools/yt_comments.py --search "podcast tounsi"        --max 30
python dataset/tools/yt_comments.py --search "rap tunisien"          --max 30
python dataset/tools/yt_comments.py --search "vlog tunisie"          --max 30
python dataset/tools/yt_comments.py --search "tunisie reaction"      --max 30
python dataset/tools/yt_comments.py --search "stand up comedy tounsi" --max 30
python dataset/tools/yt_comments.py --search "kora tounsia"          --max 30   # football talk
```
Then: `python dataset/tools/build_real_conv.py aigenerateddataset/yt_threads.jsonl`

## Best categories — MEASURED yield (clean pairs ÷ comment→reply pairs)
Measured on a real pull (2026-07). **Yield swings 3–13% by source — source selection dominates.**
| Category | Measured yield | Note |
|---|---|---|
| **Gaming** ⭐ | **~13%** | Young audience writes natively in Arabizi (Latin) → best targets |
| Comedy / sketches | ~6% | Mixed; lots of Arabic-script |
| **Rap / music** | ~3.6% | Huge comment volume but mostly Arabic-script + no-reply praise (few threads) |
| Tech / vlog / how-to | (try) | Same young-Arabizi audience as gaming — likely high |

Most comments get dropped because they're **Arabic-script** (we need *natively* Arabizi replies, not
transliterated-raw), or non-Tunisian/French, or profanity/spam. So **prefer gaming / tech / vlog / how-to
creators** and **scale up video count** (~10 clean pairs per video at 13% yield).
Use `--cap N` to limit comments fetched per video (default 400) for faster, more diversified pulls.

Search seeds: `"gaming tunisie"`, `"tech tunisie unboxing"`, `"vlog tunisie"`, `"cuisine tunisienne"`,
`"football tunisien"`, `"podcast tounsi"`, `"tutoriel tunisie"`.

## Recognizable creators to try (verify the handle)
The tool accepts `@handle` and resolves it — if a handle is wrong it errors cleanly, just fix it.
```bash
# Music (huge, very Arabizi comment sections):
python dataset/tools/yt_comments.py --channel @BaltiOfficiel  --max 15
python dataset/tools/yt_comments.py --channel @KlayBBJ        --max 15
#   others to recognize/verify: Samara, A.L.A, Nordo, Hamzaoui Med Amine, Sanfara, Kaso, Ghastone
# Podcasts / talk:
#   recognize/verify: Brainstorm Tunisia, Politichno, Tasart, Ettounsi podcast
# Comedy / creators:
#   recognize/verify: Wled Moufida (clips), Gtaa wel Hkeya, Cobra, Bahbouh
```
> Don't have the exact handle? Use `--search "<creator name>"` and the API finds the channel.

## Picking good videos (quality > quantity)
- Prefer videos with **many comments** (more reply chains = more conversational pairs).
- Skip videos whose comments are mostly **Arabic-script or pure French** — the pipeline keeps the
  Tunisian/Arabizi ones, but you waste API quota.
- 20–30 videos across 2–3 categories is plenty for a first real batch.

## After scraping
`build_real_conv.py` PII-strips, **rejects Moroccan/Algerian**, transliterates any Arabic, and keeps
only replies that score ≥60% real words vs the lexicon. Inspect `real_conv_pairs.jsonl`, then upload
it with your Kaggle dataset — the notebook folds it into the conversational pool automatically.

⚠️ Raw `yt_threads.jsonl` holds **unstripped comment text** (may contain @mentions, phone numbers
people post) — **keep it local / gitignored**. Only the cleaned `real_conv_pairs.jsonl`
(PII-stripped) is safe to keep.
