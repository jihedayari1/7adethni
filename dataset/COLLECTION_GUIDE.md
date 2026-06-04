# What to collect (hand this to your friends)

**v1 goal:** a model that writes **storytelling / content in Tunisian Arabizi**
(jokes, short stories, anecdotes, captions, proverbs).

We do **NOT** need more raw text, translations, or dictionary entries — we have enough.
We need **native, human-written Arabizi storytelling content**. Even small amounts are gold.

---

## Priority 1 — Native-written instruction → story pairs  ⭐ (the gold)
A prompt + a story/joke/caption answer, **both in Tunisian Arabizi**, written by a native.
Target: **300–500** total across everyone (≈60–100 per person). Quality > quantity.

**Format** (one JSON object per line, file `dataset/collected/native_pairs.jsonl`):
```json
{"instruction": "a7ki 7keya mou9a3a 3la wa7ed mcha l souk w nsa flousou",
 "output": "kan famma wa7ed esmou Sami, mcha l souk bch ychri 5odhra...",
 "type": "story", "author": "yourname"}
```
`type` ∈ `story | joke | caption | proverb | anecdote | dialogue`.

**Rules**
- Output MUST be real Derja Arabizi, the way you'd text a friend — NOT MSA, NOT formal.
- Use the spelling convention: `7 5 3 9 gh ch th dh` (e.g. `7keya`, `5dmt`, `3andi`, `9ahwa`).
- Keep numbers-as-letters consistent; French-style vowels (`ou` not `oo`).
- 1–6 sentences is fine. Natural beats long.

---

## Priority 2 — Raw native storytelling snippets  (no instruction needed)
Just the content itself — jokes, short stories, funny captions, proverbs you know or
that natives wrote publicly. We auto-wrap these into instruction pairs later.
Target: **as many as easy**, file `dataset/collected/native_content.jsonl`:
```json
{"text": "9alou marra wa7ed b5il...", "type": "joke", "source": "self|public"}
```
If from a public page, put the URL in `source` and only copy text that's publicly posted —
no private messages, no personal data (strip names/phones).

---

## Priority 3 (optional) — Spelling-variant lists  (lifts the RAG hit-rate)
For common words, list the ways people actually spell them. Feeds the normalizer + lexicon.
File `dataset/collected/spelling_variants.csv`:
```
canonical,variants,meaning
barcha,"barcha,barsha,barsha,b9ad",a lot
chnowa,"chnowa,chnoua,chniya,chnia",what
```

---

## How much, realistically
| Item | Per person | Total (you + ~3 friends) |
|------|-----------|--------------------------|
| P1 native pairs ⭐ | 60–100 | **300–500** |
| P2 raw snippets | whatever's easy | bonus |
| P3 variants | 20–30 words | bonus |

**Why this matters more than volume:** Claude will generate the bulk (thousands) of pairs,
but it drifts toward MSA. Your 300–500 native pairs are what we use as few-shot exemplars to
*keep Claude Tunisian*, and as part of the held-out eval. Report's words: "even 1,000
genuinely native pairs are gold."

Drop files in `dataset/collected/`. Ping me when a batch is ready and I'll validate +
normalize them automatically.
