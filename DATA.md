# Data notes

## Raw corpus chunks
`chunk_1.json` … `chunk_15.json` (~200 MB each, ~3 GB total) exceed GitHub's 100 MB
per-file limit, so they are committed **zipped**: `chunk_N.json.zip` (~20–45 MB each).

To restore the raw JSON locally:
```bash
python -c "import zipfile,glob; [zipfile.ZipFile(z).extractall() for z in glob.glob('chunk_*.json.zip')]"
```
Each archive contains one JSON array of `{\"text\": ...}` records (raw monolingual Tunisian
social-media text, mostly Arabic script with French/English code-switching).

`chunk_16.json` (3 MB) is committed uncompressed — it is an EN→Tunisian dictionary, not raw text.

## Other datasets (committed as-is)
- `derja-english.csv` — 18k Arabic-script Tunisian lexicon entries (POS, french, example).
- `clean_darija_english.csv` — 35k native Arabizi↔English parallel sentences.
- `train-00000-of-00001.parquet` — TUNIZI: 50k Arabizi tweets + sentiment label.

## Built artifacts
- `rag/` — Arabizi normalizer, RAG lexicon (17k entries), hybrid retriever. See `rag/README.md`.
- `dataset/` — dataset build roadmap, style seed bank, collection guide. See `dataset/ROADMAP.md`.
