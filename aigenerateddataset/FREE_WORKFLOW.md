# Generating the dataset for FREE (Claude website, your subscription)

No API key, no per-token cost. You use the Claude **website** (your Pro/Max plan) to write
the pairs, and a local script cleans + dedups them. Do it in parts, whenever you want.

> API vs subscription: the `--target` / `--estimate` modes of `generate_cs_dataset.py` call
> the paid **API** (needs `ANTHROPIC_API_KEY`). The steps below avoid that entirely.

## Loop (repeat to grow the dataset)
1. **Get the prompt** (once):
   ```powershell
   python aigenerateddataset/generate_cs_dataset.py --web-prompt > web_prompt.txt
   ```
2. **Open `web_prompt.txt`, copy ALL of it, paste into claude.ai.** Claude returns a JSON
   array of ~30 pairs (mixed general + customer-service).
3. **Save Claude's reply** into `aigenerateddataset/inbox/` as a `.txt` or `.json`
   (e.g. `batch1.txt`). Markdown fences / extra prose are fine — the importer handles it.
4. **Import + clean:**
   ```powershell
   python aigenerateddataset/import_pairs.py
   ```
   It filters MSA/Arabic-script/junk, dedups, and appends clean rows to `cs_pairs.jsonl`.
5. **Get more:** in the same Claude chat, say *"30 more, all different topics, same JSON
   format"* → save as `batch2.txt` → import again. Repeat. (Dedup makes overlap harmless.)

## Then review (you + friends)
```powershell
python dataset/tools/review.py --in aigenerateddataset/cs_pairs.jsonl --reviewer jihed
```
`a` accept · `f` fix · `r` reject · `s` skip · `q` quit. Resumable; friends use their own
`--reviewer` name. Accepted/fixed → `dataset/reviewed/accepted.jsonl`.

## Make it stronger later (optional, paid)
- **Apify** scrape → real customer↔shop Q&A (see `dataset/SCRAPING_SPEC.md`), cleaned via
  `dataset/tools/clean_facebook.py`, then reviewed the same way.
- **API automation** → if you ever want hands-off bulk, set `ANTHROPIC_API_KEY` and use
  `generate_cs_dataset.py --target N` (same output file). Optional, not required.

## Tips
- Each website batch ~30 pairs; ~10 batches ≈ 300 pairs to start. Quality > volume.
- Keep asking Claude to **vary topics** so you cover daily life, jokes, advice, football,
  emotions, AND the store intents (price, delivery, returns, order status...).
- Review a sample every few batches; if it drifts to MSA, tighten the prompt.
