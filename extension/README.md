# 7adethni — Chrome extension (MVP skeleton)

Helps Tunisians write **posts, replies, captions, rewrites, and translations** in natural
Tunisian Derja (Arabizi). It calls your `serving/` API which runs the fine-tuned model.

## Run it locally (dev)
1. Start the model API (see `../serving/app.py`) on a GPU machine — note its URL (e.g. `http://localhost:8000`).
2. Chrome → `chrome://extensions` → enable **Developer mode** → **Load unpacked** → select this `extension/` folder.
3. Click the 7adethni icon → ⚙️ → set **API URL** to your API → Save.
4. Pick a feature + tone, type your input, hit **Génère bel tounsi ✨**.

## Files
- `manifest.json` — MV3 config (restrict `host_permissions` to your API domain before release).
- `popup.html/.css/.js` — the UI (feature + tone + input → calls `POST /generate` → shows output + copy).

## Notes
- Add an `icon128.png` (any 128×128 logo) before publishing to the Chrome Web Store.
- The API URL + key are stored per-user in `chrome.storage` (set via ⚙️).
- v0.2 ideas: right-click context menu + in-page injection (write directly into the focused text
  box on Facebook/Instagram), an Android keyboard (IME), and a "thumbs up/fix" button that sends
  corrections back to you (the data flywheel).
