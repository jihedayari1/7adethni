# 7adethni — Arabizi writing assistant: MVP spec & build plan

**Product:** a tool that helps everyday Tunisians **write** in natural Derja/Arabizi — posts,
replies, captions, rewrites, translations. Human-in-the-loop (assist, the user edits), so the
model's current coherence is *good enough* to launch. Plays directly to our proven edge.

## 1. MVP feature set (v0.1)
| Feature | Input | Output | Why |
|---|---|---|---|
| **Write a post** | a topic | Arabizi social post (tone-selectable) | the #1 daily need |
| **Reply to a message** | the incoming message | suggested Arabizi reply | DMs, comments |
| **Rewrite / improve** | your rough text | cleaner/funnier/shorter Arabizi | polish |
| **Translate to Derja** | FR/EN/Arabic text | natural Tunisian Arabizi | the model's home run |
| **Caption** | a photo idea / description | short catchy Arabizi caption | IG/FB |
| **Ask / free text** | anything | Arabizi answer | catch-all |

**Tones:** 3adi · morfeh (funny) · rasmi (formal) · promo · 7anin (warm) · 9sir (short).

## 2. UX (v0.1 = popup; v0.2 = in-page)
- v0.1: toolbar **popup** (built in `extension/`): feature + tone + input → output + copy.
- v0.2: right-click **context menu** + inject directly into the focused text box on FB/IG/WhatsApp
  web; a "✅ good / ✏️ fix" button under each suggestion → **sends corrections back to you**.
- v0.3: **Android keyboard (IME)** — write Arabizi anywhere on the phone (where Tunisians actually type).

## 3. Architecture
```
Chrome extension  --HTTPS-->  7adethni API (serving/app.py, on a GPU)  -->  fine-tuned model (base + LoRA)
   popup.js                      FastAPI /generate                          Qwen + your adapter
```
- `serving/prompts.py` — feature → instruction templates (one source of truth).
- `serving/app.py` — FastAPI, loads base+LoRA once, `POST /generate {feature,text,tone}`, API-key gated.
- Extension stores API URL + key per-user in `chrome.storage`.

## 4. How to serve the model (pick one)
| Option | Cost | Good for |
|---|---|---|
| **RunPod / Vast.ai GPU pod** (run `serving/app.py`) | ~$0.2–0.5/hr | full control, cheapest for steady use |
| **Modal / Replicate / HF Endpoints** (serverless GPU) | pay-per-call, scale-to-zero | beta with spiky traffic, no ops |
| **Quantize to GGUF + llama.cpp on a cheap VPS/CPU** | very cheap | low volume, no GPU |
For the beta: a single RunPod pod or a Modal function is enough. Put the API behind HTTPS + a key.

## 5. Monetization
- **Freemium:** ~15 free generations/day → subscription (local price, e.g. a few DT/month).
- **Pro/Business tier:** higher limits + brand voice for influencers / community managers / shops.
- Later: **API/engine tier** — sell the Tunisian Arabizi engine to agencies & platforms (incl. CS).

## 6. Build plan (realistic, solo)
- **Week 1:** retrain model on 7B/8B + more reviewed pairs (coherence). Stand up `serving/app.py` on RunPod behind HTTPS+key.
- **Week 2:** finish the popup extension (done as skeleton), add usage limits, polish copy. Private beta with ~10 friends.
- **Week 3:** v0.2 — context-menu + in-page inject + the **fix-it feedback** button (data flywheel). Add simple auth + per-user quota (e.g. a tiny backend with Supabase/Firebase).
- **Week 4:** public beta to the Tunisian tech/social community; collect testimonials; turn on freemium.
- **Month 2+:** Android keyboard; Pro tier; pitch the engine/API to businesses.

## 7. Metrics to watch
- Activation (first generation), retention (D7 weekly active), generations/user/day.
- **Edit-rate** (how much users change the output) — your model-quality + data-flywheel signal.
- Free→paid conversion.

## 8. Risks & answers
- **Generic LLMs improve at dialects** → stay ahead on *Tunisian specifically* + UX + the data flywheel.
- **B2C monetization is hard** → keep build cheap, lean on virality + a business tier for revenue.
- **Coherence** → it's an *assist* tool (user edits) + you'll keep improving the model; far lower bar
  than autonomous customer service.
- **Serving cost** → scale-to-zero serverless for beta; quantize; cache common requests.

## 9. The moat (repeat)
Best Tunisian Arabizi **model** + a **data flywheel** from real users + the **"Tunisian AI" brand**.
The extension is the beachhead that builds all three — then everything else (business/API) follows.
