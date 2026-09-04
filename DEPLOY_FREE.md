# 7adethni — zero-cash deployment (everything on free tiers)

**Launch model = base Qwen2.5-7B (no training) + grounding.** Your own Kaggle run proved it:
the BASELINE (pre-fine-tune) outputs scored 96% dialect rate and read *better* than the
fine-tuned ones. Users edit before copying anyway — their edits become the training data.

## Stack ($0/month)
| Piece | Platform | Cost |
|---|---|---|
| Model server (7B-AWQ, GPU) | Modal, scale-to-zero | free monthly credits (~$30) |
| Gateway + flywheel DB | Modal CPU + Volume | same credit pool, pennies |
| Website | GitHub Pages | $0 |
| Extension (soft launch) | zip + chrome://extensions "Load unpacked" | $0 ($5 store fee later) |

## Deploy — one time, in order (~30 min)

```bash
# 0) Modal account (sign in with GitHub; free Starter plan has monthly credits)
pip install modal
modal setup

# 1) model server
modal secret create 7adethni-keys MODEL_API_KEY=pick-any-secret-string
modal deploy serving/modal_app.py
#    -> COPY the printed URL, e.g. https://you--7adethni-serving-server-api.modal.run

# 2) gateway (uses the URL from step 1)
modal secret create 7adethni-gateway MODEL_API_URL=<url-from-step-1> MODEL_API_KEY=<same-secret>
modal deploy backend/modal_backend.py
#    -> COPY the printed gateway URL — this is your public API

# 3) smoke test
curl <gateway-url>/health
curl -X POST <gateway-url>/generate -H "Content-Type: application/json" \
     -d '{"device_id":"test1","feature":"translate","text":"Comment ça va mon ami ?"}'
# first call is a cold start (~1-2 min: model download+load); repeat — warm calls take seconds

# 4) point the clients at it
#    extension: ⚙️ settings -> API URL = <gateway-url>
#    website:   website/app.js -> BACKEND_URL = '<gateway-url>'  (then push to GitHub Pages)
```

## Soft-launch the extension with $0
Zip the `extension/` folder and share it; testers do:
chrome://extensions → enable **Developer mode** → **Load unpacked** → select the unzipped folder
→ open the popup → ⚙️ → paste the gateway URL. (Chrome Web Store's one-time $5 fee comes later,
at public launch.)

## Attach your trained model (after training v2 passes the gate)
No redeploy of code needed — just push the adapter to the volume and restart:
```powershell
# tunisian_final = the Stage-2 adapter you downloaded from Kaggle (~100-300MB)
modal volume put 7adethni-adapter ./tunisian_final /tunisian_final
modal app stop 7adethni-serving; modal deploy serving/modal_app.py   # (';' works in PowerShell too)
curl <serving-url>/health        # -> "adapter": true, model ...+tunisian_final
```
The server auto-detects the adapter: present → serves **fp16 base + LoRA**, absent → serves the AWQ base.
(The adapter was trained on a 4-bit base — serving it on fp16 is the standard QLoRA practice; expect a
tiny, usually positive, quality delta. First adapter boot downloads the fp16 base ≈15GB once into the cache.)
Roll back: `modal volume rm -r 7adethni-adapter /tunisian_final` then redeploy. Grounding stays on either way.

## Watch the flywheel
- `GET <gateway-url>/stats` — installs, weekly generations, **weekly copies (north star)**, corrections.
- Backup/export the DB: `modal volume get 7adethni-db 7adethni.db ./backup.db`
- Turn interactions into datasets: `python dataset/tools/export_flywheel.py ./backup.db`

## Credit math (why this stays free)
A10G ≈ $1.10/hr **only while generating** (scale-to-zero, 4-min idle window). A generation is a
few seconds → the free credits cover thousands of generations/month. If credits ever run short:
shrink `scaledown_window`, or drop to `Qwen2.5-3B-Instruct` on a T4 (~half the rate) in
`serving/modal_app.py`.

## Plan B (if Modal signup demands a card)
Serve from Kaggle (30 free GPU-hrs/week): run `serving/app.py` in a notebook + a `cloudflared`
tunnel, paste the tunnel URL as MODEL_API_URL. Works, but the notebook dies every ~9h —
demo-mode only, not always-on.
