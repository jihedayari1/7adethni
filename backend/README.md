# 7adethni backend (gateway)

Cheap CPU service between the extension and the GPU model server. Adds freemium quota, usage
logging, and the **feedback/data-flywheel** store. SQLite — no external DB for the MVP.

## Run
```bash
pip install -r requirements.txt
export MODEL_API_URL="http://localhost:8000"   # the serving/app.py (GPU) endpoint
export MODEL_API_KEY="testkey123"              # optional; must match serving
export FREE_DAILY_LIMIT=15
uvicorn app:app --host 0.0.0.0 --port 9000
```

## Endpoints (what the extension calls)
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/generate` | `{device_id, feature, text, tone}` | `{request_id, output, plan, used, limit, remaining}` (429 when quota hit) |
| POST | `/feedback` | `{request_id, rating:"good"|"bad", corrected?}` | `{ok:true}` — **the data flywheel** |
| GET | `/me?device_id=…` | — | quota status |
| GET | `/health` | — | status |

## How the extension uses it
- On first run, generate a random **`device_id`** (UUID) and save it in `chrome.storage` (no login for MVP).
- Send it on every `/generate`. Show `remaining` as the quota chip.
- After each output, the 👍/✏️ buttons call `/feedback` (a ✏️ "fix" sends the user's corrected text →
  these corrections become future training data).

## Data flywheel
`feedback.corrected` + the `usage` table = real Tunisian Arabizi + human corrections. Periodically
export the high-quality ones, native-review, and fold into the training set → the model keeps
improving in a way no competitor using generic LLMs can match.

## Architecture
```
extension  ──>  backend (this, CPU, always-on)  ──>  serving/app.py (GPU, can scale-to-zero)
```
Keep them separate: the gateway is cheap and always up; the GPU model server can be a RunPod pod
or a serverless GPU (Modal/Replicate) it calls.
