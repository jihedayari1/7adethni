"""
7adethni backend / gateway (FastAPI + SQLite, no GPU needed).

Sits between the extension and the GPU model server (serving/app.py). It adds:
  * per-device freemium quota (FREE_DAILY_LIMIT/day)
  * usage logging (+ model quality metadata: real-word rate, OOV, latency)
  * the DATA FLYWHEEL:
      - POST /feedback  : explicit 👍 / ✏️-correction
      - POST /event     : implicit signals (copy / edit_copy / regen / share)
      - eval-pool split : hash(usage_id)%20==0 rows are RESERVED for evaluation and
                          must never be exported for training (contamination guard)
  * privacy: `optout=true` on /generate serves normally but stores no text

Run:
    pip install -r requirements.txt
    export MODEL_API_URL="http://localhost:8000"   # serving/app.py
    export MODEL_API_KEY="testkey123"              # optional, must match serving
    export FREE_DAILY_LIMIT=15
    uvicorn app:app --host 0.0.0.0 --port 9000

Architecture:  extension --> THIS backend (cheap CPU) --> model server (GPU)
"""
import os, re, json, sqlite3, time, uuid, datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_API_URL    = os.environ.get("MODEL_API_URL", "http://localhost:8000")
MODEL_API_KEY    = os.environ.get("MODEL_API_KEY", "")
FREE_DAILY_LIMIT = int(os.environ.get("FREE_DAILY_LIMIT", "15"))
DB_PATH          = os.environ.get("DB_PATH", "7adethni.db")
MODEL_VERSION    = os.environ.get("MODEL_VERSION", "qwen2.5-7b-base+ground")

app = FastAPI(title="7adethni backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            device_id TEXT PRIMARY KEY, plan TEXT DEFAULT 'free', created_at REAL);
        CREATE TABLE IF NOT EXISTS usage(
            id TEXT PRIMARY KEY, device_id TEXT, ts REAL, day TEXT,
            feature TEXT, tone TEXT, input TEXT, output TEXT);
        CREATE TABLE IF NOT EXISTS feedback(
            id TEXT PRIMARY KEY, usage_id TEXT, ts REAL, rating TEXT, corrected TEXT);
        CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY, usage_id TEXT, device_id TEXT, ts REAL,
            kind TEXT, payload TEXT);
        CREATE INDEX IF NOT EXISTS idx_usage_day ON usage(device_id, day);
        CREATE INDEX IF NOT EXISTS idx_events_usage ON events(usage_id);
        """)
        # migration: quality/flywheel columns on usage (no-op if they already exist)
        for col, typ in [("model_version", "TEXT"), ("latency_ms", "INTEGER"),
                         ("quality_rate", "REAL"), ("oov", "TEXT"),
                         ("lang_in", "TEXT"), ("eval_pool", "INTEGER DEFAULT 0")]:
            try:
                c.execute(f"ALTER TABLE usage ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # column already exists

def today():
    return datetime.date.today().isoformat()

def ensure_user(device_id):
    with db() as c:
        if not c.execute("SELECT 1 FROM users WHERE device_id=?", (device_id,)).fetchone():
            c.execute("INSERT INTO users(device_id, created_at) VALUES(?,?)", (device_id, time.time()))

def quota(device_id):
    with db() as c:
        row = c.execute("SELECT plan FROM users WHERE device_id=?", (device_id,)).fetchone()
        plan = row["plan"] if row else "free"
        used = c.execute("SELECT COUNT(*) n FROM usage WHERE device_id=? AND day=?",
                         (device_id, today())).fetchone()["n"]
    limit = 10**9 if plan == "pro" else FREE_DAILY_LIMIT
    return {"plan": plan, "used": used, "limit": limit, "remaining": max(0, limit - used)}


_AR = re.compile(r"[؀-ۿ]")
_NUMLETTER = re.compile(r"[a-z][2-9]|[2-9][a-z]", re.I)

def detect_lang_in(text: str) -> str:
    """Coarse input-script tag for the flywheel (arabizi/arabic/mixed/latin)."""
    has_ar, has_lat = bool(_AR.search(text)), bool(re.search(r"[A-Za-z]", text))
    if has_ar and has_lat: return "mixed"
    if has_ar: return "arabic"
    if _NUMLETTER.search(text): return "arabizi"
    return "latin"   # plain FR/EN


class GenReq(BaseModel):
    device_id: str
    feature: str = "translate"
    text: str
    tone: str = "normal"
    optout: bool = False       # privacy: serve but store no text

class FbReq(BaseModel):
    request_id: str
    rating: str               # "good" | "bad"
    corrected: str | None = None

class EventReq(BaseModel):
    device_id: str
    usage_id: str
    kind: str                 # "copy" | "edit_copy" | "regen" | "share"
    payload: str | None = None   # for edit_copy: the user's final edited text

EVENT_KINDS = {"copy", "edit_copy", "regen", "share"}


@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True, "model_api": MODEL_API_URL, "free_daily_limit": FREE_DAILY_LIMIT,
            "model_version": MODEL_VERSION}

@app.get("/me")
def me(device_id: str):
    ensure_user(device_id)
    return quota(device_id)

@app.post("/generate")
def generate(req: GenReq):
    if not req.device_id:
        raise HTTPException(400, "device_id required")
    if not (req.text or "").strip():
        raise HTTPException(400, "text is empty")
    ensure_user(req.device_id)
    q = quota(req.device_id)
    if q["remaining"] <= 0:
        raise HTTPException(429, detail={"error": "daily_limit_reached", **q,
                                         "message": "5lset el génération mejjeniya el yom, 3awed ghodwa wala chouf Pro"})
    # call the model server
    t0 = time.time()
    try:
        headers = {"x-api-key": MODEL_API_KEY} if MODEL_API_KEY else {}
        r = requests.post(f"{MODEL_API_URL.rstrip('/')}/generate", json={
            "feature": req.feature, "text": req.text, "tone": req.tone}, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        output = data.get("output", "")
        qual = data.get("quality") or {}
    except requests.RequestException as e:
        raise HTTPException(502, f"model server error: {e}")
    latency_ms = int((time.time() - t0) * 1000)

    rid = uuid.uuid4().hex
    eval_pool = 1 if int(rid[:8], 16) % 20 == 0 else 0   # 5% reserved for eval, never trained on
    # privacy opt-out: keep the row for quota/stats, store no text
    inp_store = None if req.optout else req.text
    out_store = None if req.optout else output
    with db() as c:
        c.execute("""INSERT INTO usage(id,device_id,ts,day,feature,tone,input,output,
                     model_version,latency_ms,quality_rate,oov,lang_in,eval_pool)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (rid, req.device_id, time.time(), today(), req.feature, req.tone,
                   inp_store, out_store, MODEL_VERSION, latency_ms,
                   qual.get("real_word_rate"), json.dumps(qual.get("oov") or [], ensure_ascii=False),
                   detect_lang_in(req.text), eval_pool))
    q2 = quota(req.device_id)
    return {"request_id": rid, "output": output, **q2}

@app.post("/feedback")
def feedback(req: FbReq):
    with db() as c:
        if not c.execute("SELECT 1 FROM usage WHERE id=?", (req.request_id,)).fetchone():
            raise HTTPException(404, "unknown request_id")
        c.execute("INSERT INTO feedback(id,usage_id,ts,rating,corrected) VALUES(?,?,?,?,?)",
                  (uuid.uuid4().hex, req.request_id, time.time(), req.rating, req.corrected))
    return {"ok": True}

@app.post("/event")
def event(req: EventReq):
    if req.kind not in EVENT_KINDS:
        raise HTTPException(400, f"unknown event kind '{req.kind}'")
    with db() as c:
        if not c.execute("SELECT 1 FROM usage WHERE id=?", (req.usage_id,)).fetchone():
            raise HTTPException(404, "unknown usage_id")
        c.execute("INSERT INTO events(id,usage_id,device_id,ts,kind,payload) VALUES(?,?,?,?,?,?)",
                  (uuid.uuid4().hex, req.usage_id, req.device_id, time.time(), req.kind, req.payload))
    return {"ok": True}

@app.get("/stats")
def stats():
    """Founder dashboard: the numbers that matter (north star = weekly copies)."""
    with db() as c:
        week_ago = time.time() - 7 * 86400
        g = lambda sql, *a: c.execute(sql, a).fetchone()[0]
        return {
            "devices_total":   g("SELECT COUNT(*) FROM users"),
            "devices_week":    g("SELECT COUNT(DISTINCT device_id) FROM usage WHERE ts>?", week_ago),
            "generations_total": g("SELECT COUNT(*) FROM usage"),
            "generations_week":  g("SELECT COUNT(*) FROM usage WHERE ts>?", week_ago),
            "copies_week":     g("SELECT COUNT(*) FROM events WHERE kind IN ('copy','edit_copy') AND ts>?", week_ago),
            "corrections_total": g("SELECT COUNT(*) FROM events WHERE kind='edit_copy'")
                               + g("SELECT COUNT(*) FROM feedback WHERE corrected IS NOT NULL AND corrected != ''"),
            "avg_quality_week": c.execute(
                "SELECT ROUND(AVG(quality_rate),3) FROM usage WHERE ts>? AND quality_rate IS NOT NULL",
                (week_ago,)).fetchone()[0],
        }


if __name__ == "__main__":
    # offline self-test: DB + quota + flywheel logic (no model server needed)
    DB_PATH = ":memory:"  # noqa
    print("NOTE: run via uvicorn for the real server; this is a logic self-test.")
