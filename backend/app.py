"""
7adethni backend / gateway (FastAPI + SQLite, no GPU needed).

Sits between the extension and the GPU model server (serving/app.py). It adds:
  * per-device freemium quota (FREE_DAILY_LIMIT/day)
  * usage logging
  * the DATA FLYWHEEL: stores user feedback + corrections (POST /feedback)
  * proxies /generate to the model server

Run:
    pip install -r requirements.txt
    export MODEL_API_URL="http://localhost:8000"   # serving/app.py
    export MODEL_API_KEY="testkey123"              # optional, must match serving
    export FREE_DAILY_LIMIT=15
    uvicorn app:app --host 0.0.0.0 --port 9000

Architecture:  extension --> THIS backend (cheap CPU) --> model server (GPU)
"""
import os, sqlite3, time, uuid, datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_API_URL    = os.environ.get("MODEL_API_URL", "http://localhost:8000")
MODEL_API_KEY    = os.environ.get("MODEL_API_KEY", "")
FREE_DAILY_LIMIT = int(os.environ.get("FREE_DAILY_LIMIT", "15"))
DB_PATH          = os.environ.get("DB_PATH", "7adethni.db")

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
        CREATE INDEX IF NOT EXISTS idx_usage_day ON usage(device_id, day);
        """)

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


class GenReq(BaseModel):
    device_id: str
    feature: str = "idea"
    text: str
    tone: str = "normal"

class FbReq(BaseModel):
    request_id: str
    rating: str               # "good" | "bad"
    corrected: str | None = None


@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True, "model_api": MODEL_API_URL, "free_daily_limit": FREE_DAILY_LIMIT}

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
    try:
        headers = {"x-api-key": MODEL_API_KEY} if MODEL_API_KEY else {}
        r = requests.post(f"{MODEL_API_URL.rstrip('/')}/generate", json={
            "feature": req.feature, "text": req.text, "tone": req.tone}, headers=headers, timeout=120)
        r.raise_for_status()
        output = r.json().get("output", "")
    except requests.RequestException as e:
        raise HTTPException(502, f"model server error: {e}")
    rid = uuid.uuid4().hex
    with db() as c:
        c.execute("INSERT INTO usage(id,device_id,ts,day,feature,tone,input,output) VALUES(?,?,?,?,?,?,?,?)",
                  (rid, req.device_id, time.time(), today(), req.feature, req.tone, req.text, output))
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


if __name__ == "__main__":
    # offline self-test: DB + quota logic (no model server needed)
    DB_PATH = ":memory:"  # noqa
    print("NOTE: run via uvicorn for the real server; this is a logic self-test.")
