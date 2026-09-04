"""
7adethni gateway on Modal (CPU, ~free) — reuses backend/app.py unchanged.

Zero-cash stack: this + serving/modal_app.py both run on Modal's free monthly credits.
The SQLite flywheel DB lives on a Modal Volume so it survives restarts/redeploys.

Deploy (from the repo root, after `modal setup`):
    modal secret create 7adethni-gateway MODEL_API_URL=<your modal serving URL> MODEL_API_KEY=<key>
    modal deploy backend/modal_backend.py
    # -> prints the public gateway URL; put THAT in the extension ⚙️ / website BACKEND_URL

Notes:
  * CPU container, scaledown after idle — a request costs fractions of a cent.
  * Volume writes are committed after every mutating request (DB is tiny; cheap).
  * Download a DB backup any time:  modal volume get 7adethni-db 7adethni.db ./backup.db
"""
import os
import modal

MINUTES = 60
app = modal.App("7adethni-gateway")

db_volume = modal.Volume.from_name("7adethni-db", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]", "requests", "pydantic>=2")
    .env({"DB_PATH": "/data/7adethni.db"})
    .add_local_file("backend/app.py", "/root/gateway/app.py")
)


@app.function(
    image=image,
    volumes={"/data": db_volume},
    secrets=[modal.Secret.from_name("7adethni-gateway")],   # MODEL_API_URL + MODEL_API_KEY
    scaledown_window=5 * MINUTES,
    timeout=5 * MINUTES,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def gateway():
    import sys
    sys.path.insert(0, "/root/gateway")
    os.environ["DB_PATH"] = "/data/7adethni.db"
    import app as backend                     # the existing FastAPI gateway, unchanged
    backend.init_db()

    # persist the Volume after every mutating request (SQLite file is tiny)
    @backend.app.middleware("http")
    async def commit_volume(request, call_next):
        response = await call_next(request)
        if request.method == "POST":
            try:
                db_volume.commit()
            except Exception:
                pass                          # never fail a user request on a commit hiccup
        return response

    return backend.app
