"""
Flywheel export — turn real user interactions into training data.

Reads the backend SQLite DB (usage / events / feedback) and emits three gated files:

  aigenerateddataset/flywheel_sft.jsonl    accepted outputs  -> future SFT
  aigenerateddataset/flywheel_dpo.jsonl    (chosen, rejected) -> future DPO
  dataset/eval_candidates.jsonl            eval-pool prompts  -> TunBench-Real (never trained on)

Gates (per the flywheel design):
  * eval_pool=1 rows go ONLY to eval_candidates — contamination guard by construction.
  * SFT accept = copied-unedited OR 👍, quality_rate >= 0.85, text present (opt-out rows excluded).
  * Corrections (edit_copy payload / ✏️ feedback) become SFT targets IF the correction's
    real-word rate isn't >10pts worse than the original and isn't a punctuation-only change.
  * DPO chosen/rejected: (correction, original) pairs + (copied, regenerated-away) chains.
  * PII scrub (emails/phones/@mentions/URLs) on everything exported.
  * Dedup by normalized (input, output).

Usage:
  python dataset/tools/export_flywheel.py [path/to/7adethni.db]
  python dataset/tools/export_flywheel.py --selftest        # builds a fake DB and verifies gates
"""
import json, os, re, sqlite3, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "serving"))
from normalizer import normalize                     # noqa: E402
from grounding import quality                        # noqa: E402  (lexicon real-word scorer)

SFT_OUT  = ROOT / "aigenerateddataset" / "flywheel_sft.jsonl"
DPO_OUT  = ROOT / "aigenerateddataset" / "flywheel_dpo.jsonl"
EVAL_OUT = ROOT / "dataset" / "eval_candidates.jsonl"

MIN_QUALITY   = 0.85     # SFT gate on real-word rate
MAX_Q_DROP    = 0.10     # correction may not be >10pts worse than original
REGEN_WINDOW  = 600      # regen chain: same device+input within 10 min

# ---- PII scrub (same policy as build_real_conv) --------------------------------
_EMAIL   = re.compile(r"\S+@\S+\.\S+")
_URL     = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_PHONE   = re.compile(r"\b\d{8,}\b|\+216[\s-]?\d[\d\s-]{6,}")

def pii_strip(t: str) -> str:
    t = _EMAIL.sub(" ", t or ""); t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t); t = _PHONE.sub(" [num] ", t)
    return re.sub(r"\s+", " ", t).strip()

_PUNCT_ONLY = re.compile(r"[\W_]+", re.UNICODE)
def substantive_change(a: str, b: str) -> bool:
    """True if a->b changes more than punctuation/emoji/whitespace."""
    return _PUNCT_ONLY.sub("", a or "").lower() != _PUNCT_ONLY.sub("", b or "").lower()


def rows(c, sql, *args):
    return [dict(r) for r in c.execute(sql, args).fetchall()]


def export(db_path: str):
    c = sqlite3.connect(db_path); c.row_factory = sqlite3.Row
    usage = {r["id"]: r for r in rows(c, "SELECT * FROM usage")}
    events = rows(c, "SELECT * FROM events ORDER BY ts")
    fb     = rows(c, "SELECT * FROM feedback")

    ev_by_usage = {}
    for e in events:
        ev_by_usage.setdefault(e["usage_id"], []).append(e)

    sft, dpo, evalc = [], [], []
    seen_sft = set()

    def add_sft(u, target, source):
        inp, out = pii_strip(u["input"]), pii_strip(target)
        if not inp or not out:
            return
        key = (normalize(inp)[:60], normalize(out)[:60])
        if key in seen_sft:
            return
        seen_sft.add(key)
        sft.append({"instruction": inp, "output": out, "feature": u["feature"],
                    "tone": u["tone"], "lang_in": u["lang_in"], "source": source,
                    "synthetic": False, "needs_native_review": source != "copy"})

    def corrected_ok(u, corr) -> bool:
        if not corr or not u["output"] or not substantive_change(u["output"], corr):
            return False
        # rescore BOTH with the same scorer (never mix stored vs local scores: a lexicon
        # gap like 'm3attel' would penalize only the correction and reject legit fixes)
        q_new = quality(corr)["rate"]
        q_old = quality(u["output"])["rate"]
        return q_new >= q_old - MAX_Q_DROP          # reject vandalized "corrections"

    for uid, u in usage.items():
        u = dict(u)
        if not u.get("input") or not u.get("output"):
            continue                                 # opt-out rows: nothing to export
        # ---- eval pool: reserved, exported ONLY as benchmark candidates ----
        if u.get("eval_pool"):
            evalc.append({"prompt": pii_strip(u["input"]), "feature": u["feature"],
                          "tone": u["tone"], "lang_in": u["lang_in"], "usage_id": uid})
            continue
        evs = {e["kind"]: e for e in ev_by_usage.get(uid, [])}
        # ---- corrections (gold) ----
        if "edit_copy" in evs and corrected_ok(u, evs["edit_copy"]["payload"]):
            corr = evs["edit_copy"]["payload"]
            add_sft(u, corr, "edit_copy")
            dpo.append({"prompt": pii_strip(u["input"]), "feature": u["feature"], "tone": u["tone"],
                        "chosen": pii_strip(corr), "rejected": pii_strip(u["output"]),
                        "source": "edit_copy"})
        # ---- accepted as-is ----
        elif "copy" in evs and (u["quality_rate"] or 0) >= MIN_QUALITY:
            add_sft(u, u["output"], "copy")

    # ---- explicit feedback ----
    good_ids = {f["usage_id"] for f in fb if f["rating"] == "good"}
    for uid in good_ids:
        u = usage.get(uid)
        if u and u["input"] and u["output"] and not u["eval_pool"] \
           and (u["quality_rate"] or 0) >= MIN_QUALITY:
            add_sft(dict(u), u["output"], "fb_good")
    for f in fb:
        u = usage.get(f["usage_id"])
        if not u or u["eval_pool"] or not u["input"]:
            continue
        if f["corrected"] and corrected_ok(dict(u), f["corrected"]):
            add_sft(dict(u), f["corrected"], "fb_fix")
            dpo.append({"prompt": pii_strip(u["input"]), "feature": u["feature"], "tone": u["tone"],
                        "chosen": pii_strip(f["corrected"]), "rejected": pii_strip(u["output"]),
                        "source": "fb_fix"})

    # ---- regen chains: regenerated-away output = rejected, later copied = chosen ----
    regen_uids = [e["usage_id"] for e in events if e["kind"] == "regen"]
    by_dev_input = {}
    for uid, u in usage.items():
        if u["input"]:
            by_dev_input.setdefault((u["device_id"], normalize(u["input"])[:60]), []).append(dict(u))
    for lst in by_dev_input.values():
        lst.sort(key=lambda r: r["ts"])
    for uid in regen_uids:
        u = usage.get(uid)
        if not u or u["eval_pool"] or not u["input"] or not u["output"]:
            continue
        chain = by_dev_input.get((u["device_id"], normalize(u["input"])[:60]), [])
        for nxt in chain:
            if nxt["ts"] <= u["ts"] or nxt["ts"] - u["ts"] > REGEN_WINDOW:
                continue
            nxt_evs = {e["kind"] for e in ev_by_usage.get(nxt["id"], [])}
            if {"copy", "edit_copy"} & nxt_evs and nxt["output"] \
               and substantive_change(u["output"], nxt["output"]):
                dpo.append({"prompt": pii_strip(u["input"]), "feature": u["feature"], "tone": u["tone"],
                            "chosen": pii_strip(nxt["output"]), "rejected": pii_strip(u["output"]),
                            "source": "regen_chain"})
                break

    # ---- write ----
    for path, data in [(SFT_OUT, sft), (DPO_OUT, dpo), (EVAL_OUT, evalc)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SFT  {len(sft):4}  -> {SFT_OUT.relative_to(ROOT)}")
    print(f"DPO  {len(dpo):4}  -> {DPO_OUT.relative_to(ROOT)}")
    print(f"EVAL {len(evalc):4}  -> {EVAL_OUT.relative_to(ROOT)}  (benchmark-only, never train)")
    return sft, dpo, evalc


# ---------------------------------------------------------------- self-test
def selftest():
    import tempfile
    db = os.path.join(tempfile.gettempdir(), "fly_selftest.db")
    if os.path.exists(db): os.remove(db)
    c = sqlite3.connect(db)
    c.executescript("""
    CREATE TABLE usage(id TEXT PRIMARY KEY, device_id TEXT, ts REAL, day TEXT,
      feature TEXT, tone TEXT, input TEXT, output TEXT, model_version TEXT,
      latency_ms INTEGER, quality_rate REAL, oov TEXT, lang_in TEXT, eval_pool INTEGER DEFAULT 0);
    CREATE TABLE events(id TEXT PRIMARY KEY, usage_id TEXT, device_id TEXT, ts REAL, kind TEXT, payload TEXT);
    CREATE TABLE feedback(id TEXT PRIMARY KEY, usage_id TEXT, ts REAL, rating TEXT, corrected TEXT);
    """)
    t = time.time()
    U = [  # id, input, output, quality, eval_pool
        ("u1", "comment ça va", "3aslema chnowa a7welek", 0.95, 0),          # copy unedited -> SFT
        ("u2", "je suis en retard", "rani m3attel barcha sa7bi", 0.92, 0),   # edited -> SFT+DPO
        ("u3", "hello my friend", "3aslema ya sa7bi", 0.95, 1),              # eval pool -> eval only
        ("u4", "thank you", "chokran jazilan", 0.40, 0),                     # low quality copy -> dropped
        ("u5", "good morning", "sba7 5ir version 1", 0.90, 0),               # regen'd away -> DPO rejected
        ("u6", "good morning", "sba7 el 5ir ya 3slema", 0.93, 0),            # copied after regen -> DPO chosen
        ("u7", "call me 21612345678", "3ayyetli 3al 21612345678", 0.90, 0),  # PII -> scrubbed
    ]
    for i, (uid, inp, out, q, ev) in enumerate(U):
        c.execute("INSERT INTO usage VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (uid, "dev1", t + i, "2026-07-02", "translate", "normal", inp, out,
                   "test", 100, q, "[]", "latin", ev))
    E = [("u1", "copy", None), ("u2", "edit_copy", "rani m3attel barcha 5ouya"),
         ("u4", "copy", None), ("u5", "regen", None), ("u6", "copy", None), ("u7", "copy", None)]
    for i, (uid, kind, payload) in enumerate(E):
        c.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (f"e{i}", uid, "dev1", t + i, kind, payload))
    c.execute("INSERT INTO feedback VALUES(?,?,?,?,?)", ("f1", "u1", t, "good", None))
    c.commit(); c.close()

    sft, dpo, evalc = export(db)
    assert any(r["source"] == "copy" and r["instruction"] == "comment ça va" for r in sft), "copy->SFT failed"
    assert any(r["source"] == "edit_copy" and r["output"].endswith("5ouya") for r in sft), "correction->SFT failed"
    assert not any(r["instruction"] == "thank you" for r in sft), "low-quality gate failed"
    assert not any(r["instruction"] == "hello my friend" for r in sft), "eval-pool leaked into SFT!"
    assert any(e["prompt"] == "hello my friend" for e in evalc), "eval candidate missing"
    assert any(d["source"] == "edit_copy" for d in dpo), "correction DPO pair missing"
    assert any(d["source"] == "regen_chain" and d["rejected"].endswith("version 1") for d in dpo), "regen DPO missing"
    assert all("21612345678" not in json.dumps(r) for r in sft), "PII leaked!"
    print("\nSELF-TEST PASSED — all gates hold (incl. eval-pool isolation + PII scrub)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "backend" / "7adethni.db")
        if not os.path.exists(path):
            print(f"DB not found: {path}\nPass the path to the backend SQLite DB.")
        else:
            export(path)
