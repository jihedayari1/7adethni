#!/usr/bin/env python3
"""
Import pairs you generated on the Claude WEBSITE (free, your subscription) — no API key.

Free workflow:
  1) python aigenerateddataset/generate_cs_dataset.py --web-prompt > web_prompt.txt
  2) Paste web_prompt.txt into claude.ai. Claude returns a JSON array of pairs.
  3) Save that output to a file in  aigenerateddataset/inbox/  (e.g. batch1.json / .txt)
  4) python aigenerateddataset/import_pairs.py            # imports every file in inbox/
  5) python dataset/tools/review.py --in aigenerateddataset/cs_pairs.jsonl --reviewer <name>

It runs the SAME filters as the API generator (Arabizi-only, no MSA, convention pass,
dedup) and appends clean rows to cs_pairs.jsonl. Robust to markdown fences / extra prose.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aigenerateddataset"))
sys.path.insert(0, str(ROOT / "rag"))
from generate_cs_dataset import enforce_convention, is_clean  # same quality bar
from normalizer import normalize

INBOX = ROOT / "aigenerateddataset" / "inbox"
OUT = ROOT / "aigenerateddataset" / "cs_pairs.jsonl"

USER_KEYS = ("user", "customer", "instruction", "prompt")
ASST_KEYS = ("assistant", "agent", "output", "response")


def get(d, keys):
    for k in keys:
        if d.get(k):
            return d[k]
    return ""


def extract_objects(text: str):
    """Pull pair objects out of pasted text — whole JSON array, fenced block, or JSONL."""
    text = text.strip()
    # 1) strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 2) try a whole JSON array / object
    for candidate in (text, re.search(r"\[.*\]", text, re.S).group(0) if re.search(r"\[.*\]", text, re.S) else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            if isinstance(data, dict) and "pairs" in data:
                return data["pairs"]
        except json.JSONDecodeError:
            pass
    # 3) fall back to JSONL (one object per line)
    objs = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return objs


def load_seen():
    seen = set()
    if OUT.exists():
        for l in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(l)
                seen.add((normalize(get(r, USER_KEYS))[:50], normalize(get(r, ASST_KEYS))[:50]))
            except json.JSONDecodeError:
                pass
    return seen


def main():
    files = [Path(a) for a in sys.argv[1:]] or sorted(
        list(INBOX.glob("*.json")) + list(INBOX.glob("*.txt")) + list(INBOX.glob("*.jsonl")))
    if not files:
        INBOX.mkdir(parents=True, exist_ok=True)
        print(f"No input. Save Claude's pasted output into {INBOX.relative_to(ROOT)}/ "
              f"(as .json/.txt) then rerun.\nGet the prompt with:\n"
              f"  python aigenerateddataset/generate_cs_dataset.py --web-prompt > web_prompt.txt")
        return

    seen = load_seen()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = open(OUT, "a", encoding="utf-8")
    total_in = kept = dropped_dirty = dropped_dup = 0

    for fp in files:
        if not fp.exists():
            print(f"  skip (missing): {fp}"); continue
        objs = extract_objects(fp.read_text(encoding="utf-8"))
        fkept = 0
        for p in objs:
            total_in += 1
            u = enforce_convention(get(p, USER_KEYS))
            a = enforce_convention(get(p, ASST_KEYS))
            if not is_clean(u, a):
                dropped_dirty += 1; continue
            key = (normalize(u)[:50], normalize(a)[:50])
            if key in seen:
                dropped_dup += 1; continue
            seen.add(key)
            out.write(json.dumps({
                "instruction": u, "output": a,
                "category": p.get("category", ""), "topic": p.get("topic", ""),
                "domain": p.get("domain"), "synthetic": True, "needs_native_review": True,
                "source": "claude-web",
            }, ensure_ascii=False) + "\n")
            kept += 1; fkept += 1
        print(f"  {fp.name}: {len(objs)} parsed -> {fkept} kept")
    out.close()
    print(f"\nIMPORTED {kept} clean pairs (from {total_in} pasted) -> {OUT.relative_to(ROOT)}")
    print(f"dropped: {dropped_dirty} dirty (MSA/Arabic/too-short), {dropped_dup} duplicates")
    print(f"Next: python dataset/tools/review.py --in aigenerateddataset/cs_pairs.jsonl --reviewer <name>")


if __name__ == "__main__":
    main()
