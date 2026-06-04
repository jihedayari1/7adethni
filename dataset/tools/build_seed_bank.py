"""
Phase 1, Step 1 — build the style seed bank.

Mines clean, real Tunisian Arabizi short texts from data already in the folder:
  - clean_darija_english.csv  (darija = native Arabizi, + english gloss)
  - train-00000-of-00001.parquet  (TUNIZI tweets + sentiment label)

Output: dataset/seed_bank.jsonl  ->  { text, source, english?, label?, score }
These become (a) few-shot exemplars for the Claude generator and (b) a spelling reference.

Quality heuristic ("arabiziness"): rewards Tunisian number-letters (7,3,9,5) and common
Derja function words; penalizes near-empty / emoji-only / very long rows.
"""
import csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rag"))
from normalizer import normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "seed_bank.jsonl"
csv.field_size_limit(10**7)

PER_SOURCE = 2500                       # cap per source
LEN_MIN, LEN_MAX = 15, 220

_LATIN = re.compile(r"[a-z]")
_ARAB = re.compile(r"[؀-ۿ]")
_DERJA_WORDS = {"barcha", "famma", "chnowa", "chnoua", "7keya", "3la", "mta3", "fi",
                "ya5i", "5ouya", "sa7a", "ken", "kima", "bahi", "ya3ni", "3andi",
                "n7eb", "ma3andich", "tw", "tawa", "9ahwa", "ghodwa", "lyoum",
                "behi", "mch", "mouch", "wa9tech", "win", "kifech", "3lech"}
_NUMLETTER = re.compile(r"[a-z][3579][a-z]|[3579][a-z]")  # 7 3 9 5 used as letters


def score(text: str) -> float:
    t = text.lower()
    n_latin = len(_LATIN.findall(t))
    if n_latin < 6:
        return 0.0
    s = 1.0
    s += 2.0 * len(_NUMLETTER.findall(t))                 # arabizi numerals = strong signal
    toks = set(re.findall(r"[a-z0-9']+", t))
    s += 1.5 * len(toks & _DERJA_WORDS)                   # native function words
    if _ARAB.search(text):                               # mixed-script: keep but down-weight
        s -= 1.0
    if len(text) > 160:
        s -= 1.0
    emoji_ratio = sum(ord(c) > 0x2500 for c in text) / max(len(text), 1)
    s -= 4.0 * emoji_ratio
    return s


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def harvest_csv(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt = clean(r.get("darija"))
            if not (LEN_MIN <= len(txt) <= LEN_MAX):
                continue
            sc = score(txt)
            if sc <= 1.5:
                continue
            rows.append({"text": txt, "source": "clean_darija_english",
                         "english": clean(r.get("english")) or None,
                         "label": None, "score": round(sc, 2)})
    return rows


def harvest_parquet(path: Path):
    import pandas as pd
    df = pd.read_parquet(path)
    rows = []
    for txt, lab in zip(df["Tweet"], df["label"]):
        txt = clean(str(txt))
        if not (LEN_MIN <= len(txt) <= LEN_MAX):
            continue
        sc = score(txt)
        if sc <= 1.5:
            continue
        rows.append({"text": txt, "source": "tunizi", "english": None,
                     "label": int(lab), "score": round(sc, 2)})
    return rows


def main():
    pools = []
    pools += [("clean_darija_english", harvest_csv(ROOT / "clean_darija_english.csv"))]
    pools += [("tunizi", harvest_parquet(ROOT / "train-00000-of-00001.parquet"))]

    seen, out = set(), []
    for name, rows in pools:
        rows.sort(key=lambda r: r["score"], reverse=True)
        kept = 0
        for r in rows:
            key = normalize(r["text"])[:60]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(r)
            kept += 1
            if kept >= PER_SOURCE:
                break
        print(f"  {name}: kept {kept} / {len(rows)} candidates")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWROTE {len(out)} exemplars -> {OUT.relative_to(ROOT)}")
    print("top examples:")
    for r in sorted(out, key=lambda r: r["score"], reverse=True)[:6]:
        print(f"   [{r['score']:4}] {r['text'][:80]}")


if __name__ == "__main__":
    main()
