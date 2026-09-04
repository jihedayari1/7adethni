"""
Assemble the Stage-1 CPT corpus (dual-script Tunisian) -> training/cpt.jsonl

The mix that teaches the model to SPEAK Arabizi + UNDERSTAND Tunisian (NileChat recipe):
  real_arabizi     12%  TUNIZI tweets + lexicon examples + scraped (scarce -> upsampled)
  translit_arabizi 36%  Arabic-script corpus -> deterministic Arabizi (rag/normalizer)
  raw_arabic       36%  Arabic-script Tunisian (comprehension)
  translit_pairs    8%  explicit "arabic <-> arabizi" lines (script mapping)
  fr_en retention   8%  anti-forgetting (thin in this corpus; SFT adds English retention)

Uses whatever is present; warns for missing sources (you can run it before the corpus is
processed to get a smaller TUNIZI+lexicon CPT set, then re-run after process_corpus.py).

Usage:
  python training/build_cpt_corpus.py --tokens 8000000   # ~1 Kaggle T4 session
Out: training/cpt.jsonl  ({"text": ...} lines, ready to pack in the CPT notebook)
"""
import argparse, glob, hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
from normalizer import transliterate_ar_to_arabizi        # noqa: E402

OUT = ROOT / "training" / "cpt.jsonl"
CORPUS = ROOT / "corpus" / "corpus_clean.jsonl"
LEX = ROOT / "rag" / "lexicon.jsonl"
random.seed(42)

# Mix rebalanced to what we ACTUALLY have (real Arabizi is only ~240k tokens, so it is
# upsampled; the corpus supplies the bulk via transliteration + raw Arabic).
MIX = {"real_arabizi": .12, "translit_arabizi": .36, "raw_arabic": .36,
       "translit_pairs": .08, "fr_en": .08}
# how many times a scarce-but-valuable slice may repeat to reach its share
REPEAT = {"real_arabizi": 10, "translit_pairs": 10, "fr_en": 6,
          "translit_arabizi": 1, "raw_arabic": 1}
_AR = re.compile(r"[؀-ۿ]")
def toks(s): return max(1, int(len(s.split()) * 1.4))     # rough token estimate


def load_corpus_buckets():
    buckets = {"tn_arabic": [], "fr": [], "en": []}
    if not CORPUS.exists():
        print("  ! corpus/corpus_clean.jsonl missing -> run corpus/process_corpus.py for the big volume.")
        return buckets
    for l in open(CORPUS, encoding="utf-8"):
        try: r = json.loads(l)
        except Exception: continue
        b = r.get("bucket")
        if b in buckets: buckets[b].append(r["text"])
        elif b == "tn_arabizi": buckets.setdefault("tn_arabizi", []).append(r["text"])
    return buckets


def load_real_arabizi():
    out = []
    # TUNIZI parquet
    try:
        import pandas as pd
        pq = glob.glob(str(ROOT / "**" / "train-*.parquet"), recursive=True) or glob.glob(str(ROOT / "*.parquet"))
        if pq:
            df = pd.read_parquet(pq[0])
            col = "Tweet" if "Tweet" in df.columns else df.columns[0]
            # keep LATIN-script tweets only (this bucket teaches Arabizi generation)
            tz = [str(t).strip() for t in df[col].tolist()
                  if isinstance(t, str) and len(str(t)) > 8 and not _AR.search(str(t))]
            out += tz
            print(f"  TUNIZI: +{len(tz)} real Arabizi tweets (latin-only, from {len(df)})")
    except Exception as e:
        print("  ! TUNIZI parquet skipped:", e)
    # lexicon example sentences (real Arabizi)
    lex_examples, translit_pairs = [], []
    if LEX.exists():
        for l in open(LEX, encoding="utf-8"):
            e = json.loads(l)
            ex = (e.get("example_arabizi") or "").strip()
            if len(ex) > 8: lex_examples.append(ex)
            ar, va = (e.get("arabic_script") or "").strip(), (e.get("arabizi_variants") or [])
            if ar and va:
                translit_pairs.append(f"{ar}\n{va[0]}")          # explicit script mapping
    out += lex_examples
    # scraped real conversation/translation Arabizi outputs
    for fn, key in [("aigenerateddataset/real_conv_pairs.jsonl", "output"),
                    ("aigenerateddataset/real_pairs.jsonl", "output")]:
        p = ROOT / fn
        if p.exists():
            for l in open(p, encoding="utf-8"):
                try: r = json.loads(l)
                except Exception: continue
                v = (r.get(key) or "").strip()
                if v and not _AR.search(v) and re.search(r"[a-z][2-9]|[2-9][a-z]", v.lower()):
                    out.append(v)
    return out, translit_pairs


def fill(pool, budget_tokens, transform=None, max_repeat=1):
    """Unique pass first (dedup, shuffle); then repeat up to max_repeat to reach budget."""
    random.shuffle(pool)
    picked, seen, used = [], set(), 0
    for t in pool:
        t = transform(t) if transform else t
        t = t.strip()
        if not t: continue
        h = hashlib.md5(t.lower().encode()).hexdigest()
        if h in seen: continue
        seen.add(h); picked.append(t); used += toks(t)
        if used >= budget_tokens: break
    # scarce slice: repeat it (dedup would otherwise drop the copies)
    if max_repeat > 1 and picked and used < budget_tokens:
        base, reps = list(picked), 1
        while used < budget_tokens and reps < max_repeat:
            for t in base:
                picked.append(t); used += toks(t)
                if used >= budget_tokens: break
            reps += 1
    return picked, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8_000_000, help="target total CPT tokens")
    args = ap.parse_args()

    print("loading sources...")
    cb = load_corpus_buckets()
    real_ar, translit_pairs = load_real_arabizi()
    tn_arabic = cb.get("tn_arabic", []) + cb.get("tn_arabizi", [])
    fr_en = cb.get("fr", []) + cb.get("en", [])

    plan = {
        "real_arabizi":     (real_ar,       None),
        "translit_arabizi": (tn_arabic,     transliterate_ar_to_arabizi),
        "raw_arabic":       (tn_arabic,     None),
        "translit_pairs":   (translit_pairs, None),
        "fr_en":            (fr_en,         None),
    }
    lines, total, got = [], 0, {}
    print(f"\ntarget {args.tokens:,} tokens")
    for name, frac in MIX.items():
        pool, tf = plan[name]
        if not pool:
            print(f"  {name:16} SKIP (no data)"); continue
        picked, used = fill(list(pool), int(args.tokens * frac), tf, REPEAT.get(name, 1))
        for t in picked: lines.append({"text": t, "src": name})
        total += used
        got[name] = used
        print(f"  {name:16} {len(picked):7} lines  ~{used:,} tok  (target {frac:.0%})")

    random.shuffle(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTOTAL {len(lines):,} lines  ~{total:,} tokens  -> {OUT.relative_to(ROOT)}")
    print("achieved mix:", ", ".join(f"{k} {v/max(total,1):.0%}" for k, v in got.items()))
    real_share = got.get("real_arabizi", 0) / max(total, 1)
    if real_share < 0.08:
        print("WARNING: real Arabizi is only {:.0%} of the corpus.".format(real_share))
        print("         More tokens here just adds TRANSLITERATED (vowel-poor) style.")
        print("         Use ~20M tokens, or scrape more natively-Arabizi text (yt_comments.py).")
    if total < args.tokens * 0.6:
        print("NOTE: below target — process the full corpus (corpus/process_corpus.py) for more volume.")


if __name__ == "__main__":
    main()
