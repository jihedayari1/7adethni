"""
Process the 3GB raw corpus (chunk_*.json.zip) -> corpus_clean.jsonl for CPT.

Runs on a laptop CPU (I/O-bound). Per chunk it:
  1. unzips in-memory, reads [{"text": ...}, ...]
  2. normalizes (NFC, strip URLs/HTML/@mentions/phones, collapse whitespace)
  3. hard-filters (length, emoji/symbol ratio, char-repeat ratio)
  4. exact-dedups (hash) and near-dedups (MinHash-LSH if `datasketch` is installed)
  5. buckets by script + dialect:  tn_arabizi | tn_arabic | msa | fr | en | junk
     - REJECTS Moroccan/Algerian markers (both scripts)
  6. writes {"text","bucket","tn_score"} lines

Usage:
  python corpus/process_corpus.py                       # all chunks in repo root
  python corpus/process_corpus.py --limit 2000          # quick test: 2000 records/chunk
  python corpus/process_corpus.py --chunks "chunk_1.json.zip"
Out: corpus/corpus_clean.jsonl  + printed bucket stats.
"""
import argparse, glob, hashlib, io, json, re, sys, unicodedata, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus" / "corpus_clean.jsonl"

# ---------- cleaning ----------
_URL   = re.compile(r"https?://\S+|www\.\S+")
_HTML  = re.compile(r"<[^>]+>")
_MENT  = re.compile(r"@\w+")
_PHONE = re.compile(r"\b\d{8,}\b|\+216[\s-]?\d[\d\s-]{6,}")
_WS    = re.compile(r"\s+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]")

def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "")
    t = _URL.sub(" ", t); t = _HTML.sub(" ", t); t = _MENT.sub(" ", t); t = _PHONE.sub(" ", t)
    return _WS.sub(" ", t).strip()

def junk_ratios(t: str):
    if not t: return 1.0, 1.0
    letters = sum(ch.isalpha() for ch in t)
    emoji = len(_EMOJI.findall(t))
    sym = len(t) - letters - t.count(" ")
    rep = len(re.findall(r"(.)\1{3,}", t))          # 4+ char runs
    return (sym / max(len(t), 1)), (emoji + rep)

# ---------- dialect markers ----------
_AR = re.compile(r"[؀-ۿ]")
_NUMLETTER = re.compile(r"[a-z][2-9]|[2-9][a-z]", re.I)
# Tunisian (latin + arabic script)
_TUN_LAT = re.compile(r"\b(barcha|barsha|famma|fama|chnowa|chnoua|9addech|9adech|bch|bech|taw|tawa|"
                      r"mta3|mte3|brabi|3aslema|ya5i|5ouya|yezzi|fissa3|behi|mli7|3aychek|9ahwa|"
                      r"win|wa9tech|3lech|kifech|hetha|hethi|9a3ed|barra|3andi|3andek)\b", re.I)
_TUN_AR  = re.compile(r"(برشا|برشة|فمة|شنوة|شنية|قداش|باش|متاع|ياخي|خويا|يزي|فيسع|بهي|ملي|عيشك|"
                      r"قهوة|وقتاش|علاش|كيفاش|هاذا|هاذي|قاعد|برة|عندي|عندك|توة)")
_MSA_AR  = re.compile(r"(الذي|التي|سوف|ليس|كيف|عندما|لذلك|يمكن|يجب|نحن|جداً|كثيراً|هذا|هذه|ذلك|إنّ|سوف)")
# Moroccan / Algerian -> REJECT
_MOR = re.compile(r"\b(bghit|baghi|daba|wakha|dyal|diali|bzaf|finta|zwin|khasni|lhih|drari)\b|"
                  r"(بغيت|دابا|واخا|ديال|بزاف|زوين|خصني|فينتا)", re.I)
_FR = re.compile(r"\b(le|la|les|des|une|est|dans|pour|avec|mais|vous|nous|c'est|je|tu|il|sur|pas|"
                 r"bien|merci|bonjour|voila|trop)\b", re.I)
_EN = re.compile(r"\b(the|and|you|are|this|that|with|have|for|not|but|what|when|your|from|just)\b", re.I)

def bucket(t: str):
    """Return (bucket, tn_score in 0..1). tn_score = Tunisian-ness confidence."""
    if _MOR.search(t):
        return "junk", 0.0                                  # Moroccan/Algerian -> drop
    ar = bool(_AR.search(t)); lat = bool(re.search(r"[A-Za-z]", t))
    tun_ar, tun_lat = len(_TUN_AR.findall(t)), len(_TUN_LAT.findall(t))
    nl = len(_NUMLETTER.findall(t))
    if ar and not lat:                                       # arabic script
        if _MSA_AR.search(t) and tun_ar == 0:
            return "msa", 0.1
        return "tn_arabic", min(1.0, 0.4 + 0.15 * tun_ar)
    if ar and lat:                                           # mixed script -> treat as tn_arabic-ish
        return "tn_arabic", min(1.0, 0.3 + 0.1 * (tun_ar + tun_lat))
    # latin only
    if nl or tun_lat:                                        # arabizi signal
        return "tn_arabizi", min(1.0, 0.4 + 0.1 * (tun_lat + nl))
    fr, en = len(_FR.findall(t)), len(_EN.findall(t))
    if en > fr and en >= 2: return "en", 0.0
    if fr >= 2:            return "fr", 0.0
    return "junk", 0.0                                       # unrecognized short latin

# ---------- near-dedup (optional) ----------
def get_minhasher():
    try:
        from datasketch import MinHash, MinHashLSH
        return MinHash, MinHashLSH
    except Exception:
        return None, None

def shingles(t, k=5):
    t = re.sub(r"\s+", " ", t.lower())
    return {t[i:i+k] for i in range(max(1, len(t) - k + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=None, help="glob for chunk zips (default: repo chunk_*.json.zip)")
    ap.add_argument("--limit", type=int, default=0, help="max records per chunk (0=all)")
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--max-len", type=int, default=2000)
    ap.add_argument("--no-near-dedup", action="store_true")
    args = ap.parse_args()

    zips = sorted(glob.glob(args.chunks) if args.chunks else glob.glob(str(ROOT / "chunk_*.json.zip")))
    if not zips:
        print("No chunk_*.json.zip found. Put them in the repo root (see DATA.md)."); return

    MinHash, MinHashLSH = (None, None) if args.no_near_dedup else get_minhasher()
    lsh = MinHashLSH(threshold=0.85, num_perm=64) if MinHashLSH else None
    if not lsh and not args.no_near_dedup:
        print("(datasketch not installed -> exact-dedup only. `pip install datasketch` for near-dedup.)")

    seen_exact = set()
    stats = {b: 0 for b in ["tn_arabizi", "tn_arabic", "msa", "fr", "en", "junk"]}
    raw = kept = near_dropped = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_lsh = 0

    with open(OUT, "w", encoding="utf-8") as out:
        for zp in zips:
            with zipfile.ZipFile(zp) as z:
                data = json.loads(z.read(z.namelist()[0]).decode("utf-8", "ignore"))
            for i, rec in enumerate(data):
                if args.limit and i >= args.limit:
                    break
                raw += 1
                t = clean(rec.get("text") if isinstance(rec, dict) else rec)
                if not (args.min_len <= len(t) <= args.max_len):
                    continue
                symr, noise = junk_ratios(t)
                if symr > 0.4 or len(t.split()) < 3:
                    continue
                h = hashlib.md5(t.lower().encode()).hexdigest()
                if h in seen_exact:
                    continue
                seen_exact.add(h)
                b, score = bucket(t)
                if b == "junk":
                    stats["junk"] += 1
                    continue
                if lsh is not None:
                    m = MinHash(num_perm=64)
                    for sh in shingles(t):
                        m.update(sh.encode())
                    if lsh.query(m):
                        near_dropped += 1
                        continue
                    lsh.insert(f"d{n_lsh}", m); n_lsh += 1
                out.write(json.dumps({"text": t, "bucket": b, "tn_score": round(score, 2)},
                                     ensure_ascii=False) + "\n")
                stats[b] += 1; kept += 1
            print(f"  {Path(zp).name}: running kept={kept}")

    print(f"\nRAW {raw} -> KEPT {kept}  (exact-dup + junk removed; near-dropped {near_dropped})")
    for b, n in stats.items():
        print(f"  {b:11}: {n:7}")
    print(f"-> {OUT.relative_to(ROOT)}")
    print("\nCPT-usable (tn_arabizi + tn_arabic):", stats["tn_arabizi"] + stats["tn_arabic"])


if __name__ == "__main__":
    main()
