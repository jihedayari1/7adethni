"""
TunBench — honest evaluation for a Tunisian Arabizi model.

Mechanical metrics (no vanity 'dialect rate'):
  real_word     % output words that are REAL Tunisian words (vs 17k lexicon)   -> catches gibberish
  msa_leak      % answers that leaked into MSA/Arabic-script                    -> catches "not Tunisian"
  number_rule   % answers using ONLY valid Arabizi digits (2,3,5,7,8,9)         -> catches broken Arabizi
  comprehension accuracy on a held-out slang test (EN -> expected Arabizi word) -> catches "doesn't know slang"

Two commands:
  # 1) build a held-out comprehension test from the lexicon (contamination-safe: hash split)
  python dataset/tools/tunbench.py build --n 300        -> dataset/tunbench_comprehension.jsonl
  # 2) score a predictions file  [{instruction, output, expected?}]
  python dataset/tools/tunbench.py score preds.jsonl
"""
import json, re, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "tools"))
sys.path.insert(0, str(ROOT / "rag"))
from coherence_score import load_vocab, score_text, norm, skel   # noqa: E402
try:
    from msa_leakage import score_text as _msa_raw                # noqa: E402
    def msa_label(t):
        r = _msa_raw(t)
        return r.get("label") if isinstance(r, dict) else r        # score_text returns a dict
except Exception:
    msa_label = None

LEX = ROOT / "rag" / "lexicon.jsonl"
COMP_OUT = ROOT / "dataset" / "tunbench_comprehension.jsonl"
_AR = re.compile(r"[؀-ۿ]")
VALID_DIGITS = set("235789")          # phonemic Arabizi digits (2=ء 3=ع 5=خ 7=ح 8=غ 9=ق)


# ---- number-rule conformance ----
def number_rule_ok(text: str):
    """False if any letter+digit token uses an invalid digit (0/1/4/6) as a letter."""
    bad = []
    for tok in re.findall(r"[A-Za-z0-9']+", text or ""):
        if tok.isdigit() or tok.isalpha():
            continue                                   # pure number (price) or pure word = fine
        for ch in tok:
            if ch.isdigit() and ch not in VALID_DIGITS:
                bad.append(tok); break
    return (len(bad) == 0), bad


# ---- comprehension: does the output contain the expected Arabizi word? ----
def contains_expected(output: str, expected_variants):
    out_toks = {norm(w) for w in re.findall(r"[A-Za-z0-9']+", output or "")}
    out_skel = {skel(t) for t in out_toks if len(skel(t)) >= 2}
    for v in expected_variants:
        nv = norm(v)
        if nv in out_toks or (len(skel(nv)) >= 2 and skel(nv) in out_skel):
            return True
    return False


def build_comprehension(n):
    """Held-out slang test: EN gloss -> expected Arabizi variant(s). Hash-split for safety."""
    items = []
    for l in open(LEX, encoding="utf-8"):
        e = json.loads(l)
        gloss = (e.get("english") or "").strip()
        variants = [v for v in (e.get("arabizi_variants") or []) if v]
        ex = (e.get("example_arabizi") or "").strip()
        if not gloss or not variants:
            continue
        # only entries whose hash lands in the held-out 8% (never used elsewhere)
        if int(hashlib.md5((variants[0]).encode()).hexdigest()[:8], 16) % 12 != 0:
            continue
        items.append({"instruction": f"9olha bel tounsi (arabizi): {gloss}",
                      "expected": variants, "example": ex, "task": "comprehension"})
        if len(items) >= n:
            break
    COMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(COMP_OUT, "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"built {len(items)} held-out comprehension items -> {COMP_OUT.relative_to(ROOT)}")


def score(preds_path):
    vocab, skels, _ = load_vocab()
    rows = [json.loads(l) for l in open(preds_path, encoding="utf-8") if l.strip()]
    n = len(rows) or 1
    rw = msa = numok = comp = comp_total = 0
    numviol = []
    for r in rows:
        out = r.get("output", "")
        rw += score_text(out, vocab, skels)[0]
        if msa_label:
            msa += 1 if msa_label(out) in ("msa_leak", "arabic_script") else 0
        elif _AR.search(out):
            msa += 1
        ok, bad = number_rule_ok(out)
        numok += 1 if ok else 0
        numviol += bad
        if r.get("expected"):
            comp_total += 1
            comp += 1 if contains_expected(out, r["expected"]) else 0

    print(f"n = {len(rows)}")
    print(f"  real_word   : {rw/n:.0%}   (higher = fewer invented words)")
    print(f"  msa_leak    : {msa/n:.0%}   (lower = stays Tunisian)")
    print(f"  number_rule : {numok/n:.0%}  (higher = valid Arabizi digits)")
    if comp_total:
        print(f"  comprehension: {comp/comp_total:.0%}  (n={comp_total}, held-out slang)")
    if numviol[:8]:
        print(f"  sample digit violations: {numviol[:8]}")
    return {"real_word": rw/n, "msa_leak": msa/n, "number_rule": numok/n,
            "comprehension": (comp/comp_total if comp_total else None)}


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        nn = int(sys.argv[sys.argv.index("--n")+1]) if "--n" in sys.argv else 300
        build_comprehension(nn)
    elif len(sys.argv) >= 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        print(__doc__)
