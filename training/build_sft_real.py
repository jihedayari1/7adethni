"""
Assemble the Stage-2 SFT set from REAL sources -> training/sft_real.jsonl

Teaches instruction-following in Tunisian, from curated truth (not hallucination):
  translate     EN/FR/Ar -> Arabizi           (real_pairs Arabizi-out + lexicon example_gloss->example)
  use_word      "3tini jomla b 'X'"           (lexicon example sentences)
  meaning       "chnia ma3na 'X'?"            (lexicon gloss; bilingual, minority slice)
  conversation  real comment->reply           (real_conv_pairs, YouTube)
  cs_tones      customer-service / tones       (best synthetic cs_pairs, quality-gated, capped)

Retention (15% general EN/FR) is added in the SFT notebook from an open instruction set,
not here (keeps this file Tunisian-only + auditable).

Usage:  python training/build_sft_real.py
Out:    training/sft_real.jsonl  ({instruction, output, task, source})
"""
import hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset" / "tools"))
sys.path.insert(0, str(ROOT / "rag"))
from coherence_score import load_vocab, score_text          # noqa: E402

OUT = ROOT / "training" / "sft_real.jsonl"
LEX = ROOT / "rag" / "lexicon.jsonl"
random.seed(42)

_AR = re.compile(r"[؀-ۿ]")
def is_arabizi(t): return bool(t) and not _AR.search(t) and bool(re.search(r"[a-z][2-9]|[2-9][a-z]", t.lower()))

# instruction templates (varied -> avoid overfitting to one phrasing)
T_TRANSLATE = ["9olha bel tounsi (arabizi): {x}", "7awwelha l derja: {x}",
               "kifech n9oulou hethi bel tounsi: {x}", "3tini el jomla hethi bel arabizi: {x}"]
T_USE = ["3tini jomla b el kelma '{w}'", "esta3mel '{w}' fi jomla", "a3mel mthal 3al kelma '{w}'"]
T_MEAN = ["chnia ma3na '{w}' ?", "chnowa te3ni '{w}' ?", "fasserli '{w}'"]

CAPS = {"translate": 12000, "use_word": 6000, "meaning": 2500, "conversation": 99999, "cs_tones": 1500}


def add(rows, seen, instr, out, task, source, gate_vocab=None):
    instr, out = (instr or "").strip(), (out or "").strip()
    if not instr or not out:
        return False
    key = (instr[:60], out[:60])
    if key in seen:
        return False
    if gate_vocab and is_arabizi(out):
        rate, _, n = score_text(out, *gate_vocab)
        if n >= 3 and rate < 0.70:            # drop noisy Arabizi targets
            return False
    seen.add(key)
    rows.append({"instruction": instr, "output": out, "task": task, "source": source})
    return True


def main():
    print("loading lexicon vocab for quality gate...")
    vocab, skels, _ = load_vocab()
    gate = (vocab, skels)
    rows, seen = [], set()
    counts = {}

    def capped(task):
        counts[task] = counts.get(task, 0)
        return counts[task] >= CAPS[task]

    # ---- lexicon-derived comprehension/translation ----
    # CONTAMINATION GUARD: skip entries reserved for the TunBench comprehension test
    # (same rule as dataset/tools/tunbench.py build: hash(variants[0]) % 12 == 0)
    def is_heldout(e):
        v = (e.get("arabizi_variants") or [None])[0]
        return bool(v) and int(hashlib.md5(v.encode()).hexdigest()[:8], 16) % 12 == 0

    lex = [json.loads(l) for l in open(LEX, encoding="utf-8")] if LEX.exists() else []
    # word-level exclusion: homographs share variant strings across entries, so drop ANY
    # entry that shares a variant with a held-out entry (else the word still gets taught)
    held_words = set()
    for e in lex:
        if is_heldout(e):
            held_words.update(v.lower() for v in (e.get("arabizi_variants") or []) if v)
    before = len(lex)
    lex = [e for e in lex
           if not any((v or "").lower() in held_words for v in (e.get("arabizi_variants") or []))]
    print(f"held out {before - len(lex)} lexicon entries ({len(held_words)} word strings) for TunBench")
    random.shuffle(lex)
    for e in lex:
        ex_ar = (e.get("example_arabizi") or "").strip()
        ex_gl = (e.get("example_gloss") or "").strip()
        gloss = (e.get("english") or e.get("french") or "").strip()
        variants = [v for v in (e.get("arabizi_variants") or []) if v]
        w = variants[0] if variants else None
        # translate: english SENTENCE -> arabizi sentence (real pair)
        if ex_ar and ex_gl and is_arabizi(ex_ar) and not capped("translate"):
            if add(rows, seen, random.choice(T_TRANSLATE).format(x=ex_gl), ex_ar, "translate", "lexicon", gate):
                counts["translate"] += 1
        # use-in-sentence
        if ex_ar and w and is_arabizi(ex_ar) and not capped("use_word"):
            if add(rows, seen, random.choice(T_USE).format(w=w), ex_ar, "use_word", "lexicon", gate):
                counts["use_word"] += 1
        # meaning (bilingual, minority)
        if w and gloss and not capped("meaning"):
            if add(rows, seen, random.choice(T_MEAN).format(w=w), f"'{w}' ya3ni: {gloss}", "meaning", "lexicon"):
                counts["meaning"] += 1

    # ---- real translation pairs (Arabizi output) ----
    p = ROOT / "aigenerateddataset" / "real_pairs.jsonl"
    if p.exists():
        rp = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        random.shuffle(rp)
        for r in rp:
            if capped("translate"): break
            if is_arabizi(r.get("output", "")):
                if add(rows, seen, r["instruction"], r["output"], "translate", "real_pairs", gate):
                    counts["translate"] += 1

    # ---- real conversation (YouTube) ----
    p = ROOT / "aigenerateddataset" / "real_conv_pairs.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            add(rows, seen, r["instruction"], r["output"], "conversation", "real_conv", gate)

    # ---- customer-service / tones (best synthetic, capped + gated) ----
    p = ROOT / "aigenerateddataset" / "cs_pairs.jsonl"
    if p.exists():
        cs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        random.shuffle(cs)
        for r in cs:
            if capped("cs_tones"): break
            if add(rows, seen, r["instruction"], r["output"], "cs_tones", "cs_synth", gate):
                counts["cs_tones"] += 1

    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_task = Counter(r["task"] for r in rows)
    print(f"\nSFT rows: {len(rows)}")
    for k, v in by_task.most_common():
        print(f"  {k:13}: {v}")
    print(f"-> {OUT.relative_to(ROOT)}")
    print("NOTE: the SFT notebook adds ~15% EN/FR retention (open instruction set) on top.")


if __name__ == "__main__":
    main()
