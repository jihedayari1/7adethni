"""
Build REAL Tunisian Arabizi training pairs from the lexicon's human-written examples.

Why: the synthetic (Claude-generated) pairs let the model invent words. The lexicon
holds ~17k REAL example sentences (from the derja-english corpus) + English glosses +
Arabic script. Because the Arabizi side is copied from real humans, the model CANNOT
hallucinate spelling or invent words — it learns correct words in real usage.

Emits three task types (varied instruction phrasings so the model doesn't overfit one):
  1. translate    EN gloss            -> real Arabizi   (the 'translate' feature)
  2. comprehend   real Arabizi        -> EN meaning     (understanding)
  3. vocab        "what does X mean"   -> gloss          (lexical grounding)

Output: aigenerateddataset/real_pairs.jsonl   (synthetic=False, needs_native_review=False)
Run:    python aigenerateddataset/build_real_pairs.py
"""
import json, os, re, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEX  = os.path.join(ROOT, "rag", "lexicon.jsonl")
OUT  = os.path.join(ROOT, "aigenerateddataset", "real_pairs.jsonl")
random.seed(7)

TRANSLATE_INSTR = [
    "9olha bel tounsi (arabizi): {x}",
    "traduci l derja tounsiya: {x}",
    "kifech n9oulou hethi bel tounsi: {x}",
    "7awwelha l derja: {x}",
]
COMPREHEND_INSTR = [
    "chnowa ma3neha hethi: {x}",
    "fasser hel jomla bel ingliz: {x}",
    "chnia el ma3na mta3: {x}",
]
VOCAB_INSTR = [
    "chnowa ma3neha el kelma '{x}'?",
    "3ach ya3ni '{x}' bel tounsi?",
]

def clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())

def ok_arabizi(s):
    # must look like Arabizi (latin + maybe digits), reasonable length, real sentence
    if not s or len(s) < 6 or len(s) > 200:
        return False
    letters = sum(c.isalpha() for c in s)
    return letters >= 4 and bool(re.search(r"[a-zA-Z]", s))

def emit(rows, instr_pool, x, y, task, script):
    rows.append({
        "instruction": random.choice(instr_pool).format(x=x),
        "output": y,
        "category": "real_lexicon",
        "task": task,
        "input_script": script,
        "synthetic": False,
        "needs_native_review": False,
        "source": "derja-english/lexicon",
    })

def main():
    rows, seen = [], set()
    n_examples = n_vocab = 0
    with open(LEX, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("is_vulgar"):
                continue
            ar  = clean(e.get("example_arabizi"))
            glo = clean(e.get("example_gloss"))
            eng = clean(e.get("english"))
            variants = e.get("arabizi_variants") or []

            # 1 + 2: real example sentence <-> its English gloss
            if ok_arabizi(ar) and glo and len(glo) > 4:
                key = ar.lower()
                if key not in seen:
                    seen.add(key)
                    emit(rows, TRANSLATE_INSTR, glo, ar, "translate", "en")
                    emit(rows, COMPREHEND_INSTR, ar, glo, "comprehend", "arabizi")
                    n_examples += 1

            # 3: short vocab gloss (only clean single-word variants with a real gloss)
            if variants and eng and 1 <= len(eng) <= 60 and " " not in variants[0] and len(variants[0]) >= 2:
                emit(rows, VOCAB_INSTR, variants[0], eng, "vocab", "arabizi")
                n_vocab += 1

    random.shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"real example sentences used : {n_examples}")
    print(f"vocab pairs                 : {n_vocab}")
    print(f"TOTAL pairs written         : {len(rows)}  -> {os.path.relpath(OUT, ROOT)}")
    print("\nsample:")
    for r in rows[:4]:
        print(f"  [{r['task']:9}] {r['instruction'][:48]:48}  ->  {r['output'][:48]}")

if __name__ == "__main__":
    main()
