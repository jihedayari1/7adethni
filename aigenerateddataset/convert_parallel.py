#!/usr/bin/env python3
"""
Convert existing PARALLEL data into instruction pairs (no API, no GPU).

Source: ../clean_darija_english.csv  (darija = real Arabizi, + english)
Produces two directions:
  * EN -> Arabizi  (instruction in English, OUTPUT in Arabizi)  -> teaches Arabizi OUTPUT ⭐
  * Arabizi -> EN  (sampled)                                    -> teaches comprehension
Optional (--lexicon): ../derja-english.csv -> "what does <arabic-word> mean" (Arabic-script
  INPUT -> English meaning) -> teaches Arabic-script comprehension.

Output: aigenerateddataset/parallel_pairs.jsonl  (kept SEPARATE from cs_pairs.jsonl on purpose;
  we mix a sampled subset with the hand-curated conversational pairs at training time).

Same quality bar + dedup as the rest of the pipeline.
"""
import csv, json, re, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aigenerateddataset"))
sys.path.insert(0, str(ROOT / "rag"))
from generate_cs_dataset import enforce_convention, is_clean
from normalizer import normalize
csv.field_size_limit(10**7)

OUT = ROOT / "aigenerateddataset" / "parallel_pairs.jsonl"
_AR = re.compile(r"[؀-ۿ]")

# varied instruction templates so the model doesn't overfit one phrasing
EN2AR = [
    "Reply in Tunisian Arabizi: {x}",
    "Translate to Tunisian Derja (Arabizi): {x}",
    "Say this in Tunisian, Latin letters: {x}",
    "9olha bel tounsi (arabizi): {x}",
    "Kifech n9oulou '{x}' bel derja?",
]
AR2EN = [
    "Translate to English: {x}",
    "What does this mean in English? {x}",
    "Chnowa ma3natha bel anglais: {x}",
]
MEANING = [
    "Chnowa ma3na '{x}'?",
    "What does '{x}' mean?",
    "3allemni chnowa ma3na: {x}",
]


def clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def row_ok(a, b, lo=2, hi=300):
    return a and b and lo <= len(a) <= hi and lo <= len(b) <= hi


def emit(rows, key_seen, out_f, instruction, output, direction, source):
    instruction, output = clean(instruction), enforce_convention(output) if direction == "en2ar" else clean(output)
    if not is_clean(instruction, output):
        return 0
    k = (normalize(instruction), normalize(output)[:120])
    if k in key_seen:
        return 0
    key_seen.add(k)
    out_f.write(json.dumps({
        "instruction": instruction, "output": output,
        "category": "translation" if direction != "meaning" else "lexicon",
        "topic": direction,
        "input_script": "arabic" if _AR.search(instruction) else "latin",
        "human_parallel": True, "needs_native_review": False,
        "source": source,
    }, ensure_ascii=False) + "\n")
    return 1


def load_existing_keys():
    seen = set()
    cs = ROOT / "aigenerateddataset" / "cs_pairs.jsonl"
    if cs.exists():
        for l in open(cs, encoding="utf-8"):
            try:
                r = json.loads(l)
                seen.add((normalize(r.get("instruction", "")), normalize(r.get("output", ""))[:120]))
            except json.JSONDecodeError:
                pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar2en-every", type=int, default=2,
                    help="emit Arabizi->EN for every Nth row (default 2 = half). 0 = none.")
    ap.add_argument("--lexicon", action="store_true",
                    help="also convert derja-english.csv (Arabic-script word -> English meaning)")
    ap.add_argument("--lexicon-max", type=int, default=12000)
    args = ap.parse_args()

    seen = load_existing_keys()
    n_en2ar = n_ar2en = n_meaning = 0
    f = open(OUT, "w", encoding="utf-8")

    # ---- clean_darija_english.csv : real Arabizi <-> English ----
    with open(ROOT / "clean_darija_english.csv", encoding="utf-8") as fp:
        for i, r in enumerate(csv.DictReader(fp)):
            ar, en = clean(r.get("darija")), clean(r.get("english"))
            if not row_ok(ar, en):
                continue
            if _AR.search(ar):           # darija side must be Arabizi (Latin), skip stray Arabic
                continue
            n_en2ar += emit(None, seen, f, EN2AR[i % len(EN2AR)].format(x=en), ar, "en2ar", "clean_darija_english")
            if args.ar2en_every and i % args.ar2en_every == 0:
                n_ar2en += emit(None, seen, f, AR2EN[i % len(AR2EN)].format(x=ar), en, "ar2en", "clean_darija_english")

    # ---- derja-english.csv : Arabic-script word -> English meaning (optional) ----
    if args.lexicon:
        with open(ROOT / "derja-english.csv", encoding="utf-8") as fp:
            for i, r in enumerate(csv.DictReader(fp)):
                if n_meaning >= args.lexicon_max:
                    break
                term = clean(r.get("term_in_arabic_normalized") or r.get("term_in_arabic"))
                meaning = clean(r.get("definition_in_english"))
                if not row_ok(term, meaning, lo=2, hi=200):
                    continue
                n_meaning += emit(None, seen, f, MEANING[i % len(MEANING)].format(x=term), meaning, "meaning", "derja-english")
    f.close()

    total = n_en2ar + n_ar2en + n_meaning
    print(f"WROTE {total} parallel instruction pairs -> {OUT.relative_to(ROOT)}")
    print(f"  EN->Arabizi (output Arabizi) : {n_en2ar}")
    print(f"  Arabizi->EN (comprehension)  : {n_ar2en}")
    print(f"  Arabic-word->meaning (lexicon): {n_meaning}")
    print("\nKept SEPARATE from cs_pairs.jsonl. At training: use ALL conversational pairs +")
    print("a SAMPLED subset of these (e.g. 5-10k) so translation doesn't drown out chatting.")


if __name__ == "__main__":
    main()
