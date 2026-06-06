#!/usr/bin/env python3
"""
Convert TUNISIAN parallel data into instruction pairs (no API, no GPU).

IMPORTANT — dialect safety:
  clean_darija_english.csv was found to be MOROCCAN Darija (gha-/ta- prefixes, bghit, wakha,
  dyal, ch7al...) — NOT Tunisian — so it is NOT used here (it would corrupt the model).
  Only derja-english.csv is used: verified Tunisian (باش / إلي / Espérance...).
  A Moroccan-marker filter rejects any contaminated row as defense-in-depth.

Source: ../derja-english.csv (Tunisian lexicon, Arabic script + English + example sentences)
Produces comprehension / translation pairs (Arabic-script Tunisian INPUT -> English OUTPUT):
  * "Chnowa ma3na <word>?"            -> english definition   (Arabic-script understanding)
  * "Translate to English: <example>" -> english translation  (Arabic-script comprehension)

Output: aigenerateddataset/parallel_pairs.jsonl  (kept SEPARATE; sampled subset mixed at training).

NOTE: for genuine Tunisian *translation* data at scale, the right sources are MADAR (Tunis/Sfax
<-> EN/FR/MSA), TEDxTN, and WANLP Tunisian — download those when ready.
"""
import csv, json, re, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aigenerateddataset"))
sys.path.insert(0, str(ROOT / "rag"))
from generate_cs_dataset import is_clean
from normalizer import normalize
csv.field_size_limit(10**7)

OUT = ROOT / "aigenerateddataset" / "parallel_pairs.jsonl"
_AR = re.compile(r"[؀-ۿ]")

# --- Moroccan/Algerian markers (latin + arabic script) -> reject any row containing them ---
_MOR_LAT = re.compile(r"\b(bghit|bgha|baghi|daba|wakha|wa5a|dyal|dial|ghadi|ghadya|kayn|kayna|"
                      r"khass|khassni|3afak|bzaf|chno|chnou|ch7al|wach|gha[ndy]|ta[ynt]|kay[a-z])\b", re.I)
_MOR_AR = ["غادي", "بغيت", "دابا", "واخا", "ديال", "شحال", "كاين", "خاصني", "بزاف", "اشنو", "واش"]

def is_moroccan(*texts) -> bool:
    blob = " ".join(t or "" for t in texts)
    if _MOR_LAT.search(blob):
        return True
    return any(m in blob for m in _MOR_AR)

MEANING = [
    "Chnowa ma3na '{x}'?",
    "What does '{x}' mean?",
    "3allemni chnowa ma3na: {x}",
]
TRANSLATE = [
    "Translate to English: {x}",
    "What does this mean in English? {x}",
]


def clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())


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


def emit(seen, out_f, instruction, output, topic):
    instruction, output = clean(instruction), clean(output)
    if not is_clean(instruction, output):
        return 0
    k = (normalize(instruction), normalize(output)[:120])
    if k in seen:
        return 0
    seen.add(k)
    out_f.write(json.dumps({
        "instruction": instruction, "output": output,
        "category": "lexicon" if topic == "meaning" else "translation",
        "topic": topic,
        "input_script": "arabic" if _AR.search(instruction) else "latin",
        "dialect": "tunisian", "human_parallel": True, "needs_native_review": False,
        "source": "derja-english",
    }, ensure_ascii=False) + "\n")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="cap total pairs (0 = no cap)")
    args = ap.parse_args()

    seen = load_existing_keys()
    n_meaning = n_translate = rejected = 0
    f = open(OUT, "w", encoding="utf-8")
    with open(ROOT / "derja-english.csv", encoding="utf-8") as fp:
        for i, r in enumerate(csv.DictReader(fp)):
            term = clean(r.get("term_in_arabic_normalized") or r.get("term_in_arabic"))
            meaning = clean(r.get("definition_in_english"))
            ex_ar = clean(r.get("example_sentence_in_arabic_normalized"))
            ex_en = clean(r.get("example_sentence_in_english"))
            if is_moroccan(term, ex_ar):              # dialect safety net
                rejected += 1
                continue
            if 2 <= len(term) <= 200 and 2 <= len(meaning) <= 200:
                n_meaning += emit(seen, f, MEANING[i % len(MEANING)].format(x=term), meaning, "meaning")
            if 4 <= len(ex_ar) <= 250 and 4 <= len(ex_en) <= 250:
                n_translate += emit(seen, f, TRANSLATE[i % len(TRANSLATE)].format(x=ex_ar), ex_en, "ar2en")
            if args.max and (n_meaning + n_translate) >= args.max:
                break
    f.close()
    total = n_meaning + n_translate
    print(f"WROTE {total} TUNISIAN parallel pairs -> {OUT.relative_to(ROOT)}")
    print(f"  word->meaning (Arabic-script comprehension): {n_meaning}")
    print(f"  example->English (translation)             : {n_translate}")
    print(f"  rejected (Moroccan-marker safety filter)    : {rejected}")
    print("\nKept SEPARATE from cs_pairs.jsonl. These are Arabic-input -> English-output")
    print("(comprehension/translation). Mix a SAMPLED subset at training; Arabizi-OUTPUT")
    print("behavior comes from the conversational pairs. For Tunisian translation at scale,")
    print("download MADAR (Tunis/Sfax), TEDxTN, or WANLP Tunisian.")


if __name__ == "__main__":
    main()
