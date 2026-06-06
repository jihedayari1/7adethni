#!/usr/bin/env python3
"""
MSA-leakage detector — the HEADLINE metric (report section 2.5 / 3).

Measures what % of outputs are genuine Tunisian Derja vs formal Arabic (MSA/fos7a),
the thing that quietly kills dialect models. Heuristic for v1 (no training needed);
upgrade to a learned classifier later when there's labeled data.

Usage:
  python dataset/tools/msa_leakage.py                       # self-test on examples
  python dataset/tools/msa_leakage.py preds.jsonl           # score a model's outputs
  python dataset/tools/msa_leakage.py preds.jsonl --field output
"""
import json, re, sys, argparse

# --- Tunisian (Derja) markers: words/features that appear in Derja but NOT in MSA ---
TUNISIAN = {
    "barcha", "famma", "famma", "chnowa", "chnoua", "9addech", "9adech", "kifech", "kifach",
    "3lech", "3lesh", "win", "wa9tech", "ya5i", "5ouya", "mte3", "mta3", "bch", "taw", "tawa",
    "ken", "kima", "brabi", "3aslama", "3aslema", "marhba", "chwaya", "barka", "fissa3", "zin",
    "behi", "bahi", "mli7", "3andi", "3andou", "3andek", "hethi", "hetha", "hakka", "haka",
    "sa7a", "3aychek", "yezzi", "lawej", "9a3ed", "mch", "mouch", "moch", "ma3andich", "manich",
    "n7eb", "t7eb", "9olli", "na7ki", "yaani", "ya3ni", "fel", "lel", "el", "w", "9ahwa",
    "el li", "ча", "el yom", "ghodwa", "lyoum", "enti", "enta", "ena", "houwa", "hia", "a7na",
    "naw", "9addch", "wella", "wala", "5dma", "5edma", "dar", "bnin", "tounsi", "tounes",
}
# strong feature: Arabizi numerals used as letters (7keya, 5dma, 3andi, 9ahwa, 8roub)
_NUMLETTER = re.compile(r"[a-z][3-9'][a-z]|^[3-9][a-z]|[a-z][3-9]$", re.I)
# French/English code-switch (very common in Derja, never in MSA)
FRENCH = {"livraison", "prix", "commande", "merci", "bonjour", "stock", "weekend", "promo",
          "garantie", "couleur", "taille"}

# --- MSA markers: formal Arabic that should NOT appear in fluent Derja ---
MSA_LATIN = {
    "hadha", "hadhihi", "alladhi", "allati", "sawfa", "laysa", "lasna", "kayfa", "limadha",
    "3indama", "7inama", "ladhalika", "lakinna", "jiddan", "kathiran", "qalilan", "yumkinu",
    "yajibu", "na7nu", "antum", "ayyuha", "inna", "anna", "sayakun", "yakunu", "dhalika",
    "tilka", "hunaka", "faqat", "aydan", "ladayna", "ladayhi", "sa", "lan", "lam", "qad",
    "ka2anna", "bayd", "wa9ad", "thumma", "indama",
}
MSA_AR = ["الذي", "التي", "سوف", "ليس", "كيف", "عندما", "لذلك", "يمكن", "يجب", "نحن",
          "جداً", "جدا", "كثيراً", "كثيرا", "ذلك", "هذا", "هذه", "هؤلاء", "إنّ", "حينما"]

_ARABIC = re.compile(r"[؀-ۿ]")
_WORD = re.compile(r"[a-z0-9'7359]+", re.I)


def score_text(text: str) -> dict:
    """Classify one output: tunisian | mixed | msa_leak | arabic_script | unknown."""
    t = (text or "").strip()
    has_ar = bool(_ARABIC.search(t))
    toks = set(w.lower() for w in _WORD.findall(t))
    tun = len(toks & TUNISIAN) + len(toks & FRENCH) + len(_NUMLETTER.findall(t.lower()))
    msa = len(toks & MSA_LATIN) + sum(t.count(m) for m in MSA_AR)

    if has_ar and not (toks & TUNISIAN):
        label = "arabic_script"                 # output should be Arabizi, not Arabic letters
    elif tun == 0 and msa == 0:
        label = "unknown"                        # too short / no signal
    elif msa > 0 and tun == 0:
        label = "msa_leak"
    elif tun > 0 and msa == 0:
        label = "tunisian"
    else:
        label = "tunisian" if tun >= 2 * msa else "mixed"
    score = tun / (tun + msa) if (tun + msa) else 0.0
    return {"label": label, "tun": tun, "msa": msa, "dialect_score": round(score, 2),
            "has_arabic": has_ar}


def score_file(path: str, field: str):
    labels = {}
    n = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = r.get(field) or r.get("output") or r.get("prediction") or r.get("response") or ""
        lab = score_text(out)["label"]
        labels[lab] = labels.get(lab, 0) + 1
        n += 1
    if not n:
        print("no rows scored"); return
    print(f"scored {n} outputs from {path}:")
    for lab in ("tunisian", "mixed", "msa_leak", "arabic_script", "unknown"):
        c = labels.get(lab, 0)
        print(f"  {lab:14}: {c:5}  ({c/n:.0%})")
    headline = labels.get("tunisian", 0) / n
    print(f"\n  >> DIALECT RATE (headline): {headline:.0%}  (target: as high as possible; "
          f"watch msa_leak + arabic_script)")


def _selftest():
    cases = [
        ("famma barcha 5dma el yom w 9ahwa m3a s7abi", "tunisian"),
        ("hadha alladhi sawfa yakunu jiddan kathiran", "msa_leak"),
        ("el livraison b 7 dt w tousel ghodwa", "tunisian"),
        ("نحن سوف نذهب الى المدرسة", "arabic_script"),
        ("ok", "unknown"),
    ]
    print("self-test:")
    for txt, exp in cases:
        got = score_text(txt)["label"]
        print(f"  {'OK ' if got==exp else 'XX '} expect {exp:13} got {got:13} | {txt[:45]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="?", help="jsonl of model outputs to score")
    ap.add_argument("--field", default="output")
    args = ap.parse_args()
    if args.preds:
        score_file(args.preds, args.field)
    else:
        _selftest()


if __name__ == "__main__":
    main()
