"""
Clean + pair Facebook comment scrapes into customer-service training pairs.

  * STRIPS all personal data (profileName/Id/Picture) — names are only used momentarily
    to detect "is this the shop replying", then discarded.
  * Builds  question (customer top-level comment) -> answer (reply, page reply preferred).
  * Filters noise (emoji-only, @tags, too short, non-questions).

Usage:  python dataset/tools/clean_facebook.py  <raw.json> [raw2.json ...]
        (defaults to the existing scrape in the project root)
Output: dataset/collected/facebook_pairs.jsonl  + printed yield stats.
"""
import json, re, sys, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "collected" / "facebook_pairs.jsonl"

PII_FIELDS = {"profileName", "profileId", "profilePicture", "facebookId", "feedbackId"}
_EMOJI_ONLY = re.compile(r"^[\W\d_]+$")
_TAG_ONLY = re.compile(r"^(@\w+\s*)+$")
_QWORD = re.compile(r"\b(chnowa|chnoua|9adech|9adesh|9addech|kifech|win|wa9tech|3lech|3lesh|"
                    r"famma|3andkom|disponible|dispo|prix|combien|livraison|taman|"
                    r"sizes?|mqas|couleur|9addh|wa9t|comment|svp|stp)\b", re.I)


def is_question(t: str) -> bool:
    return ("?" in t) or bool(_QWORD.search(t))


def good_answer(t: str) -> bool:
    if len(t.split()) < 2:
        return False
    if _EMOJI_ONLY.match(t) or _TAG_ONLY.match(t):
        return False
    return True


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def pairs_from_file(path):
    data = json.load(open(path, encoding="utf-8"))
    page = (data[0].get("groupTitle") or data[0].get("postTitle") or "").strip() if data else ""
    pairs, last_q = [], None
    for c in data:
        depth = c.get("threadingDepth", 0) or 0
        text = clean_text(c.get("text"))
        is_page = bool(page) and (c.get("profileName", "").strip() == page)
        post = clean_text(c.get("postTitle"))
        if not text:
            continue
        if depth == 0:
            last_q = {"context": post, "question": text} if is_question(text) else None
        elif last_q and good_answer(text):
            pairs.append({
                "context": last_q["context"] or None,
                "question": last_q["question"],
                "answer": text,
                "is_page_answer": is_page,
                "likes": int(c.get("likesCount") or 0),
                "source_page": page or None,
                "source": "facebook",
            })
            last_q = None  # one answer per question (the first/closest reply)
    return pairs, len(data)


def main():
    files = sys.argv[1:] or glob.glob(str(ROOT / "dataset_facebook*.json"))
    if not files:
        print("No raw facebook file found. Pass a path.")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_pairs, raw_total = [], 0
    for f in files:
        p, n = pairs_from_file(f)
        raw_total += n
        all_pairs += p
        print(f"  {Path(f).name}: {n} comments -> {len(p)} pairs")
    # dedup
    seen, uniq = set(), []
    for p in all_pairs:
        k = (p["question"][:60], p["answer"][:60])
        if k not in seen:
            seen.add(k); uniq.append(p)
    with open(OUT, "w", encoding="utf-8") as out:
        for p in uniq:
            out.write(json.dumps(p, ensure_ascii=False) + "\n")
    page_ans = sum(p["is_page_answer"] for p in uniq)
    print(f"\nRAW comments: {raw_total}")
    print(f"CLEAN pairs : {len(uniq)}  (yield {len(uniq)/max(raw_total,1):.0%}; "
          f"{page_ans} are shop replies)")
    print(f"PII stripped: {', '.join(sorted(PII_FIELDS))}")
    print(f"-> {OUT.relative_to(ROOT)}\n")
    for p in uniq[:5]:
        print(f"  Q: {p['question'][:70]}")
        print(f"  A: {p['answer'][:70]}  (page={p['is_page_answer']})\n")


if __name__ == "__main__":
    main()
