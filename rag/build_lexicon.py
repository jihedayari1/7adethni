"""
Build the RAG lexicon (report section 2.4) from the data already in this folder.

Inputs (in parent folder):
  - derja-english.csv   : 18k Arabic-script Tunisian entries (rich: POS, french, vulgar, example)
  - chunk_16.json       : ~400 EN->Tunisian dictionary snippets (with transliteration)

Output:
  - rag/lexicon.jsonl   : one entry per line, schema below, with GENERATED arabizi_variants
  - rag/lexicon.stats.json

Schema per entry:
  { arabizi_variants:[...], arabic_script, french, english, pos,
    example_arabizi, example_gloss, region, is_vulgar, source }
"""
import csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import transliterate_ar_to_arabizi, generate_variants, normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "lexicon.jsonl"
csv.field_size_limit(10**7)


def from_derja_english(path: Path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ar = (row.get("term_in_arabic_normalized") or row.get("term_in_arabic") or "").strip()
            if not ar:
                continue
            variants = generate_variants(ar, max_variants=6)
            if not variants:
                continue
            ex_ar = (row.get("example_sentence_in_arabic_normalized") or "").strip()
            yield {
                "arabizi_variants": variants,
                "arabic_script": ar,
                "french": (row.get("french") or "").strip() or None,
                "english": (row.get("definition_in_english") or "").strip() or None,
                "pos": (row.get("part_of_speech") or "").strip() or None,
                "example_arabizi": transliterate_ar_to_arabizi(ex_ar) if ex_ar else None,
                "example_gloss": (row.get("example_sentence_in_english") or "").strip() or None,
                "region": "general",
                "is_vulgar": str(row.get("is_vulgar")).strip().lower() == "true",
                "is_from_french": str(row.get("is_from_french")).strip().lower() == "true",
                "source": "derja-english",
            }
            n += 1
    print(f"  derja-english.csv -> {n} entries")


# chunk_16 rows look like:
#  "headword" (Language) In English In transliterated Tounsi بالتونسي "headword"(in English) <gloss> <arabic> <translit> <example...>
_AR = re.compile(r"[؀-ۿ]")
_HEAD = re.compile(r'^\s*[""“]?([a-zA-Z][a-zA-Z \-/]+?)[""”]?\s*\(')

def from_chunk16(path: Path):
    if not path.exists():
        return
    data = json.load(open(path, encoding="utf-8"))
    n = 0
    for rec in data:
        t = (rec.get("text") or "").replace("‏", " ")
        head = _HEAD.match(t)
        # arabic-script tokens in the line
        ar_tokens = re.findall(r"[؀-ۿً-ْ]+", t)
        ar_tokens = [w for w in ar_tokens if w not in ("بالتونسي",) and len(w) > 1]
        if not ar_tokens:
            continue
        ar = max(ar_tokens, key=len)  # the headword is usually the standalone (longest) token
        english = head.group(1).strip() if head else None
        yield {
            "arabizi_variants": generate_variants(ar, max_variants=6),
            "arabic_script": ar,
            "french": None,
            "english": english,
            "pos": None,
            "example_arabizi": None,
            "example_gloss": None,
            "region": "general",
            "is_vulgar": False,
            "is_from_french": False,
            "source": "chunk_16",
        }
        n += 1
    print(f"  chunk_16.json -> {n} entries")


def main():
    seen = set()
    written = 0
    sources = {}
    with open(OUT, "w", encoding="utf-8") as out:
        for gen in (from_derja_english(ROOT / "derja-english.csv"),
                    from_chunk16(ROOT / "chunk_16.json")):
            for e in gen:
                key = normalize(e["arabic_script"]) + "|" + (e["english"] or "")
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(e, ensure_ascii=False) + "\n")
                written += 1
                sources[e["source"]] = sources.get(e["source"], 0) + 1
    stats = {"total_entries": written, "by_source": sources}
    json.dump(stats, open(Path(__file__).parent / "lexicon.stats.json", "w"), indent=2)
    print(f"\nWROTE {written} entries -> {OUT.name}")
    print("stats:", stats)


if __name__ == "__main__":
    main()
