"""
Retrieval hit-rate on a tiny held-out set of slang queries (report section 4.5).
Extend GOLD with native-reviewed items over time. Run: python rag/eval_retrieval.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from retriever import ArabiziRetriever

# (arabizi token, substring that must appear in the retrieved english gloss)
GOLD = [
    ("barcha", "lot"),     ("chnowa", "what"),   ("7keya", "story"),
    ("9ahwa", "coffee"),   ("5ouya", "brother"), ("win", "where"),
    ("3aslama", "hello"),  ("famma", "mouth"),   ("ghodwa", "tomorrow"),
    ("ya5i", "brother"),   ("sa7a", "health"),   ("9ahwa", "coffee"),
    ("kahw", "enough"),    ("flouss", "money"),  ("3ich", "live"),
]

def main():
    r = ArabiziRetriever()
    ok = 0
    for tok, expect in GOLD:
        res = r.lookup(tok)
        gloss = (res[0].get("english") or "").lower() if res else ""
        hit = expect.lower() in gloss
        ok += hit
        print(f"{'HIT ' if hit else 'MISS'} {tok:10} expect '{expect}'  ->  {gloss[:55]}")
    print(f"\nhit-rate: {ok}/{len(GOLD)} = {ok/len(GOLD):.0%}")

if __name__ == "__main__":
    main()
