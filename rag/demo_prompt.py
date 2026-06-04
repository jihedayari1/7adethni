"""
End-to-end RAG prompt assembly (report section 4.4).

  user message (Arabizi)
    -> normalize -> detect slang tokens -> retrieve lexicon meanings
    -> inject a "Meanings:" block into the system prompt
    -> (hand to the chosen base model: Labess / ALLaM, fine-tuned)

This file does NOT call an LLM (no base model wired yet); it prints the exact
augmented prompt you would send, so the grounding is inspectable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from retriever import ArabiziRetriever

SYSTEM = (
    "Enti assistant tunsi. Jaweb DIMA bel derja tunisiya bel arabizi "
    "(7ourouf latiniya w arqam). Esta3mel el ma3ani el provided ken famma."
)

def build_prompt(retriever, user_msg, k=5):
    hits = retriever.retrieve(user_msg, k=k)
    ctx = retriever.build_context_block(hits)
    system = SYSTEM + (("\n\n[" + ctx + "]") if ctx else "")
    return system, user_msg, hits


if __name__ == "__main__":
    r = ArabiziRetriever()
    for msg in ["chnowa el 7keya, famma barcha 5edma lyoum?",
                "3tini fekra l ftour sa7i"]:
        system, user, hits = build_prompt(r, msg)
        print("=" * 70)
        print("USER   :", user)
        print("-" * 70)
        print("SYSTEM :", system)
        print("(retrieved", len(hits), "meanings)\n")
