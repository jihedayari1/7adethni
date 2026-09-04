"""
Deterministic grounding / quality layer for 7adethni generation.

The model (even base 7B) drifts into invented words on long outputs. Instead of
trusting it, we score every generation against the 17k-word Tunisian lexicon and:
  * quality(text)      -> real-word rate + the OOV ("no meaning") words
  * canonicalize(text) -> safe display fixes for the Arabizi number-rules
  * input_meanings(x)  -> a "Meanings: ..." block so the model UNDERSTANDS slang input

app.py uses quality() to do best-of-N regeneration: keep the most coherent candidate.
No training required — this is pure inference-time quality control.
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from normalizer import normalize          # noqa: E402
from retriever import ArabiziRetriever, _STOP  # noqa: E402

# French / English loanwords Tunisians write verbatim (not in the Derja lexicon)
_LOAN = set((
    "livraison prix commande merci bonjour stock weekend promo garantie couleur taille "
    "produit model facture clim steg cash style post reply caption message photo video film "
    "cafe coffee telephone batterie portable internet wifi page profil compte budget offre code "
    "ok pack collection stock service client adresse total "
    # currency / unit abbreviations + a few high-frequency forms the lexicon misses
    "dt tnd dh km kg ml cm mejjeni mejjeniya mejjania djej dejaj esmou"
).split())

_TOK = re.compile(r"[A-Za-z0-9']+")
_retriever = None

def _R() -> ArabiziRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ArabiziRetriever(use_semantic=False)
    return _retriever


def quality(text: str):
    """Return {rate, oov, n} — fraction of content words that are REAL Tunisian words."""
    r = _R()
    toks = [t for t in _TOK.findall(text or "") if not t.isdigit()]
    content = [t for t in toks if len(normalize(t)) >= 2 and t.lower() not in _STOP]
    if not content:
        return {"rate": 1.0, "oov": [], "n": 0}
    real, oov = 0, []
    for t in content:
        if t.lower() in _LOAN or r.lookup(t, fuzzy_cutoff=82) is not None:
            real += 1
        else:
            oov.append(t)
    return {"rate": real / len(content), "oov": oov, "n": len(content)}


# ----------------------------------------------------- safe number-rule canonicalization
# Only the UNAMBIGUOUS Arabizi forms: $ -> ch, 7' -> 5. We deliberately do NOT touch
# 'kh'/'sh' (some users prefer them) or digits/inflection, to avoid corrupting meaning.
def canonicalize(text: str) -> str:
    if not text:
        return text
    text = text.replace("7'", "5").replace("$", "ch")
    return text


def input_meanings(user_text: str, k: int = 5) -> str:
    """'Meanings: foo = ...; bar = ...' for slang in the user's input (comprehension).
    Uses a HIGH fuzzy cutoff: a wrong gloss ('9addech = toddle') misleads the model,
    so we only inject confident matches and drop the rest.
    """
    r = _R()
    hits = [h for h in r.retrieve(user_text or "", k=k, fuzzy_cutoff=92)
            if h["how"] == "exact" or h["score"] >= 92]
    return r.build_context_block(hits)


def pick_best(candidates):
    """Given [(text, ...)], return index of the most coherent by real-word rate,
    tie-broken by fewer OOV words then shorter length."""
    scored = []
    for i, c in enumerate(candidates):
        q = quality(c)
        scored.append((q["rate"], -len(q["oov"]), -len(c), i, q))
    scored.sort(reverse=True)
    best = scored[0]
    return best[3], best[4]


if __name__ == "__main__":
    samples = [
        "3aslema, el livraison l Sfax b 6 dt w mejjeniya ki el commande tfout 100 dt",
        "kan famma wa7ed esmou Souleiman kan yfta7 kol 7aja b sibr el najje9 njaafolek el chabbi",
        "chwaya djej zit fi so5ba b chwaya 3jja w bhar mle7, fissa3 w bnin",
    ]
    for s in samples:
        q = quality(s)
        print(f"[{q['rate']:5.0%} real / {q['n']:2d} content]  {s[:55]}")
        if q["oov"]:
            print(f"        OOV: {', '.join(q['oov'])}")
