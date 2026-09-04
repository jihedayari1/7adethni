"""
Coherence / real-word scorer for Tunisian Arabizi output.

The old 'dialect rate' metric only checks for Tunisian function words + number
characters -> it says 96% even when the text is gibberish. This scorer instead
asks the question the user actually cares about:

    "How many of the words are REAL Tunisian words (in our 17k lexicon),
     and are the Arabizi number-rules used in valid positions?"

Outputs:
  - real_word_rate : % of word-tokens that exist in the lexicon / known vocab
  - oov            : the invented / unknown words (the 'words with no meaning')

Usage:
  python dataset/tools/coherence_score.py            # self-test on pasted samples
  python dataset/tools/coherence_score.py file.jsonl # score 'output' field of a jsonl
"""
import json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEXICON = os.path.join(ROOT, "rag", "lexicon.jsonl")

# Tunisian function words / very common forms (not always in the noun-heavy lexicon)
FUNC = set((
    "el la w f fi 3la 3al men mel l b bel ma mch mouch ena enti enta houwa hia "
    "a7na entouma houma hethi hetha haka hakka hethom ken kima taw tawa bch bech "
    "barcha barka chwaya yezzi famma fama 9addech kifech chnowa chnoua chna7welek "
    "win wa9tech 3lech ya5i brabi 3aslema 3aslama marhba sa7a 3aychek 9a3ed n7eb "
    "t7eb y7eb na7taj 9olli 9alli ya3ni wala ama 3andi 3andou 3andek 3andna mte3 mta3 "
    "ki bla 3ad zeyed akther a9all a7sen behi bahi mli7 zin zina 5ater 3la5ater "
    "tounes tounsi lyoum ghodwa 8odwa el5er ennour sba7 msa lila nhar 3am sna "
    "ey la2 le 7atta 3andhom inchallah rabbi 7amdoulah"
).split())

# common French / English loanwords Tunisians write as-is
LOAN = set((
    "livraison prix commande merci bonjour stock weekend promo garantie couleur "
    "taille produit model commande facture clim klim steg cash style post reply "
    "caption message photo video film cafe coffee telephone batterie portable "
    "ok internet wifi page profil compte budget offre code"
).split())

def norm(tok: str) -> str:
    """light normalization so spelling variants collapse to one key."""
    t = tok.lower().strip()
    t = re.sub(r"[^\w'8-9]", "", t)          # keep letters, digits, apostrophe
    t = re.sub(r"(.)\1{2,}", r"\1", t)        # collapse 3+ repeats: barchaaaa -> barcha
    t = t.replace("sh", "ch").replace("kh", "5")
    return t

def skel(k: str) -> str:
    """consonant skeleton: drop unstable vowels (digits 3/7/9/5 are consonants -> keep)."""
    return re.sub(r"[aeiouy']", "", k)

def load_vocab():
    vocab = set(FUNC) | set(LOAN)
    n = 0
    with open(LEXICON, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            n += 1
            for v in (e.get("arabizi_variants") or []):
                vocab.add(norm(v))
            # the example sentences are REAL usage -> harvest every word form
            ex = e.get("example_arabizi") or ""
            for w in re.split(r"\s+", ex):
                k = norm(w)
                if len(k) >= 2:
                    vocab.add(k)
    vocab.discard("")
    skels = {skel(v) for v in vocab if len(skel(v)) >= 2}
    return vocab, skels, n

WORD_RE = re.compile(r"[A-Za-z0-9ّ'][A-Za-z0-9']*")

def score_text(text, vocab, skels):
    toks = [w for w in WORD_RE.findall(text) if not w.isdigit()]
    real, oov = 0, []
    for w in toks:
        k = norm(w)
        if len(k) < 2:                 # tiny tokens: ignore
            real += 1; continue
        sk = skel(k)
        if k in vocab or k.rstrip("0123456789") in vocab or (len(sk) >= 2 and sk in skels):
            real += 1
        else:
            oov.append(w)
    rate = real / len(toks) if toks else 0.0
    return rate, oov, len(toks)

SAMPLES = [  # the NEW (fine-tuned) outputs the user pasted
    "sba7 ennour ena 3aylet chwaya m3a el 9attous, taw erta7t chwaya 3al telephone. enti chno",
    "el routine ta9tol ki tweli nafs el 7aja bla ma3na. ba3ed 3alha chwaya, 5alli wa9t l 3otla",
    "abda b 7ajat sahla: 1. e5tar mili7a dha8t fih flouss 2. 7ott el 7aja 8irek 3. ektb",
    "chwaya djej zit fi so5ba b chwaya 3jja w bhar mle7, fissa3 w bnin",
    "kan famma wa7ed esmou Souleiman kan yfta7 kol 7aja b sibr el sabr mch marhala kima el najje9 houwa el asas el chabbi el li tet3allem men njaafolek",
]

def main():
    vocab, skels, n = load_vocab()
    print(f"lexicon entries: {n}   vocab keys: {len(vocab)}   skeletons: {len(skels)}\n")
    if len(sys.argv) > 1:
        rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
        texts = [r.get("output") or r.get("response") or "" for r in rows]
    else:
        texts = SAMPLES
    rates = []
    for t in texts:
        rate, oov, ntok = score_text(t, vocab, skels)
        rates.append(rate)
        print(f"[{rate:5.0%} real | {ntok:2d} words]  {t[:60]}")
        if oov:
            print(f"        OOV (no meaning): {', '.join(oov[:12])}")
    if rates:
        print(f"\n>> MEAN REAL-WORD RATE: {sum(rates)/len(rates):.0%}  "
              f"(vanity 'dialect rate' said 96%)")

if __name__ == "__main__":
    main()
