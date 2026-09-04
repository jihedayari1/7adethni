"""
Real Tunisian CONVERSATIONAL pipeline  (scrape -> transliterate -> clean -> pairs).

This is the path to *coherence*: real human comment->reply threads, not synthetic.
It is SOURCE-AGNOSTIC — feed it raw threads from YouTube (dataset/tools/yt_comments.py),
a Facebook export (clean_facebook.py output), Reddit, or any JSON — and it produces
conversational training pairs in the SAME instruction/output schema the notebook trains on.

Stages
  1. ingest      raw threads -> (parent, child[, context]) text pairs
  2. pii_strip   remove @mentions, emails, URLs, phone numbers (keep prices/emojis)
  3. dialect     keep Tunisian; REJECT Moroccan/Algerian markers (bghit, daba, dyal...)
  4. arabizi     Arabic-script -> deterministic Arabizi (rag/normalizer); else normalize
  5. quality     real-word rate vs the 17k lexicon, length bounds, dedup
  6. structure   parent -> instruction ("Jaweb..."), child -> output

Run (offline demo, proves the transform):   python dataset/tools/build_real_conv.py
Run on real data:                            python dataset/tools/build_real_conv.py threads.jsonl [more.jsonl]
Out: aigenerateddataset/real_conv_pairs.jsonl
"""
import json, re, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(Path(__file__).parent))
from normalizer import transliterate_ar_to_arabizi, normalize     # noqa: E402
from coherence_score import load_vocab, score_text, norm, skel    # noqa: E402
LEXICON = ROOT / "rag" / "lexicon.jsonl"

OUT = ROOT / "aigenerateddataset" / "real_conv_pairs.jsonl"
random.seed(42)

# ---- filters -------------------------------------------------------------------
_AR        = re.compile(r"[؀-ۿ]")
_EMOJI_ONLY= re.compile(r"^[\W\d_]+$")
_TAG_ONLY  = re.compile(r"^(@\w+\s*)+$")
_EMAIL     = re.compile(r"\S+@\S+\.\S+")
_URL       = re.compile(r"https?://\S+|www\.\S+")
_MENTION   = re.compile(r"@\w+")
_PHONE     = re.compile(r"\b\d{8,}\b|\+216[\s-]?\d[\d\s-]{6,}")   # 8+ digit run = phone (prices are <=4)

# distinctive Moroccan / Algerian markers -> reject (Tunisian dialect only)
_MOROCCAN = re.compile(r"\b(bghit|baghi|bagha|daba|wakha|dyal|diali|dialk|bzaf|bzzaf|"
                       r"finta|fin\s|zwin|zwina|khasni|khassni|kayen|ghadi\s+\w*\s*chi|"
                       r"lhih|9bila|3afak|lwa9ila|drari|chwiya\s+dyal)\b", re.I)
# Tunisian signal -> require at least one (markers OR arabizi number-letters)
_TUN = re.compile(r"\b(barcha|barsha|famma|fama|chnowa|chnoua|9addech|9adech|bch|bech|taw|tawa|"
                  r"mta3|mte3|brabi|3aslema|3aslama|ya5i|5ouya|yezzi|fissa3|behi|mli7|3aychek|"
                  r"9ahwa|ghodwa|8odwa|win|wa9tech|3lech|kifech|hetha|hethi|9a3ed)\b", re.I)
_NUMLETTER = re.compile(r"[a-z][2-9]|[2-9][a-z]", re.I)

MIN_W, MAX_W = 3, 40
# self-promo / spam (very common in comments: "abonnez", "sub to my channel"...)
_PROMO = re.compile(r"\b(abonne\w*|abone\w*|s'abonner|subscribe|my channel|chaine|cha[iî]ne|"
                    r"jdid fil? youtoub|ena jdid|follow me|check my|lien fil? bio|promo code)\b", re.I)


def pii_strip(t: str) -> str:
    t = _EMAIL.sub(" ", t or "")
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _PHONE.sub(" [num] ", t)
    return re.sub(r"\s+", " ", t).strip()


def to_arabizi(t: str) -> str:
    if _AR.search(t):
        t = transliterate_ar_to_arabizi(t)
    return re.sub(r"\s+", " ", t).strip()


def is_tunisian(t: str) -> bool:
    if _MOROCCAN.search(t):
        return False
    return bool(_TUN.search(t) or _NUMLETTER.search(t))


def usable(t: str) -> bool:
    w = len(t.split())
    return MIN_W <= w <= MAX_W and not _EMOJI_ONLY.match(t) and not _TAG_ONLY.match(t)


# ---- toxicity filter (real comments are full of insults/profanity) -------------
# Unambiguous profanity/insult roots. Skeleton matching only fires for len>=4 skeletons so
# short roots don't collide with clean words (e.g. 3ars=pimp vs 3ors=wedding both -> '3rs').
_VULGAR_SEED = ("nik naik nayek nayk tnek tnekt tnayek tnaykou tnaikou tnekthom mnayek mnayka "
                "9a7ba 9a7be 9hab zebi zeb zab taftaf tmnik tmneek tkriz tkryz 5ra khra 5ara "
                "j3boun j3bun kahba 3aher 3ahra zamel putain connard enculer nike").split()

def load_vulgar():
    v = {norm(w) for w in _VULGAR_SEED}
    try:
        for line in open(LEXICON, encoding="utf-8"):
            e = json.loads(line)
            if e.get("is_vulgar"):
                for x in (e.get("arabizi_variants") or []):
                    v.add(norm(x))
    except Exception:
        pass
    v.discard("")
    return v, {skel(x) for x in v if len(skel(x)) >= 4}   # long skeletons only (avoid collisions)

_VULGAR, _VULGAR_SKEL = load_vulgar()

_ARTICLE = re.compile(r"^(el|al|l)")
_VOWDUP  = re.compile(r"([aeiou])\1+")           # nayeek->nayek, niiik->nik (evasion)

def is_clean(text: str) -> bool:
    for w in re.findall(r"[A-Za-z0-9']+", text or ""):
        k = _VOWDUP.sub(r"\1", norm(w))
        if len(k) < 3:
            continue
        for cand in (k, _ARTICLE.sub("", k)):     # also test article-stripped (al5ra -> 5ra)
            if cand in _VULGAR:                   # exact normalized match (safe for short roots)
                return False
            sk = skel(cand)
            if len(sk) >= 4 and sk in _VULGAR_SKEL:  # skeleton only for longer words
                return False
    return True


# ---- ingest (pluggable) --------------------------------------------------------
def ingest(path):
    """Yield (context, parent, child) from many possible raw shapes."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] \
        if str(path).endswith((".jsonl",)) else json.load(open(path, encoding="utf-8"))
    for r in rows:
        ctx = r.get("context") or r.get("post") or r.get("source_post") or ""
        # shape A: flat pair
        parent = r.get("question") or r.get("prompt") or r.get("parent") or r.get("comment")
        child  = r.get("answer")   or r.get("response") or r.get("reply") or r.get("child")
        if parent and child:
            yield ctx, parent, child
        # shape B: nested thread {post, comments:[{text, replies:[{text}]}]}
        for c in (r.get("comments") or []):
            ptxt = c.get("text", "")
            for rep in (c.get("replies") or []):
                if ptxt and rep.get("text"):
                    yield (r.get("post") or ""), ptxt, rep["text"]


# a tiny REAL-STYLE demo so the transform is provable with no external data
DEMO = [
    {"post": "el 7ar el yom barcha 🥵", "comments": [
        {"text": "شنوة الحكاية يا صاحبي، القهوة غالية برشة",
         "replies": [{"text": "والله نفس الحكاية، الستاق غدوة باش يقطع"}]},
        {"text": "3aslema, 9addech el livraison l Sfax?",
         "replies": [{"text": "b 7 dt w tousel ghodwa inchallah, 5ouya"}]},
    ]},
    {"post": "promo jdida", "comments": [
        {"text": "bghit nchri dyal hadchi bzaf",   # Moroccan -> must be REJECTED
         "replies": [{"text": "wakha a sahbi ghadi nchoufou"}]},
        {"text": "0612345678 contactez moi svp @page",  # PII -> stripped
         "replies": [{"text": "ميرسي برشة، خدمة محترمة"}]},
    ]},
]


def main():
    vocab, skels, _ = load_vocab()
    files = sys.argv[1:]
    triples = []
    if files:
        for f in files:
            triples += list(ingest(f))
    else:
        print("(no input files -> running built-in DEMO threads)\n")
        tmp = ROOT / "aigenerateddataset" / "_demo_threads.json"
        tmp.write_text(json.dumps(DEMO, ensure_ascii=False), encoding="utf-8")
        triples = list(ingest(tmp))
        tmp.unlink()

    templates = [
        "Jaweb 3al message hetha b tari9a 3adia w tabi3iya: {x}",
        "Chnowa tjaweb 3la: {x}",
        "Rodd 3al comment hetha bel tounsi: {x}",
    ]
    keep_translit = "--keep-translit" in sys.argv    # allow raw transliterated-Arabic targets
    pairs, seen, seen_child = [], set(), set()
    stats = {"in": 0, "rej_translit": 0, "rej_dialect": 0, "rej_vulgar": 0,
             "rej_spam": 0, "rej_quality": 0, "out": 0}
    for ctx, parent, child in triples:
        stats["in"] += 1
        child = pii_strip(child)
        # the REPLY is our training target: require it be NATIVELY Arabizi, not Arabic-script
        # transliterated to raw vowel-less text (those make poor generation targets).
        if not keep_translit and _AR.search(child):
            stats["rej_translit"] += 1; continue
        # PII strip -> transliterate -> THEN gate (markers must be in Arabizi to be seen)
        a_parent = to_arabizi(pii_strip(parent))
        a_child  = to_arabizi(child)
        # the REPLY (target) itself must be Tunisian — rejects French/Spanish/other replies
        if not is_tunisian(a_child) or _MOROCCAN.search(a_parent):
            stats["rej_dialect"] += 1; continue
        if not (is_clean(a_parent) and is_clean(a_child)):    # drop insults/profanity
            stats["rej_vulgar"] += 1; continue
        if _PROMO.search(a_child) or _PROMO.search(a_parent):  # drop self-promo / spam
            stats["rej_spam"] += 1; continue
        if not (usable(a_parent) and usable(a_child)):
            stats["rej_quality"] += 1; continue
        # child (the OUTPUT we train on) must be real Tunisian words, not noise
        rate, _oov, _n = score_text(a_child, vocab, skels)
        if rate < 0.6:
            stats["rej_quality"] += 1; continue
        child_key = normalize(a_child)[:40]                   # cross-thread copy-paste spam
        if child_key in seen_child:
            stats["rej_spam"] += 1; continue
        key = (a_parent[:50], a_child[:50])
        if key in seen:
            continue
        seen.add(key); seen_child.add(child_key)
        instr = random.choice(templates).format(x=a_parent)
        if ctx:
            instr = f"(Context: {to_arabizi(pii_strip(ctx))[:120]})\n" + instr
        pairs.append({"instruction": instr, "output": a_child, "category": "conversational",
                      "task": "reply", "input_script": "arabizi", "synthetic": False,
                      "needs_native_review": True, "source": "real_conv"})
        stats["out"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"ingested {stats['in']} | rejected: translit-arabic {stats['rej_translit']}, "
          f"dialect {stats['rej_dialect']}, vulgar {stats['rej_vulgar']}, "
          f"spam {stats['rej_spam']}, quality {stats['rej_quality']} | KEPT {stats['out']}")
    print(f"-> {OUT.relative_to(ROOT)}\n")
    for p in pairs[:6]:
        print("  Q:", p["instruction"].replace(chr(10), " ")[:74])
        print("  A:", p["output"][:74], "\n")


if __name__ == "__main__":
    main()
