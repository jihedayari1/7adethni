"""
7adethni — prompt templates per writing feature.

The fine-tuned model was trained on instruction -> Tunisian-Arabizi-output. Each feature here
wraps the user's input into the instruction the model expects. Shared by the serving API and tests.
"""

SYSTEM = (
    "Enti '7adethni', musa3ed kteba tounsi. Tektb DIMA bel derja tounsiya bel arabizi "
    "(7ourouf latiniya w arqam), b tari9a tabi3iya kima el touensa el 7a9i9iyin. "
    "3tini barka el nass el matloub, bla char7 zeyed w bla mou9addima."
)

# UI tone value -> Arabizi descriptor injected into the instruction
TONES = {
    "normal": "3adi w tabi3i",
    "funny":  "morfeh w fih chwaya dhe7k",
    "formal": "rasmi w m7taram",
    "promo":  "promo, yjbed el client w ybi3",
    "short":  "9sir barka, jomla wala thnin",
    "warm":   "7anin w 9rib mel 9alb",
}

# feature -> builds the user-turn instruction. {t}=tone descriptor, {x}=user input
FEATURES = {
    "post":      "Ekteb post l social media 3al mawdou3 hetha, b tone {t}: {x}",
    "reply":     "Jaweb 3al message hetha b tari9a {t}: {x}",
    "rewrite":   "3awed sou8 el nass hetha bel tounsi b tone {t}, 5allih a7la: {x}",
    "translate": "9olha bel tounsi (arabizi), tabi3iya: {x}",
    "caption":   "A3tini caption 9sir w jthab l tswira/fekra hethi, tone {t}: {x}",
    "idea":      "{x}",
}

FEATURE_LABELS = {
    "post": "Write a post", "reply": "Reply to a message", "rewrite": "Rewrite / improve",
    "translate": "Translate to Derja", "caption": "Caption", "idea": "Ask / free text",
}


def build_messages(feature: str, text: str, tone: str = "normal", context: str = ""):
    """Return [system, user] chat messages for the requested feature.
    `context` is an optional 'Meanings: ...' block (from the RAG retriever) that helps
    the model understand slang in the user's input; it is added to the system turn.
    """
    if feature not in FEATURES:
        feature = "idea"
    t = TONES.get(tone, TONES["normal"])
    user = FEATURES[feature].format(t=t, x=(text or "").strip())
    system = SYSTEM
    if context:
        system = SYSTEM + "\n\n(Mou3jam ysa3dek tefhem el input — esta3mlou ken yelzem)\n" + context
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
