"""
Tunisian Arabizi normalizer + Arabic-script -> Arabizi transliterator.

Built once, reused everywhere (report sections 3 and 4.2):
  * normalize()                  -> canonical key for matching ("barcha"=="barsha"=="barchaaa")
  * transliterate_ar_to_arabizi  -> seed an Arabizi spelling from Arabic-script lexicon entries
  * generate_variants            -> the multiple spelling variants stored per lexicon entry

Spelling convention chosen (documented, French-background Tunizi):
  sh,$  -> ch        |  ou,oo -> u        |  ee -> i        |  y(vowel) -> i
  7'    -> 5 (kh)    |  2     -> dropped at match time (hamza)
  digits 3,5,7,9 are PHONEMIC and kept (they are letters, not noise).
Normalization is intentionally lossy: it is a *matching* key, not a display form.
"""
import re
import unicodedata

# ---------------------------------------------------------------- normalize
_REPEAT = re.compile(r"(.)\1{2,}")          # 3+ repeats -> collapse to 1 (barchaaaa)
_DOUBLE = re.compile(r"(.)\1")              # then collapse remaining doubles (mhh)
_NONWORD = re.compile(r"[^a-z0-9]+")

# order matters: longer / more specific first
_VARIANT_MAP = [
    ("$", "ch"), ("sh", "ch"), ("tch", "ch"),
    ("7'", "5"), ("kh", "5"),
    ("gh", "8"),                # fold gh<->8 so "maghrib"/"ma8rib" collapse (internal key only)
    ("ph", "f"),
    ("ou", "u"), ("oo", "u"), ("ww", "w"),
    ("ee", "i"), ("ei", "i"), ("ai", "i"),
    ("dj", "j"),
]

# accented letters that carry a CONSONANT sound -> fix before accent stripping
_PRECOMPOSE = [("š", "ch"), ("ž", "j"), ("č", "ch"), ("ġ", "gh"), ("ḥ", "7"),
               ("ḫ", "5"), ("ç", "s"), ("ʕ", "3"), ("ʔ", "2")]

def normalize(text: str) -> str:
    """Canonical matching key. Apply to BOTH the query and the lexicon keys."""
    if not text:
        return ""
    text = text.lower()
    for a, b in _PRECOMPOSE:
        text = text.replace(a, b)
    # strip accents (café -> cafe) but keep base letters
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _REPEAT.sub(r"\1", text)
    for a, b in _VARIANT_MAP:
        text = text.replace(a, b)
    text = text.replace("2", "")            # hamza marker is noise for matching
    text = _DOUBLE.sub(r"\1", text)
    # vowel skeletonization is too lossy for short words; keep vowels but unify o/u
    text = text.replace("o", "u")
    text = _NONWORD.sub(" ", text).strip()
    return text


# ------------------------------------------------ Arabic script -> Arabizi
# Primary Tunizi transliteration (one best guess per letter).
_AR2LAT = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ٱ": "a",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "5",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "ch",
    "ص": "s", "ض": "dh", "ط": "t", "ظ": "dh", "ع": "3", "غ": "gh",
    "ف": "f", "ق": "9", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "i", "ى": "a", "ة": "a", "ء": "2",
    "ئ": "2", "ؤ": "2", "پ": "p", "ڤ": "v", "ڨ": "g", "چ": "ch",
    "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "",
    "ِ": "", "ّ": "", "ْ": "", "ـ": "",   # tashkeel + tatweel
}
# letters that often surface as a vowel when long/medial
_ALT = {
    "و": ["w", "ou", "u", "o"],
    "ي": ["i", "y", "ee"],
    "ق": ["9", "k", "q"],
    "غ": ["gh", "8"],
    "خ": ["5", "kh"],
    "ج": ["j", "dj"],
    "ة": ["a", "et", ""],
}

def transliterate_ar_to_arabizi(arabic: str) -> str:
    """Best-guess single Arabizi spelling for an Arabic-script word/phrase."""
    out = []
    for ch in arabic:
        if ch.isspace():
            out.append(" ")
        else:
            out.append(_AR2LAT.get(ch, ch if ch.isalnum() else ""))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def generate_variants(arabic: str, max_variants: int = 6) -> list[str]:
    """Produce several plausible Arabizi spellings for an Arabic word.
    Used to populate `arabizi_variants` so retrieval succeeds despite no fixed spelling.
    """
    forms = [""]
    for ch in arabic:
        if ch.isspace():
            forms = [f + " " for f in forms]
            continue
        opts = _ALT.get(ch)
        if opts is None:
            base = _AR2LAT.get(ch, "")
            forms = [f + base for f in forms]
        else:
            new = []
            for f in forms:
                for o in opts:
                    new.append(f + o)
                    if len(new) >= max_variants * 3:
                        break
            forms = new
    # clean, dedupe, cap
    seen, out = set(), []
    for f in forms:
        f = re.sub(r"\s+", " ", f).strip()
        if f and f not in seen:
            seen.add(f)
            out.append(f)
        if len(out) >= max_variants:
            break
    return out


if __name__ == "__main__":
    for w in ["barcha", "barsha", "barchaaaa", "barša", "chnowa", "chnoua", "7keya"]:
        print(f"{w:12} -> {normalize(w)}")
    print()
    for ar in ["برشة", "شنوة", "الحكاية", "قهوة", "غدوة"]:
        print(f"{ar}  ->  {transliterate_ar_to_arabizi(ar)}   variants={generate_variants(ar)}")
