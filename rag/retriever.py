"""
Hybrid Arabizi retriever over the lexicon (report sections 4.2 - 4.4).

Pipeline:  normalize -> detect candidate slang tokens (lexicon lookup)
           -> hybrid match (exact-normalized + fuzzy; optional semantic backstop)
           -> build a "Meanings: ..." context block to inject into the LLM prompt.

Lexical is primary (report: for Arabizi, fuzzy/lexical beats pure embeddings because
embedders barely know the script). Semantic embeddings are an OPTIONAL backstop.
"""
import json, re, sys
from pathlib import Path
from rapidfuzz import process, fuzz

sys.path.insert(0, str(Path(__file__).parent))
from normalizer import normalize

LEX = Path(__file__).parent / "lexicon.jsonl"

# function words we don't bother glossing (they're not the "unknown" tokens)
_STOP = {
    "el", "le", "la", "fi", "fil", "f", "w", "o", "ou", "ana", "enti", "enta",
    "houa", "hia", "7na", "homa", "ken", "kima", "3la", "men", "min", "ila",
    "ma", "mch", "mech", "mouch", "moch", "yes", "no", "ok", "lel", "bin",
    "ya", "wala", "wela", "ama", "9bal", "ba3d", "hatta", "barra", "haka",
}
_TOK = re.compile(r"[a-zA-Z0-9']+")
_VOWELS = str.maketrans("", "", "aeiouy")

def _skeleton(s: str) -> str:
    """Consonant skeleton: drop vowels (digits 3/5/7/9 are consonants, kept)."""
    return s.translate(_VOWELS)


class ArabiziRetriever:
    def __init__(self, lexicon_path: Path = LEX, use_semantic: bool = False):
        self.entries = [json.loads(l) for l in open(lexicon_path, encoding="utf-8")]
        self.exact: dict[str, int] = {}
        self.keys: list[str] = []
        self.owner: list[int] = []
        for i, e in enumerate(self.entries):
            for v in e.get("arabizi_variants", []):
                nk = normalize(v)
                if not nk:
                    continue
                self.exact.setdefault(nk, i)
                self.keys.append(nk)
                self.owner.append(i)
        self.use_semantic = use_semantic
        self._emb = None
        if use_semantic:
            self._init_semantic()

    # ---------------------------------------------------------------- semantic backstop
    def _init_semantic(self):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np
        self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        corpus = [
            " ".join(filter(None, [e.get("english"), e.get("french"),
                                   " ".join(e.get("arabizi_variants", [])[:3])]))
            for e in self.entries
        ]
        self._emb = self._model.encode(corpus, normalize_embeddings=True,
                                       show_progress_bar=True, batch_size=256)

    # ---------------------------------------------------------------- single token
    def lookup(self, token: str, fuzzy_cutoff: int = 84):
        """Return (entry, score, how) for one token, or None.
        Fuzzy candidates must also pass a consonant-skeleton check, which kills the
        false positives that vowel-less (Arabic-derived) variants would otherwise cause.
        """
        nk = normalize(token)
        if not nk:
            return None
        if nk in self.exact:                                   # 1) exact normalized
            return self.entries[self.exact[nk]], 100, "exact"
        qskel = _skeleton(nk)
        best = None                                            # 2) fuzzy + skeleton gate
        for match, score, idx in process.extract(nk, self.keys, scorer=fuzz.ratio,
                                                  limit=15, score_cutoff=fuzzy_cutoff - 4):
            skel_ratio = fuzz.ratio(qskel, _skeleton(match))
            if skel_ratio < 78:
                continue
            same_first = nk[:1] == match[:1]
            composite = 0.6 * score + 0.4 * skel_ratio + (5 if same_first else 0)
            if best is None or composite > best[0]:
                best = (composite, score, idx)
        if best and best[0] >= fuzzy_cutoff:
            return self.entries[self.owner[best[2]]], int(best[1]), "fuzzy"
        return None

    # ---------------------------------------------------------------- detect + retrieve
    def detect_candidate_tokens(self, message: str) -> list[str]:
        toks = [t.lower() for t in _TOK.findall(message)]
        return [t for t in toks if len(t) >= 2 and t not in _STOP and not t.isdigit()]

    def retrieve(self, message: str, k: int = 5, fuzzy_cutoff: int = 82):
        hits, seen = [], set()
        for tok in self.detect_candidate_tokens(message):
            r = self.lookup(tok, fuzzy_cutoff)
            if not r:
                continue
            entry, score, how = r
            eid = id(entry)
            if eid in seen:
                continue
            seen.add(eid)
            hits.append({"token": tok, "score": score, "how": how, "entry": entry})
        # optional semantic backstop if lexical found little
        if self.use_semantic and len(hits) < k:
            hits += self._semantic(message, k - len(hits), seen)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    def _semantic(self, message: str, k, seen):
        if self._emb is None or k <= 0:
            return []
        q = self._model.encode([normalize(message)], normalize_embeddings=True)
        sims = (self._emb @ q[0])
        out = []
        for idx in self._np.argsort(-sims)[: k + len(seen)]:
            e = self.entries[idx]
            if id(e) in seen:
                continue
            out.append({"token": "(semantic)", "score": float(sims[idx] * 100),
                        "how": "semantic", "entry": e})
            if len(out) >= k:
                break
        return out

    # ---------------------------------------------------------------- prompt injection
    @staticmethod
    def build_context_block(hits) -> str:
        if not hits:
            return ""
        lines = []
        for h in hits:
            e = h["entry"]
            gloss = e.get("english") or e.get("french") or "?"
            disp = e["arabizi_variants"][0] if e.get("arabizi_variants") else e.get("arabic_script", "")
            lines.append(f"{disp} = {gloss}")
        return "Meanings: " + "; ".join(lines)


if __name__ == "__main__":
    semantic = "--semantic" in sys.argv
    r = ArabiziRetriever(use_semantic=semantic)
    print(f"Loaded {len(r.entries)} entries, {len(r.keys)} variant keys, semantic={semantic}\n")
    tests = [
        "famma barcha nes fel 7afla",
        "chnowa el 7keya",
        "3tini 9ahwa 3al kahw",
        "ya5i 5ouya win mchit ?",
    ]
    for msg in tests:
        print("Q:", msg)
        hits = r.retrieve(msg, k=5)
        for h in hits:
            e = h["entry"]
            print(f"   [{h['how']:7} {h['score']:3}] {h['token']:10} -> "
                  f"{e['arabizi_variants'][:2]} ({e.get('english')})")
        print("   >>", r.build_context_block(hits))
        print()
