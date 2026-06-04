# Building a Tunisian Derja / Arabizi LLM
## Dataset Requirements & RAG Design Report

---

## 1. Scope and the decisions that shape everything

The goal is a model that **converses naturally in Tunisian Derja, primarily in Arabizi** (Latin-script: *"chnowa a7welek"*), rather than a model that merely understands the dialect or translates it. That single goal has consequences that ripple through every dataset choice below, so two decisions need to be locked first.

**Decision 1 — the target script.** Arabizi (Latin + numerals) is how most Tunisians actually type online, but it is *even more* low-resource than Arabic-script Derja and, critically, has **no standardized spelling** ("chnowa / chnoua / chniwa", "7" vs "h"). Almost every existing dataset and dialect model is Arabic-script. This means you will likely need a **transliteration bridge** (Arabic-script Derja ↔ Arabizi) because the data lives on the Arabic-script side, while your output target is Arabizi.

**Decision 2 — the base model.** The two realistic candidates behave very differently:
- **Labess-7b-chat** is dialect-adapted but outputs **Arabic script**, not Arabizi.
- **ALLaM-7B-Instruct** is a clean, strong base but **MSA-centric** — out of the box it answers in formal Arabic, not Derja.

Neither produces fluent Arabizi today. Whichever you choose, the dataset must do the heavy lifting of shifting it toward Arabizi output.

**The honest framing:** this is a low-resource, non-standardized target. **Data quality, a fixed spelling convention, and native review matter far more than raw volume.** A small, clean, consistent dataset beats a large noisy one here.

---

## 2. The datasets you need

You need five distinct datasets. People often collect only one and wonder why the model doesn't talk well. Each trains a different capability.

| # | Dataset | What it trains | Arabizi on output side? | Size target (v1) | Priority |
|---|---------|----------------|--------------------------|-------------------|----------|
| 1 | Raw monolingual corpus | Fluency, vocabulary, code-switching | n/a (plain text) | tens of millions of tokens | Medium |
| 2 | Instruction / conversation pairs | **Talking** (the core) | **Yes — essential** | 3,000–10,000 pairs | **Highest** |
| 3 | Parallel / translation data | Surface forms + comprehension anchoring | Both directions | 5,000–20,000 pairs | High |
| 4 | Lexicon / dictionary | Feeds the RAG layer + spelling normalization | n/a (entries) | 10,000–50,000 entries | High (for RAG) |
| 5 | Evaluation set | Measuring quality honestly | Yes | 300–500 gold items | High |

### 2.1 Raw monolingual corpus

**Purpose.** Continual pre-training so the model absorbs the rhythm, vocabulary, and French/English code-switching of real Derja before you teach it to follow instructions. Optional for a first version, but it raises the ceiling.

**Schema.** Plain text with light metadata: `{ text, script (arabizi|arabic), source, region? }`.

**Sources (license-clean, no scraping needed).** The ~800k-row unified Derja corpus on Hugging Face; `linagora/fineweb2_Tunisian_Arabic`; OSCAR and OPUS Tunisian slices; TUNIZI and the large Arabizi sentiment corpora for Latin-script text.

**Caveats.** Most of this is Arabic-script. To grow the *Arabizi* side you will need transliteration (Section 3 normalizer) or Arabizi-native sources (social media text, which carries scraping/privacy constraints — collect only public text, strip personal data).

### 2.2 Instruction / conversation pairs — the core that makes it *talk*

**Purpose.** This is the dataset that turns a model that *recognizes* Derja into one that *converses* in it. The decisive property: the **response (output) side must be in Arabizi/Derja**. A model learns to produce a language by being trained to produce it — not by seeing a translation beside it.

**Two formats, both useful:**

*Native Arabizi conversation* — input and output both Derja/Arabizi:
```json
{ "instruction": "3tini fekra l ftour sa7i",
  "output": "tnajem tekol chwaya cho7an m3a 7lib w fawekeh, wala bidh maslou9 m3a khobz kamel..." }
```

*Cross-lingual instruction* — input in a language the base already understands (EN/FR/MSA), output forced into Arabizi. This is one of the strongest techniques for a low-resource **output** language, because comprehension is anchored in a high-resource language:
```json
{ "instruction": "Reply in Tunisian Arabizi: give me a healthy breakfast idea",
  "output": "tnajem tekol chwaya cho7an m3a 7lib w fawekeh..." }
```

**How to build it.**
1. *Translate open instruction datasets* into Arabizi, then have native speakers fix them. Fast bootstrap.
2. *Synthetic generation* with a strong LLM, then human filtering. Watch hard for MSA leakage.
3. *Human-written* pairs — the highest value. Even 1,000 genuinely native pairs are gold.

**Quality control.** Native-speaker review is non-negotiable here. The dominant failure mode is data that is secretly MSA with a few Tunisian words sprinkled in — train on that and the model sounds like a news anchor.

### 2.3 Parallel / translation data

**Purpose.** Two real benefits: it teaches the model the **surface forms** of Arabizi (how spellings map to meaning), and it anchors comprehension cross-lingually. Translation fine-tuning is a documented way to adapt a model to dialectal surface forms. *Note:* this trains translation/comprehension — it supports fluency but does **not** by itself make the model conversational. Pair it with 2.2.

**Schema.**
```json
{ "arabizi": "chnowa el 7keya",
  "arabic_script": "شنوة الحكاية",
  "french": "c'est quoi l'histoire",
  "english": "what's the story" }
```

**Sources (human-translated, reusable).**
- **MADAR** — parallel sentences for Tunis and Sfax dialects aligned with English, French, and MSA. This is the closest match to what you want, already done by humans.
- **TEDxTN** — Tunisian Arabic ↔ English speech translation, 108 talks, ~25 hours, code-switched, multiple regions.
- The 2025 benchmark with parallel **Tunizi (Latin) / Arabic-script / English** rows plus sentiment labels.
- WANLP Tunisian parallel resources (TD–MSA aligned corpora).

**Augment synthetically, legally.** Take monolingual Derja (2.1) and translate it with a *strong LLM* (much better at Derja than general MT engines, and far better than relying on social-platform auto-translation, which is poor for Derja and against most platforms' terms). Then human-verify a sample. Do **not** train a translator on another system's translations as your only source — you would inherit its errors.

### 2.4 Lexicon / dictionary — and the foundation of your RAG

**Purpose.** A structured word/phrase store. It does double duty: it powers the RAG retrieval layer (Section 4) and provides the spelling-variant data you need for normalization.

**Schema (designed for RAG + Arabizi variation):**
```json
{ "arabizi_variants": ["barcha", "barsha", "barša"],
  "arabic_script": "برشة",
  "french": "beaucoup",
  "english": "a lot / many",
  "pos": "adverb",
  "example_arabizi": "famma barcha nes",
  "example_gloss": "there are a lot of people",
  "region": "general" }
```
Storing **multiple spelling variants per entry** is the single most important design choice — it is what makes retrieval work despite Arabizi having no fixed spelling.

**Sources (openly licensed).**
- **Wiktionary** Tunisian Arabic entries — CC BY-SA, downloadable and reusable with attribution. The cleanest legal word-level source.
- The **MADAR** Tunis/Sfax lexicon (aligned to French and English).
- The WANLP human-checked Derja lexicon (~44k entries).
- For any third-party dictionary site: **ask the owner for permission / an export** rather than scraping — several block automated access in their robots.txt, and a curated dictionary is a copyrighted compilation.

### 2.5 Evaluation set

**Purpose.** Measuring quality honestly, and catching MSA leakage. Hand-build it, keep it small, and **never train on it.**

**Contents.** ~300–500 prompts spanning your real use cases (chat, Q&A, storytelling, translation) with reference answers in Arabizi.

**Metrics.**
- **Dialect rate / MSA-leakage** — a classifier measuring what % of outputs are actually Tunisian vs formal Arabic. Track this as your **headline metric**; it is the thing that quietly kills dialect models.
- **chrF++ and BERTScore** against references for fidelity.
- **Human native-speaker ratings** — the real ground truth.

---

## 3. Cross-cutting data quality controls

These apply to every dataset above and matter more than any single source.

**Fix an Arabizi spelling convention first.** Because Arabizi has no standard, inconsistent targets will cap your model's quality no matter how much data you collect. Either adopt explicit conventions (decide "7" vs "h", French-style vs English-style vowels, etc.) or run every example through a **normalizer** that maps variants to a canonical form. This normalizer is reused at RAG query time (Section 4), so build it once, well.

**Make MSA-leakage your tracked metric**, not an afterthought — measure it on every data batch and every model checkpoint.

**Native-speaker review loop.** Budget real human time. Even reviewing a sample of each batch catches systemic problems early.

**Licensing hygiene.** Record the license of every source in a manifest. Prefer CC / research-released / self-collected data. Avoid sources that disallow automated access; avoid training a redistributable dataset on copyrighted compilations without permission.

**Hygiene passes.** De-duplicate, strip personal data from any social-media text, and filter obvious noise before training.

---

## 4. The RAG layer — design and honest limits

You want a retrieval layer over a Tunisian Arabizi/Derja knowledge base so the model can look up words it doesn't understand. This is a good idea **for the right job** — but be precise about what RAG can and cannot do here.

**What RAG is good for:** supplying *meanings* — rare slang, regional terms, idioms, named entities, and factual grounding. It injects knowledge the model lacks.

**What RAG cannot do:** make the model *fluent* or fix its *style*. Retrieving a definition does not teach the model to phrase things like a Tunisian. **Fluency comes from fine-tuning (Section 2.2); RAG complements it, it does not replace it.** Keep both in the plan.

### 4.1 The knowledge base

Primarily the **lexicon from Section 2.4** (variant-rich entries with glosses and examples), optionally extended with an idiom/proverb store and any factual documents you want the model to be able to cite.

### 4.2 The hard part: retrieving over Arabizi

Standard RAG assumes you can match a query to the knowledge base. Arabizi breaks that assumption because the same word is spelled many ways and is usually **out-of-vocabulary for general embedding models**. Three mechanisms, used together (hybrid retrieval):

1. **Normalize both sides.** Run the query *and* the lexicon keys through the same normalizer (Section 3) so "barsha" and "barcha" collapse to one form before matching.
2. **Store spelling variants** per entry (already in the 2.4 schema) so exact/near-exact matches succeed.
3. **Fuzzy + semantic matching.** Combine character-level fuzzy matching (e.g., edit distance / BM25 over normalized strings) with multilingual sentence embeddings. For Arabizi, **lexical/fuzzy matching often outperforms pure embeddings** because embedders barely know the script — so weight the lexical signal heavily and use embeddings as a backstop. Optionally add a phonetic key (sound-alike matching) for stubborn cases.

### 4.3 Detecting "words the model doesn't understand"

You cannot directly read the model's confusion, so use practical proxies, simplest first:
- **Lexicon lookup over content tokens** — for each non-trivial token in the user's message, check the lexicon; if it's a known slang/regional entry, retrieve its gloss. Robust and cheap.
- **A maintained slang/OOV list** — curated terms that always trigger retrieval.
- **Low token-probability signals** (advanced) — flag tokens the model assigns low confidence to.

For v1, the lexicon-lookup proxy is the pragmatic choice.

### 4.4 The end-to-end pipeline

```
user message (Arabizi)
  → normalize
  → detect candidate unknown/slang tokens (lexicon lookup)
  → retrieve top-k lexicon entries (hybrid: fuzzy + embeddings)
  → inject as a context block into the prompt:
       "Meanings: barcha = a lot; 7keya = story/matter ..."
  → model generates a grounded reply in Arabizi
```

This reframes the model's job from "recall what this slang means" (which it fails at) to "use these provided meanings to answer" (which it does well) — the same principle that fixes factual hallucination by grounding.

### 4.5 Build order for the RAG layer

Lexicon (2.4) → normalizer (3) → hybrid retriever → prompt injection → evaluate retrieval hit-rate on a held-out set of slang queries.

---

## 5. Suggested collection roadmap

**Phase 0 — Decide.** Lock the Arabizi spelling convention and the base model (compare Labess and ALLaM raw outputs side by side first).

**Phase 1 — Gather existing open data.** Unified Derja corpus, `fineweb2_Tunisian_Arabic`, MADAR (Tunis/Sfax ↔ EN/FR/MSA), TEDxTN, the Tunizi/Arabic/English benchmark, Wiktionary lexicon. This alone covers raw text, parallel data, and the lexicon seed.

**Phase 2 — Build the lexicon + RAG.** Assemble the variant-rich lexicon, build the normalizer, stand up the hybrid retriever. You get a usable vocabulary-lookup system before any training.

**Phase 3 — Build the instruction set.** Translate open instruction data → Arabizi, generate synthetic pairs, write native pairs; review everything; track MSA-leakage. This is where most of your effort should go.

**Phase 4 — Build the evaluation set.** 300–500 gold items, held out.

**Phase 5 — Train and iterate.** LoRA/QLoRA fine-tune the chosen base on the instruction + parallel mix; evaluate with the dialect-rate classifier, chrF++/BERTScore, and human ratings; iterate on the weakest axis.

---

## 6. Priorities in one paragraph

If you do only three things: (1) fix an Arabizi spelling convention and build the normalizer, because inconsistency caps everything; (2) invest the most effort in **instruction/conversation pairs with Arabizi on the output side**, because that is what makes the model talk; and (3) build the **lexicon first**, since it serves both the RAG layer and normalization. Treat RAG as a vocabulary/knowledge aid layered on top of a fine-tuned model — not as a substitute for the fine-tuning that produces fluency. Everything else (raw corpus, parallel data) raises the ceiling but is secondary to these three.
