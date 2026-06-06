# Training strategy — how to get the BEST Tunisian model (not just "a" model)

**Honest framing:** SFT on instruction pairs is the necessary CORE and the right first step,
but it is NOT the complete best-results recipe alone. The best dialectal models use a fuller
pipeline. We already own data for every stage. The #1 lever is DATA QUALITY + DIVERSITY +
NATIVE REVIEW, more than any training trick (proven by GemMaroc matching SOTA with quality SFT).

## Full pipeline (best results)
| Stage | Purpose | Our data | When |
|-------|---------|----------|------|
| 0. Base model | reasoning/knowledge "brain" | Qwen3-8B (Apache 2.0) | chosen |
| 1. Continual Pre-Training (CPT) | deep Tunisian understanding/fluency from raw text | 3 GB corpus chunks | **v2** (heavy compute) |
| 2. SFT on instruction pairs | teach it to talk / follow instructions / answer in Arabizi | cs_pairs.jsonl (+ converted parallel) | **v1 — core, now** |
| 3. Preference tuning (DPO) | prefer good answers, cut MSA leakage further | from review/eval | **v2** (polish) |
| 4. Evaluation loop | measure dialect-rate / MSA-leakage so we KNOW it improves | 300-item held-out eval | **v1 — essential** |
| 5. RAG at runtime | correct facts (prices, slang) w/o hallucination | rag/lexicon.jsonl | built ✅ |

## What matters MOST (the real quality levers)
1. **Data quality + diversity + native-naturalness** (biggest lever by far).
2. **Native review** of a real sample (you + friends) — separates "good" from "slightly off".
3. **Keep ~10–20% English/MSA pairs** in SFT → prevents catastrophic forgetting of reasoning.
4. **A real eval set + MSA-leakage metric** → the measure→improve loop that makes "excellent" reachable.
5. Strong base model + dual-script data (Arabic-letter & FR/EN input → Arabizi output).

## What quietly RUINS quality (avoid)
- Skipping native review → sounds subtly off, MSA leaks.
- No eval set → flying blind, can't tell if changes helped.
- Too-narrow / repetitive data → narrow, robotic model.
- Expecting SFT to inject KNOWLEDGE → that's the base model + RAG's job, not SFT.
- Over-tuning until it forgets how to reason (no English mix).

## The staged plan
**v1 (now) — SFT done right:**
1. Grow pairs: convert the **Tunisian** parallel source (derja-english 18k → 32k comprehension/
   translation pairs) — NOTE: clean_darija_english.csv is **Moroccan** Darija and is EXCLUDED
   (would corrupt the model; a Moroccan-marker filter enforces this). Keep generating diverse
   conversational Tunisian pairs (the priority). For Tunisian translation at scale, download
   MADAR (Tunis/Sfax), TEDxTN, or WANLP Tunisian.
2. Native-review a strong sample (review.py).
3. Mix ~10–20% English/MSA pairs.
4. Build 300-item eval set + MSA-leakage classifier.
5. QLoRA fine-tune Qwen3-8B on free Kaggle; measure; fix weakest axis; repeat.
6. RAG for facts at runtime.

**v2 (level-up, after v1 works):** add CPT on the 3 GB corpus (deeper fluency) + DPO (polish).

## Reality check on "perfect"
No v1 is perfect. The METHOD to reach excellent is iterative: build → measure dialect-rate/
MSA-leakage on the held-out eval → fix the weakest axis → retrain. The eval set is what turns
"perfect" from a vibe into a target you can actually hit.
