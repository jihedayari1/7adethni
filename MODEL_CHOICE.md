# Base model comparison — Tunisian Derja/Arabizi assistant

**Goal:** fine-tune (LoRA/QLoRA) a base model so it talks + understands Tunisian Derja in
**Arabizi** (Latin + numbers, heavy French code-switch), for a sellable, self-hosted product.

## The decisive insight
The output is **Arabizi (Latin script)**, not Arabic script. So the best base is a strong
**multilingual** model with good **French + English + some Arabic** and a clean Latin tokenizer —
NOT necessarily a pure-Arabic model. Arabic-script-centric models (Jais, ALLaM, Labess) are
great for Arabic-script Derja but fight you on Arabizi.

## Open vs API (the "paid/unpaid" axis)
- **Open-weight (unpaid weights, you pay only compute):** you FINE-TUNE, OWN, and SELF-HOST the
  result. No per-token fee, no vendor lock-in, full control. **Right choice for a product you sell.**
- **Closed/API (paid):** fine-tune via the vendor, but it stays hosted — you pay per token forever,
  can't self-host, can't truly own it. Faster to prototype, higher raw intelligence, but bad unit
  economics when you serve many stores. Note: **Claude/Anthropic does not offer weight fine-tuning.**

---

## Candidates

### Open-weight (recommended for this product)

| Model | Size(s) | Arabic | French / code-switch | Arabizi fit | License (sell?) | FT tooling | Notes |
|---|---|---|---|---|---|---|---|
| **Qwen2.5 / Qwen3 Instruct** ⭐ | 0.5–72B (use **7–8B**) | good | good | **very good** | **Apache 2.0** ✅ | excellent (Unsloth/Axolotl/LLaMA-Factory) | Best all-round: multilingual, strong reasoning, shines on low-resource SFT, clean tokenizer. **Top pick.** |
| **Gemma 2 / Gemma 3** | 2–27B (use **9B**) | good | good | very good | Gemma license (commercial ✅) | excellent | Efficient, strong multilingual + Arabic. Great alternate. |
| **Mistral NeMo 12B** | 12B | moderate | **excellent** (French-native vendor) | very good | **Apache 2.0** ✅ | excellent | Best French — helpful since Tunisian leans French. Slightly bigger to serve. |
| **Llama 3.1 8B Instruct** | 8B (also 70B) | moderate | good | good | Llama license (commercial ✅, <700M MAU) | **biggest ecosystem** | Tons of dialect-FT precedent + tooling. Safe, popular. |
| **Jais 13B / 30B** | 13–30B | **best raw Arabic** | weak | weak (Arabic-script-centric) | Apache 2.0 (13B) | ok | Excellent for **Arabic-script** Derja; not ideal for Arabizi. Heavier. |
| **ALLaM 7B Instruct** | 7B | strong (MSA) | weak | weak | check license | ok | MSA-centric (report's note); needs heavy steering to dialect + Arabizi. |

### Closed / API (paid, hosted — prototype only)

| Model | Fine-tune? | Own/self-host | Cost model | Verdict for this product |
|---|---|---|---|---|
| **GPT-4o-mini** | yes (OpenAI FT) | ❌ hosted | pay to train + **pay per token forever** | Fast prototype, strong, but lock-in + bad economics at scale |
| **Gemini Flash** | yes (Vertex tuning) | ❌ hosted | pay per token | Same trade-off as above |
| **Claude** | ❌ no weight FT | ❌ | per token | Use for *generating data* (you already do), not as the fine-tune base |

---

## Recommendation

**Primary: Qwen2.5-7B-Instruct (or Qwen3-8B).** Apache 2.0 (sell freely), strong multilingual
incl. French/English/Arabic, great with small SFT datasets, best tooling. Sweet spot of quality
vs. cheap serving.

**Alternates:** Gemma-2-9B (efficient), Mistral-NeMo-12B (max French), Llama-3.1-8B (ecosystem).

**Size:** start at **7–9B** — quantized (4-bit) it serves cheap/fast and is plenty for a Tunisian
CS + chat product. Prototype possible on 3–4B; 70B only if you later need max quality.

**Avoid as primary (for an Arabizi target):** Jais / ALLaM / Labess — reach for these only if you
also target **Arabic-script** Derja.

## Why this matters for power/efficiency
- The base gives the **brains** (knowledge, reasoning). Your data gives the **Tunisian voice**.
- A 7–9B open model + your data + RAG = an assistant that **beats GPT/Claude on Tunisian Derja
  and your CS domain**, runs cheap, and you fully own. It won't match frontier general
  intelligence — and it doesn't need to.

---

# Deeper comparison (research-backed) — for the refined goal

Refined requirements: (1) UNDERSTAND input in Arabizi OR Arabic-script Derja OR FR/EN
(multilingual), (2) strong reasoning "brain", (3) OUTPUT fluent Arabizi with no errors.

## Direct precedents (closest published work to this exact task)
- **NileChat (2505.18383)** — dialectal Arabic **+ Arabizi**, base = **Qwen-2.5-3B**. Chosen for
  strong MSA + best tokenizer efficiency on Arabic. They trained **dual-script** (Arabic +
  ~2M examples converted to Arabizi). → closest analog to our goal; validates **Qwen + Arabizi**.
- **GemMaroc (2505.17082)** — Moroccan **Darija**, base = **Gemma 3 (4B/27B)**. Learned the
  dialect from only ~50k reasoning-dense examples and **kept its reasoning** (GSM8K 84.2%,
  DarijaMMLU 61.6% with ~1/10 the data). **Arabic-script only — no Arabizi.**
- **Leaderboards:** Llama-3.3-70B tops OALL v2 / AraGen; Qwen2.5 is the strong baseline; Qwen3
  tops HELM-Arabic among open weights and dominates small-model fine-tuning benchmarks.

## Scored against the 3 requirements (7–9B class, for cheap serving)

| Model | (1) Understand Arabizi+Arabic+FR/EN | (2) Reasoning brain | (3) Arabizi output (after FT) | Tokenizer on Arabic | Precedent | Verdict |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B / Qwen3-8B** ⭐ | **very strong** (MSA+multilingual; Qwen used for Arabizi in NileChat) | **top** (Qwen3 #1 small-model FT) | proven (NileChat) | efficient | **NileChat = dialect+Arabizi** | **Best fit — pick this** |
| **Gemma 3 (4B/12B)** | strong | **top** (kept GSM8K 84%) | untested for Arabizi | efficient | GemMaroc = Darija (Arabic-script) | Strong #2; great if you also want Arabic-script |
| **Llama 3.1-8B (3.3-70B)** | good (70B best on Arabic leaderboards) | strong | workable | ok | leaderboard #1 at 70B | Great ecosystem; 8B Arabic weaker than Qwen |
| Jais / ALLaM | best Arabic-script, weak FR/Arabizi, weaker reasoning | lower | weak | n/a | Arabic-script only | Not for an Arabizi target |

## FINAL PICK
**Qwen3-8B-Instruct** (primary) — newest, best reasoning, Apache 2.0, and the only family with a
published **dialect-+-Arabizi** precedent (NileChat). **Qwen2.5-7B-Instruct** is the safe,
heavily-documented fallback (more tutorials, the exact NileChat base family).
**Gemma-3-12B-it** is the strong alternate, especially if you later add an Arabic-script mode.

## Method lessons to apply at fine-tune time (from both papers)
- Keep **~20% English/MSA** data in the mix → prevents catastrophic forgetting of the brain.
- **Reasoning-dense, quality** instruction data beats large generic data (GemMaroc: 1/10 the data).
- Train **dual-script understanding**: include Arabic-script Derja inputs too (we have the 3GB
  corpus + Arabic-script lexicon + parallel data) so it understands Arabic-letter input, while
  always **answering in Arabizi** (NileChat's dual-script recipe).

## Sources
HELM-Arabic / siliconflow Arabic guide; Open Arabic LLM Leaderboard v2 (HF blog); GemMaroc
(arXiv 2505.17082); NileChat (arXiv 2505.18383); distil labs small-model FT benchmark.

## Compute / cost to fine-tune (all open options)
QLoRA on 7–9B fits a single 24 GB GPU (RTX 3090/4090) or cloud (~$0.5–2/hr on RunPod/Colab/Lambda).
Effectively free to iterate. You keep the resulting adapter/weights.
