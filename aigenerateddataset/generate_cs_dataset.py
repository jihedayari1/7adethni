#!/usr/bin/env python3
"""
Claude-powered generator for a Tunisian conversational assistant (Arabizi).

The model must speak Tunisian Derja about ANY topic. Customer-service is the priority
slice, but it is NOT the whole dataset — so this generates a MIX:
  * GENERAL conversation  — chat, Q&A, advice, opinions, jokes, daily life, ... (broad fluency)
  * CUSTOMER_SERVICE      — a polite shop agent helping a client (the priority feature)

Tune the blend with --cs-ratio (default 0.45 = ~45% customer-service, ~55% general).

Fixes vs the hand-written generator:
  1. SCALE   — loops over many topics/intents and calls Claude to produce thousands.
  2. VOICE   — general chat = friendly Tunisian; CS = shop agent (not peer Facebook banter).
  3. SPELLING— pins one Arabizi convention (7 5 3 9 gh ch th dh); rejects MSA + Arabic script.

Design (per the claude-api skill):
  * model = claude-opus-4-8 (switch to claude-sonnet-4-6 for cheaper bulk — your call)
  * prompt caching: the big system+few-shot block is cached; the volatile per-call ask
    (category + topic) goes in the user turn, AFTER the cached prefix.
  * structured output: output_config.format json_schema -> guaranteed-parseable pairs.

Run:
  python aigenerateddataset/generate_cs_dataset.py --dry-run          # see prompts, no API
  python aigenerateddataset/generate_cs_dataset.py --estimate         # token + cost estimate
  python aigenerateddataset/generate_cs_dataset.py --target 300       # generate 300 pairs
  python aigenerateddataset/generate_cs_dataset.py --target 300 --cs-ratio 0.3
"""
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
from normalizer import normalize  # reused for dedup keys

OUT_DEFAULT = ROOT / "aigenerateddataset" / "cs_pairs.jsonl"

MODEL = "claude-opus-4-8"     # bulk-cost option: "claude-sonnet-4-6" (~3x cheaper). Your decision.
PER_CALL = 8                  # pairs requested per API call
MAX_TOKENS = 8000

# ---- CUSTOMER-SERVICE intents (priority slice; from dataset/SCRAPING_SPEC.md §5) ----
CS_INTENTS = {
    "greeting": "customer opens the chat / says hello",
    "availability": "is a product in stock",
    "price": "how much does X cost",
    "sizes_variants": "sizes, colors, or variants",
    "product_details": "specs / material / how it works",
    "delivery_time": "how long delivery takes",
    "delivery_cost": "delivery price",
    "delivery_zones": "do you deliver to <city>",
    "payment_methods": "cash on delivery, D17, flouci, card",
    "place_order": "customer wants to order",
    "order_status": "where is my order / tracking",
    "modify_cancel": "change or cancel an order",
    "returns_refunds": "return or refund / exchange",
    "warranty": "guarantee / after-sales",
    "complaints": "damaged item, late, wrong product",
    "store_hours_location": "opening hours / address",
    "promotions": "discounts / promo codes",
    "thanks_closing": "customer thanks / closes the chat",
}
DOMAINS = ["clothing", "phones & electronics", "cosmetics & parfum", "shoes & bags",
           "food & pastry delivery", "baby & mom", "home & deco", "sport & supplements"]

# ---- GENERAL conversation topics (broad Tunisian fluency, any subject) ----
GENERAL_TOPICS = {
    "daily_life": "everyday small talk, how are you, what did you do today",
    "food_cooking": "what to cook, recipes, restaurants, cravings",
    "football_sport": "matches, teams, training, results",
    "family_friends": "family, friends, relationships, social life",
    "work_study": "job, school, university, exams, career",
    "health_wellbeing": "feeling tired/sick, fitness, sleep, doctors",
    "emotions_support": "stress, sadness, motivation, encouragement",
    "technology": "phones, apps, internet, gadgets, social media",
    "travel_places": "cities, trips, beaches, directions, distances",
    "opinions_debate": "what do you think about X, light debates",
    "jokes_banter": "tell a joke, funny banter, teasing",
    "advice": "how do I start X, recommend me X, life tips",
    "general_knowledge": "simple facts, how-to, explanations",
    "money_economy": "prices, saving, salaries, cost of living (casual)",
    "culture_traditions": "Tunisian customs, holidays, music, habits",
    "weather": "heat, rain, seasons, what to wear",
    "hobbies_music": "music, movies, series, games, free time",
    "motivation_goals": "plans, goals, self-improvement, encouragement",
}

# ---- FIXED few-shot (stable bytes => cacheable) ----
GENERAL_FEWSHOT = """\
GENERAL conversation examples (user -> assistant) — friendly natural Tunisian:

[daily_life] USER: chna7welek el yom?
ASSISTANT: el 7amdoulah, chwaya t3ab mel 5edma ama labes 😅 enti chnowa a7welek?

[food_cooking] USER: chnowa ntayeb el lila? ma3andich fekra
ASSISTANT: a3mel 3jja b batata wala makarouna b djej, fissa3 w bnina 😋

[football_sport] USER: chkoun bch yarbah el match el lila fi rayek?
ASSISTANT: el espérance 9wiya hethi el saison, ama fel classico kollou momken 😎

[advice] USER: n7eb nabda nat3allem anglais, mnin nabda?
ASSISTANT: abda b application kima Duolingo w etfarrej series b sous-titres, chwaya chwaya bch tetla3 💪

[emotions_support] USER: 7asis rou7i down el yumin hethom
ASSISTANT: rou7i 3leha 🤍 kol wa7ed yfout b nhar s3ib, 5oudh nafsek w o5rouj m3a s7abek chwaya.

[general_knowledge] USER: 9adech el masafa bin tounes w sfax?
ASSISTANT: taghreeban 270 km, sa3a w noss b karhba 3la l autoroute.

[jokes_banter] USER: 9olli nokta t-dha7ek
ASSISTANT: 3lech el ordinateur ma 9adch yo5dem? 5atrou 3andou virus 😂
"""

CS_FEWSHOT = """\
CUSTOMER-SERVICE examples (customer -> shop agent) — polite, helpful, concise:

[price] CUSTOMER: 3aslema, 9adech taman el robe el a7mar el li fel post?
AGENT: marhba bik 🌸 el robe b 89 dt, w famma promo -10% el weekend hetha barka.

[availability] CUSTOMER: el sac el aswed mazel mawjoud wala 5las?
AGENT: ey mazel 3andna 🙌 t7ebbi nra7dhoulek wa7ed? 9olili el couleur w el model.

[delivery_zones] CUSTOMER: famma livraison l Gabes? w b 9adech?
AGENT: ey n3aytiw l Gabes lkol, el livraison b 7 dt w tousel fi 48 sa3a inchallah.

[order_status] CUSTOMER: win wsel el colis mte3i? 3malt commande el sbe7.
AGENT: 3tini el numéro mte3 el commande w nchoufoulek el position tawa 🙏

[returns_refunds] CUSTOMER: ken el taille ma jatnich n9der nbaddel?
AGENT: ey 3andek 7a9 el tabdil fi 7 ayyem, ki ma yji el 9is njiboulek wa7ed a5er bla mochkla.

[complaints] CUSTOMER: jeni el produit ma5sour w 9lej 😡
AGENT: sama7ni barcha 3al haka 🙏 nraddoulek wa7ed jdid wala nraddou flousek 3la 7asb ma t7ebb.
"""

SYSTEM = f"""You generate TRAINING DATA for a Tunisian conversational assistant.

The assistant MUST speak natural Tunisian Derja (Arabizi) about ANY topic — casual chat, \
questions, advice, opinions, jokes, daily life — AND also handle online-store customer service. \
You will be told the CATEGORY (general or customer_service) and a topic for each batch.

VOICE:
- general        -> a friendly, natural Tunisian person/assistant chatting like a real friend.
- customer_service -> a polite, helpful shop agent helping a client (NOT a random commenter).

HARD RULES (follow all):
- Output ONLY natural Tunisian Derja in Arabizi (Latin letters + numbers). \
NEVER Modern Standard Arabic / fos7a. If it sounds like a TV news anchor, it is WRONG.
- NEVER use Arabic script. Latin + numbers only.
- Spelling convention (use consistently): 7=ح, 5=خ, 3=ع, 9=ق, gh=غ, ch=ش, th=ث, dh=ذ. \
French-style vowels (ou, not oo). Do NOT use 8 for غ; use gh.
- Natural French/English code-switching is realistic and welcome (prix, livraison, weekend, stock).
- Short, realistic turns (1-3 sentences). Vary phrasing, mood, and subject. Light emojis ok (0-2).
- Produce NEW content — do NOT copy the examples.

{GENERAL_FEWSHOT}
{CS_FEWSHOT}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "user": {"type": "string"},
                    "assistant": {"type": "string"},
                },
                "required": ["topic", "user", "assistant"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}

# ---- quality filters ----
_MSA = re.compile(r"\b(hadha|hadhihi|alladhi|sawfa|laysa|kayfa|ladayna|yumkinu|na7nu|"
                  r"jiddan|kathiran|inna|dhalika)\b", re.I)
_ARABIC = re.compile(r"[؀-ۿ]")
_REPEAT = re.compile(r"(.)\1{3,}")

def enforce_convention(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("$", "ch")
    t = _REPEAT.sub(r"\1\1", t)          # barchaaaaa -> barchaa
    return re.sub(r"\s+", " ", t)

def is_clean(user: str, assistant: str) -> bool:
    # Dual-script (NileChat recipe): the INPUT (user) may be Arabizi, Arabic-script Derja,
    # French, or English — we want the model to understand ALL of them. The OUTPUT (assistant)
    # must ALWAYS be fluent Arabizi (Latin), never Arabic script and never MSA.
    if not (2 <= len(user) <= 300 and 2 <= len(assistant) <= 400):
        return False
    if _ARABIC.search(assistant):                  # OUTPUT must be Arabizi, not Arabic script
        return False
    if _MSA.search(assistant):                     # MSA leakage only matters on the OUTPUT
        return False
    if len(assistant.split()) < 2:
        return False
    return True


def user_message(category: str, topic: str, topic_desc: str, k: int, domain: str = "") -> str:
    if category == "customer_service":
        ctx = f"Store domain: {domain}. Intent: {topic} ({topic_desc})."
        who = "Each pair = one CUSTOMER message + one SHOP AGENT reply."
    else:
        ctx = f"Topic: {topic} ({topic_desc})."
        who = "Each pair = one USER message + one friendly ASSISTANT reply (casual Tunisian)."
    return (f"Category: {category}.\n{ctx}\n{who}\n"
            f"Generate {k} NEW, diverse pairs. Vary phrasing, subject, and tone. "
            f'Tag each with topic="{topic}".')


def build_request_kwargs(category, topic, topic_desc, k, domain=""):
    return dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user",
                   "content": user_message(category, topic, topic_desc, k, domain)}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"},
    )


def next_task(i: int, cs_ratio: float):
    """Interleave customer_service and general tasks by ratio (deterministic round-robin)."""
    cs_topics = list(CS_INTENTS.items())
    gen_topics = list(GENERAL_TOPICS.items())
    # every 1/cs_ratio-th item is customer_service (deterministic, well-spread)
    is_cs = (cs_ratio > 0) and (int((i + 1) * cs_ratio) > int(i * cs_ratio))
    if is_cs:
        topic, desc = cs_topics[i % len(cs_topics)]
        domain = DOMAINS[(i // len(cs_topics)) % len(DOMAINS)]
        return "customer_service", topic, desc, domain
    topic, desc = gen_topics[i % len(gen_topics)]
    return "general", topic, desc, ""


def generate(target: int, per_call: int, cs_ratio: float, out_path: Path):
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

    seen, written, cache_reads = set(), 0, 0
    counts = {"general": 0, "customer_service": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a", encoding="utf-8")
    i = 0
    while written < target:
        category, topic, desc, domain = next_task(i, cs_ratio)
        i += 1
        try:
            resp = client.messages.create(
                **build_request_kwargs(category, topic, desc, per_call, domain))
        except anthropic.APIError as e:
            print(f"  ! API error ({topic}): {str(e)[:90]}"); time.sleep(2); continue
        cache_reads += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            print(f"  ! refusal on {topic}; skipping"); continue
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            pairs = json.loads(text).get("pairs", [])
        except json.JSONDecodeError:
            print("  ! unparseable batch; skipping"); continue
        kept = 0
        for p in pairs:
            u = enforce_convention(p.get("user", ""))
            a = enforce_convention(p.get("assistant", ""))
            if not is_clean(u, a):
                continue
            key = (normalize(u)[:50], normalize(a)[:50])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({
                "instruction": u, "output": a,
                "category": category, "topic": p.get("topic", topic),
                "domain": domain or None,
                "synthetic": True, "needs_native_review": True,
                "model": MODEL, "source": "claude-gen",
            }, ensure_ascii=False) + "\n")
            f.flush()
            written += 1; kept += 1; counts[category] += 1
        print(f"  [{written:>4}/{target}] {category:<16} {topic:<20} +{kept}  "
              f"(cache_read tok: {cache_reads})")
    f.close()
    print(f"\nDONE: {written} clean pairs -> {out_path.relative_to(ROOT)}")
    print(f"mix: {counts['customer_service']} customer_service / {counts['general']} general")
    print("All flagged synthetic + needs_native_review. Review a sample before training.")


def estimate(per_call: int, target: int, cs_ratio: float):
    import anthropic
    client = anthropic.Anthropic()
    ct = client.messages.count_tokens(
        model=MODEL,
        system=[{"type": "text", "text": SYSTEM}],
        messages=[{"role": "user", "content": user_message("general", "daily_life",
                                                           GENERAL_TOPICS["daily_life"], per_call)}],
    )
    calls = max(1, target // max(1, per_call))
    sys_tok = ct.input_tokens
    in_cost = (sys_tok * 5e-6) + (calls - 1) * (sys_tok * 0.1 * 5e-6)
    out_cost = calls * MAX_TOKENS * 0.5 * 25e-6
    print(f"system prompt: ~{sys_tok} input tokens (cached after call 1)")
    print(f"~{calls} calls for {target} pairs at {per_call}/call on {MODEL} (cs_ratio={cs_ratio})")
    print(f"rough cost: input ~${in_cost:.2f} + output ~${out_cost:.2f} = ~${in_cost+out_cost:.2f} "
          f"(caching saves ~${(calls-1)*sys_tok*0.9*5e-6:.2f})")
    print("Switch MODEL to claude-sonnet-4-6 to cut this ~ in half.")


WEB_ASK = """\

============================ YOUR TASK ============================
Output a JSON array of 30 NEW pairs.
- ~45% with "category": "customer_service" (vary store domains: clothing, phones,
  cosmetics, food/pastry, shoes, baby, home, sport).
- ~55% with "category": "general" (vary topics widely: daily_life, food_cooking,
  football_sport, family_friends, work_study, health_wellbeing, emotions_support,
  technology, travel_places, opinions_debate, jokes_banter, advice, general_knowledge,
  money_economy, culture_traditions, weather, hobbies_music, motivation_goals).
Each element exactly: {"category": "...", "topic": "...", "user": "...", "assistant": "..."}
Obey ALL the rules above (Arabizi only, the 7/5/3/9/gh/ch/th/dh convention, NO fos7a).
Output ONLY the raw JSON array — no prose, no markdown fences.
=================================================================="""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--web-prompt", action="store_true",
                    help="print the FREE copy-paste prompt for the Claude website (no API key needed)")
    ap.add_argument("--per-call", type=int, default=PER_CALL)
    ap.add_argument("--cs-ratio", type=float, default=0.45,
                    help="share of customer-service pairs (rest is general). default 0.45")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()

    if args.web_prompt:
        print(SYSTEM + WEB_ASK)
        return
    if args.dry_run:
        print("=== SYSTEM (cached, both categories) ===\n" + SYSTEM[:1200] + "\n...[truncated]...\n")
        print("=== USER (general) ===\n" +
              user_message("general", "jokes_banter", GENERAL_TOPICS["jokes_banter"], args.per_call))
        print("\n=== USER (customer_service) ===\n" +
              user_message("customer_service", "price", CS_INTENTS["price"], args.per_call, "clothing"))
        print(f"\n(general topics={len(GENERAL_TOPICS)}, cs intents={len(CS_INTENTS)}, "
              f"domains={len(DOMAINS)}; default mix ~{int(args.cs_ratio*100)}% CS)")
        # show interleave for first 12 tasks
        seq = [next_task(j, args.cs_ratio)[0][:3] for j in range(12)]
        print("interleave (cs/gen) first 12:", " ".join(seq))
        return
    if args.estimate:
        estimate(args.per_call, args.target, args.cs_ratio); return
    generate(args.target, args.per_call, args.cs_ratio, Path(args.out))


if __name__ == "__main__":
    main()
