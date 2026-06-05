#!/usr/bin/env python3
"""
Claude-powered generator for Tunisian customer-service conversation pairs (Arabizi).

Fixes the three problems of the hand-written generator:
  1. SCALE   — loops over intents x product-domains and calls Claude to produce thousands.
  2. VOICE   — generates B2C *shop-agent* replies (not peer Facebook banter).
  3. SPELLING— pins one Arabizi convention (7 5 3 9 gh ch th dh) via fixed few-shot + a
               light post-pass; auto-rejects MSA leakage and Arabic script.

Design (per the claude-api skill):
  * model            = claude-opus-4-8 (switch to claude-sonnet-4-6 for cheaper bulk — your call)
  * prompt caching   = the big system+few-shot block is cached (cache_control), so every call
                       after the first reuses it at ~0.1x cost. The volatile per-call ask
                       (intent + domain) goes in the user turn, AFTER the cached prefix.
  * structured output= output_config.format json_schema -> guaranteed-parseable pairs.

Run:
  setx ANTHROPIC_API_KEY "sk-ant-..."      (Windows; new shell after)
  python aigenerateddataset/generate_cs_dataset.py --dry-run          # see the prompt, no API
  python aigenerateddataset/generate_cs_dataset.py --estimate         # token+cost estimate
  python aigenerateddataset/generate_cs_dataset.py --target 300       # generate 300 pairs
"""
import argparse, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
from normalizer import normalize  # reused for dedup keys

OUT_DEFAULT = ROOT / "aigenerateddataset" / "cs_pairs.jsonl"

MODEL = "claude-opus-4-8"     # bulk-cost option: "claude-sonnet-4-6" (3x cheaper). Your decision.
PER_CALL = 8                  # pairs requested per API call
MAX_TOKENS = 8000

# ---- customer-service intents (from dataset/SCRAPING_SPEC.md §5) ----
INTENTS = {
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

# ---- FIXED few-shot (stable bytes => cacheable). On-convention B2C shop-agent voice. ----
FEWSHOT = """\
Examples (customer -> shop agent), the EXACT style/voice/spelling to copy:

[price] CUSTOMER: 3aslema, 9adech taman el robe el a7mar el li fel post?
AGENT: marhba bik 🌸 el robe b 89 dt, w famma promo -10% el weekend hetha barka.

[availability] CUSTOMER: el sac el aswed mazel mawjoud wala 5las?
AGENT: ey mazel 3andna 🙌 t7ebbi nra7dhoulek wa7ed? 9olili el couleur w el model.

[delivery_zones] CUSTOMER: famma livraison l Gabes? w b 9adech?
AGENT: ey n3aytiw l Gabes lkol, el livraison b 7 dt w tousel fi 48 sa3a inchallah.

[payment_methods] CUSTOMER: n9der na5les cash a la livraison?
AGENT: akid, na9blou cash 3and el livraison wala b D17 / flouci ki t7ebb.

[order_status] CUSTOMER: win wsel el colis mte3i? 3malt commande el sbe7.
AGENT: 3tini el numéro mte3 el commande w nchoufoulek el position tawa 🙏

[returns_refunds] CUSTOMER: ken el taille ma jatnich n9der nbaddel?
AGENT: ey 3andek 7a9 el tabdil fi 7 ayyem, ki ma yji el 9is njiboulek wa7ed a5er bla mochkla.

[sizes_variants] CUSTOMER: el pointure 42 mawjouda fel modele hetha?
AGENT: 5alli nchouf el stock... ey mawjouda 👌 n7adharhalek? 9olili el couleur.

[complaints] CUSTOMER: jeni el produit ma5sour w 9lej 😡
AGENT: sama7ni barcha 3al haka 🙏 nraddoulek wa7ed jdid wala nraddou flousek 3la 7asb ma t7ebb.

[greeting] CUSTOMER: 3aslema
AGENT: 3aslema w marhba bik fi <store> 🌟 chnowa n9der n3awnek?

[thanks_closing] CUSTOMER: ya3tik essa7a 3al 5edma
AGENT: el 3afw 🌸 famma 7aja o5ra? marhba bik dima, w nchoufouk 9rib inchallah.
"""

SYSTEM = f"""You generate TRAINING DATA for a Tunisian online-store customer-service assistant.

Every pair is: a CUSTOMER message + the SHOP AGENT's reply, both in **Tunisian Derja written \
in Arabizi** (Latin letters + numbers). This is B2C support — the AGENT is a polite, helpful \
Tunisian shop assistant, NOT a random Facebook commenter.

HARD RULES (follow all):
- Output ONLY natural Tunisian Derja in Arabizi. NEVER Modern Standard Arabic / fos7a. \
If it sounds like a TV news anchor, it is WRONG.
- NEVER use Arabic script. Latin + numbers only.
- Spelling convention (use consistently): 7=ح, 5=خ, 3=ع, 9=ق, gh=غ, ch=ش, th=ث, dh=ذ. \
French-style vowels (ou, not oo). Do NOT use 8 for غ; use gh.
- Natural French/English code-switching is fine and realistic (livraison, prix, commande, stock, promo).
- The AGENT voice: warm, concise, helpful; offers the next step; light emojis ok (0-2).
- Keep each message short and realistic (1-3 sentences). Vary products, phrasing, customer mood.
- Do NOT copy the examples verbatim — produce NEW content.

{FEWSHOT}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "customer": {"type": "string"},
                    "agent": {"type": "string"},
                },
                "required": ["intent", "customer", "agent"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}

# ---- quality filters ----
_MSA = re.compile(r"\b(hadha|hadhihi|alladhi|sawfa|laysa|kayfa|ladayna|yumkinu|na7nu|"
                  r"jiddan|kathiran|sa-|inna|dhalika)\b", re.I)
_ARABIC = re.compile(r"[؀-ۿ]")
_REPEAT = re.compile(r"(.)\1{3,}")

def enforce_convention(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("$", "ch")
    t = _REPEAT.sub(r"\1\1", t)          # barchaaaaa -> barchaa
    return re.sub(r"\s+", " ", t)

def is_clean(customer: str, agent: str) -> bool:
    if not (2 <= len(customer) <= 300 and 2 <= len(agent) <= 400):
        return False
    if _ARABIC.search(customer + agent):           # must be Arabizi, not Arabic script
        return False
    if _MSA.search(customer + agent):              # MSA leakage
        return False
    if len(agent.split()) < 2:
        return False
    return True


def user_message(intent: str, domain: str, k: int) -> str:
    return (f"Generate {k} NEW, diverse customer-service pairs.\n"
            f"Intent: {intent} ({INTENTS[intent]}).\n"
            f"Store domain: {domain}.\n"
            f"Vary the products, customer phrasing, and tone. Tag each with intent=\"{intent}\".")


def build_request_kwargs(intent: str, domain: str, k: int):
    return dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message(intent, domain, k)}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"},
    )


def generate(target: int, per_call: int, out_path: Path):
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

    seen, written = set(), 0
    cache_reads = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a", encoding="utf-8")
    intents = list(INTENTS)
    i = 0
    while written < target:
        intent = intents[i % len(intents)]
        domain = DOMAINS[(i // len(intents)) % len(DOMAINS)]
        i += 1
        try:
            resp = client.messages.create(**build_request_kwargs(intent, domain, per_call))
        except anthropic.APIError as e:
            print(f"  ! API error ({intent}/{domain}): {str(e)[:90]}")
            time.sleep(2)
            continue
        cache_reads += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            print(f"  ! refusal on {intent}/{domain}; skipping")
            continue
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            pairs = json.loads(text).get("pairs", [])
        except json.JSONDecodeError:
            print("  ! unparseable batch; skipping")
            continue
        kept = 0
        for p in pairs:
            cust = enforce_convention(p.get("customer", ""))
            agent = enforce_convention(p.get("agent", ""))
            if not is_clean(cust, agent):
                continue
            key = (normalize(cust)[:50], normalize(agent)[:50])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({
                "instruction": cust, "output": agent,
                "intent": p.get("intent", intent), "domain": domain,
                "synthetic": True, "needs_native_review": True,
                "model": MODEL, "source": "claude-gen",
            }, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            kept += 1
        print(f"  [{written:>4}/{target}] {intent:<16} {domain:<22} +{kept}  "
              f"(cache_read tok so far: {cache_reads})")
    f.close()
    print(f"\nDONE: {written} clean pairs -> {out_path.relative_to(ROOT)}")
    print("All flagged synthetic + needs_native_review. Review a sample before training.")


def estimate(per_call: int, target: int):
    import anthropic
    client = anthropic.Anthropic()
    ct = client.messages.count_tokens(
        model=MODEL,
        system=[{"type": "text", "text": SYSTEM}],
        messages=[{"role": "user", "content": user_message("price", DOMAINS[0], per_call)}],
    )
    calls = max(1, target // max(1, per_call))
    sys_tok = ct.input_tokens
    # 1st call full price; rest read the cached system prefix at ~0.1x
    in_cost = (sys_tok * 5e-6) + (calls - 1) * (sys_tok * 0.1 * 5e-6)
    out_cost = calls * MAX_TOKENS * 0.5 * 25e-6  # assume ~half max_tokens out
    print(f"system prompt: ~{sys_tok} input tokens (cached after call 1)")
    print(f"~{calls} calls for {target} pairs at {per_call}/call on {MODEL}")
    print(f"rough cost estimate: input ~${in_cost:.2f} + output ~${out_cost:.2f} "
          f"= ~${in_cost+out_cost:.2f}  (caching saves ~{(calls-1)*sys_tok*0.9*5e-6:.2f}$)")
    print("Switch MODEL to claude-sonnet-4-6 to cut this ~ in half.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--per-call", type=int, default=PER_CALL)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true", help="print one assembled prompt, no API call")
    ap.add_argument("--estimate", action="store_true", help="token + cost estimate, minimal API use")
    args = ap.parse_args()

    if args.dry_run:
        print("=== SYSTEM (cached) ===\n" + SYSTEM[:1500] + "\n...[truncated]...\n")
        print("=== USER (per call) ===\n" + user_message("price", DOMAINS[0], args.per_call))
        print(f"\n(system is ~{len(SYSTEM)} chars; intents={len(INTENTS)}, domains={len(DOMAINS)})")
        return
    if args.estimate:
        estimate(args.per_call, args.target); return
    generate(args.target, args.per_call, Path(args.out))


if __name__ == "__main__":
    main()
