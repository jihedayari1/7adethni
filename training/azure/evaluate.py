"""
TunBench evaluation with ABLATION + CONFIDENCE INTERVALS.  This is what makes a result trustworthy.

It answers the three questions the old setup could not:
  1. "Did CPT actually help?"        -> ablation: base vs cpt vs cpt+sft vs sft-only
  2. "Is the gap real or noise?"     -> paired bootstrap CI + win-rate vs base
  3. "Would I get this again?"       -> greedy decoding (deterministic), fixed seed, saved preds

Metrics: real_word (lexicon), msa_leak, number_rule, comprehension (held-out slang).

Usage
  python training/azure/evaluate.py \
      --adapters base=none cpt=outputs/tunisian_cpt final=outputs/tunisian_final \
      --out outputs/eval
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset" / "tools"))
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(Path(__file__).parent))
from coherence_score import load_vocab, score_text          # noqa: E402
from tunbench import number_rule_ok, contains_expected      # noqa: E402
try:
    from msa_leakage import score_text as _msa_raw
    def msa_label(t):
        r = _msa_raw(t)
        return r.get("label") if isinstance(r, dict) else r
except Exception:
    msa_label = None

SYSTEM = ("Enti '7adethni', musa3ed tounsi. Tefhem el arabizi, el 3arbi, el francais w el anglais, "
          "w tjaweb DIMA bel derja tounsiya bel arabizi (7ourouf w arqam) b tari9a tabi3iya. "
          "Ken el user yotlob 7aja b lugha o5ra, 3awenou ama 5alli el asas tounsi.")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapters", nargs="+", default=["base=none"],
                    help="name=path pairs; use 'none' for the untuned base")
    ap.add_argument("--eval", default=str(ROOT / "dataset" / "eval_set.jsonl"))
    ap.add_argument("--comprehension", default=str(ROOT / "dataset" / "tunbench_comprehension.jsonl"))
    ap.add_argument("--comp-n", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--comp-max-new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--quant", choices=["auto", "none", "4bit"], default="auto")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/eval")
    return ap.parse_args()


# ---------------- metrics ----------------
def per_item_scores(preds, vocab, skels, expected=None):
    """Return per-item metric lists (needed for bootstrap)."""
    rw, msa, num, comp = [], [], [], []
    for i, p in enumerate(preds):
        rw.append(score_text(p, vocab, skels)[0])
        if msa_label:
            msa.append(1.0 if msa_label(p) in ("msa_leak", "arabic_script") else 0.0)
        else:
            msa.append(0.0)
        num.append(1.0 if number_rule_ok(p)[0] else 0.0)
        if expected is not None:
            comp.append(1.0 if contains_expected(p, expected[i]) else 0.0)
    return {"real_word": rw, "msa_leak": msa, "number_rule": num,
            **({"comprehension": comp} if expected is not None else {})}


def ci(values, n_boot, seed):
    """95% bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (statistics.mean(values), means[int(.025 * n_boot)], means[int(.975 * n_boot)])


def paired_delta(a_vals, b_vals, n_boot, seed):
    """P(model A > model B) via paired bootstrap -> 'is the gap real?'"""
    if not a_vals or len(a_vals) != len(b_vals):
        return None
    rng = random.Random(seed)
    n = len(a_vals)
    wins = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_vals[i] for i in idx) / n
        db = sum(b_vals[i] for i in idx) / n
        if da > db:
            wins += 1
    return wins / n_boot


def main():
    a = parse()
    random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    variants = []
    for spec in a.adapters:
        name, _, path = spec.partition("=")
        variants.append((name, None if path.lower() in ("none", "") else path))

    ev = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    comp = []
    if Path(a.comprehension).exists():
        comp = [json.loads(l) for l in open(a.comprehension, encoding="utf-8") if l.strip()][:a.comp_n]
    print("eval prompts: {} | comprehension: {}".format(len(ev), len(comp)))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    sys.path.insert(0, str(Path(__file__).parent))
    from common import autoconfig, pick_attention
    hw = autoconfig(1024)
    quant = a.quant if a.quant != "auto" else hw["quant"]

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                     # correct for batched generation

    kw = {"device_map": {"": 0}, "attn_implementation": pick_attention()}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if hw["bf16"] else torch.float16,
            bnb_4bit_use_double_quant=True)
    kw["torch_dtype"] = torch.bfloat16 if hw["bf16"] else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.eval()

    # attach every adapter once; switch between them by name
    peft_model = None
    for name, path in variants:
        if path is None:
            continue
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)
    active = peft_model or model

    def generate(prompts, max_new):
        outs = []
        for i in range(0, len(prompts), a.batch):
            chunk = prompts[i:i + a.batch]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = active.generate(**enc, max_new_tokens=max_new,
                                    do_sample=False,             # GREEDY = reproducible
                                    pad_token_id=tok.eos_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip())
            print("  {}/{}".format(min(i + a.batch, len(prompts)), len(prompts)), end="\r")
        return outs

    vocab, skels, _ = load_vocab()
    results, raw_scores = {}, {}
    for name, path in variants:
        print("\n--- evaluating: {} ({}) ---".format(name, path or "untuned base"))
        t0 = time.time()
        if path is None:
            ctx = active.disable_adapter() if peft_model else _NullCtx()
            with ctx:
                preds = generate([r["instruction"] for r in ev], a.max_new)
                cpreds = generate([r["instruction"] for r in comp], a.comp_max_new) if comp else []
        else:
            active.set_adapter(name)
            preds = generate([r["instruction"] for r in ev], a.max_new)
            cpreds = generate([r["instruction"] for r in comp], a.comp_max_new) if comp else []

        s = per_item_scores(preds, vocab, skels)
        if comp:
            cs = per_item_scores(cpreds, vocab, skels, expected=[c["expected"] for c in comp])
            s["comprehension"] = cs["comprehension"]
        raw_scores[name] = s
        results[name] = {k: ci(v, a.bootstrap, a.seed) for k, v in s.items()}
        with open(out / "preds_{}.jsonl".format(name), "w", encoding="utf-8") as f:
            for r, p in zip(ev, preds):
                f.write(json.dumps({"instruction": r["instruction"], "output": p},
                                   ensure_ascii=False) + "\n")
        print("  done in {:.1f} min".format((time.time() - t0) / 60))

    # ---------------- report ----------------
    metrics = ["real_word", "number_rule", "comprehension", "msa_leak"]
    print("\n" + "=" * 78)
    print("TUNBENCH  (greedy decoding, 95% bootstrap CI)")
    print("=" * 78)
    header = "{:<10}".format("model") + "".join("{:>21}".format(m) for m in metrics if m in results[variants[0][0]])
    print(header)
    for name, _ in variants:
        row = "{:<10}".format(name)
        for m in metrics:
            if m in results[name]:
                mean, lo, hi = results[name][m]
                row += "{:>21}".format("{:.0%} [{:.0%}-{:.0%}]".format(mean, lo, hi))
        print(row)

    base_name = variants[0][0]
    if len(variants) > 1:
        print("\nIs the gap real?  P(model > {}) by paired bootstrap".format(base_name))
        for name, _ in variants[1:]:
            parts = []
            for m in metrics:
                if m in raw_scores[name] and m in raw_scores[base_name]:
                    p = paired_delta(raw_scores[name][m], raw_scores[base_name][m], a.bootstrap, a.seed)
                    if m == "msa_leak" and p is not None:
                        p = 1 - p                     # lower is better here
                    parts.append("{} {:.0%}".format(m, p))
            print("  {:<10} {}".format(name, "  |  ".join(parts)))
        print("\n  >90% = clearly better. 40-60% = indistinguishable (do NOT ship on this).")

    report = {"model": a.model, "variants": {n: p for n, p in variants},
              "n_eval": len(ev), "n_comprehension": len(comp),
              "decoding": "greedy", "bootstrap": a.bootstrap,
              "results": {n: {m: {"mean": v[0], "lo": v[1], "hi": v[2]}
                              for m, v in r.items()} for n, r in results.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nsaved -> {}/report.json  + preds_*.jsonl".format(a.out))


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *x): return False


if __name__ == "__main__":
    main()
