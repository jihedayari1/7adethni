"""
Stage 2 - SFT (instruction tuning) on top of the CPT adapter.  Azure-grade, result-trusted.

Rigor built in:
  * held-out VALIDATION split -> eval_loss (catches the overfitting that ruined run #1)
  * best-checkpoint selection on val loss + early stopping
  * prompt masking (loss only on the answer, never on the question)
  * stratified val split across tasks, so val is representative
  * English retention mixed in (anti-forgetting), fraction is explicit and logged
  * manifest.json with data hashes + metrics

Usage
  python training/azure/train_sft.py --cpt outputs/tunisian_cpt --out outputs/tunisian_final
  python training/azure/train_sft.py --dry-run
  # ablation: train SFT WITHOUT the CPT adapter, to prove CPT's contribution
  python training/azure/train_sft.py --cpt none --out outputs/sft_only
"""
import argparse, collections, json, math, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (set_seed_all, sha256, autoconfig, pick_attention,
                    training_args, write_manifest)   # noqa: E402

SYSTEM = ("Enti '7adethni', musa3ed tounsi. Tefhem el arabizi, el 3arbi, el francais w el anglais, "
          "w tjaweb DIMA bel derja tounsiya bel arabizi (7ourouf w arqam) b tari9a tabi3iya. "
          "Ken el user yotlob 7aja b lugha o5ra, 3awenou ama 5alli el asas tounsi.")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/sft_real.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cpt", default="outputs/tunisian_cpt",
                    help="Stage-1 adapter to continue, or 'none' for the ablation")
    ap.add_argument("--out", default="outputs/tunisian_final")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32, help="only used when --cpt none")
    ap.add_argument("--retention", type=int, default=3000,
                    help="English instructions mixed in to prevent forgetting")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--val-max", type=int, default=600)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=3, help="early-stopping evals without improvement")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--quant", choices=["auto", "none", "4bit"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_rows(a):
    rows = []
    for l in open(a.data, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        rows.append({"instruction": r["instruction"], "output": r["output"],
                     "task": r.get("task", "tunisian")})
    n_tun = len(rows)
    if a.retention > 0:
        try:
            from datasets import load_dataset
            al = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=a.seed)
            al = al.select(range(min(a.retention, len(al))))
            for r in al:
                instr = r["instruction"] + (("\n" + r["input"]) if r["input"] else "")
                rows.append({"instruction": instr, "output": r["output"], "task": "retention_en"})
            print("retention added: {}".format(len(rows) - n_tun))
        except Exception as e:
            print("retention SKIPPED (no internet?):", e)
    return rows, n_tun


def stratified_split(rows, val_frac, val_max, seed):
    """Val set mirrors the task mix, so eval_loss is representative (not all one task)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    rng = random.Random(seed)
    val = []
    for task, items in by.items():
        rng.shuffle(items)
        k = max(1, int(len(items) * val_frac))
        val += items[:k]
    rng.shuffle(val)
    val = val[:val_max]
    val_ids = {id(r) for r in val}
    train = [r for r in rows if id(r) not in val_ids]
    rng.shuffle(train)
    return train, val


def main():
    a = parse()
    set_seed_all(a.seed)
    t0 = time.time()

    hw = autoconfig(a.seq)
    quant = a.quant if a.quant != "auto" else hw["quant"]
    batch = a.batch or hw["batch"]
    accum = a.grad_accum or max(1, hw["grad_accum"] * hw["batch"] // batch)
    use_bf16 = hw["bf16"] and quant == "none"
    use_cpt = a.cpt and a.cpt.lower() != "none"

    print("=" * 72)
    print("GPU      : {} ({} GB) -> {}".format(hw["name"], hw["vram_gb"], hw["tier"]))
    print("mode     : {}".format("CPT->SFT (continue adapter)" if use_cpt
                                 else "SFT ONLY (ablation, fresh LoRA)"))
    print("batch    : {} x accum {} x seq {}".format(batch, accum, a.seq))
    print("=" * 72)

    rows, n_tun = load_rows(a)
    train_rows, val_rows = stratified_split(rows, a.val_frac, a.val_max, a.seed)
    mix = collections.Counter(r["task"] for r in train_rows)
    print("SFT rows: train {:,} | val {:,}".format(len(train_rows), len(val_rows)))
    print("task mix:", dict(mix))

    if a.dry_run:
        write_manifest(a.out, {"stage": "sft", "dry_run": True, "config": vars(a), "hardware": hw,
                               "data": {"sha": sha256(a.data), "tunisian_rows": n_tun,
                                        "train": len(train_rows), "val": len(val_rows),
                                        "task_mix": dict(mix)}})
        return

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kw = {"device_map": {"": 0}, "attn_implementation": pick_attention()}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if hw["bf16"] else torch.float16,
            bnb_4bit_use_double_quant=True)
    kw["torch_dtype"] = torch.bfloat16 if hw["bf16"] else torch.float16
    base = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    base.config.use_cache = False
    if quant == "4bit":
        base = prepare_model_for_kbit_training(base)

    if use_cpt:
        model = PeftModel.from_pretrained(base, a.cpt, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def fmt(r):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["instruction"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + r["output"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False, truncation=True, max_length=a.seq)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]     # supervise ONLY the answer
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids),
                "labels": labels[:len(f_ids)]}

    cols = ["instruction", "output", "task"]
    train_ds = Dataset.from_list(train_rows).map(fmt, remove_columns=cols)
    val_ds = Dataset.from_list(val_rows).map(fmt, remove_columns=cols)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    args = training_args(
        output_dir=a.out, per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum, num_train_epochs=a.epochs, learning_rate=a.lr,
        warmup_ratio=0.03, lr_scheduler_type="cosine", weight_decay=0.01,
        logging_steps=25, eval_strategy="steps", eval_steps=a.eval_steps,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=use_bf16, fp16=not use_bf16,
        optim="paged_adamw_8bit" if quant == "4bit" else "adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=a.report_to, seed=a.seed, label_names=["labels"], remove_unused_columns=False)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
    res = trainer.train(resume_from_checkpoint=a.resume)
    metrics = trainer.evaluate()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    write_manifest(a.out, {
        "stage": "sft", "config": vars(a), "hardware": hw,
        "from_cpt": a.cpt if use_cpt else None,
        "effective": {"batch": batch, "grad_accum": accum, "quant": quant,
                      "precision": "bf16" if use_bf16 else "fp16"},
        "data": {"file": a.data, "sha": sha256(a.data), "tunisian_rows": n_tun,
                 "train": len(train_rows), "val": len(val_rows), "task_mix": dict(mix)},
        "metrics": {"train_loss": res.training_loss, "eval_loss": metrics["eval_loss"],
                    "val_perplexity": round(math.exp(min(20, metrics["eval_loss"])), 2)},
        "runtime_hours": round((time.time() - t0) / 3600, 2)})
    print("\nDONE. val_loss={:.4f} -> {}".format(metrics["eval_loss"], a.out))


if __name__ == "__main__":
    main()
