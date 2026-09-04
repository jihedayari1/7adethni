"""
Stage 1 - CPT (continued pre-training) on Tunisian text.  Azure-grade, result-trusted.

Upgrades over the Kaggle notebook:
  * held-out VALIDATION split -> eval_loss during training (you can SEE overfitting)
  * best-checkpoint selection on val loss (not "whatever the last step produced")
  * hardware auto-config (bf16 + no quantization on A100/H100; 4-bit only if VRAM is small)
  * flash-attention when available, packing, resumable, seeded
  * manifest.json: config + data hash + metrics, so a result traces back to its inputs

Usage
  python training/azure/train_cpt.py --data training/cpt.jsonl --out outputs/tunisian_cpt
  python training/azure/train_cpt.py --dry-run          # no GPU needed: validates data+config
Resume
  python training/azure/train_cpt.py --resume outputs/tunisian_cpt/checkpoint-500
"""
import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (set_seed_all, sha256, autoconfig, pick_attention,
                    training_args, write_manifest)   # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/cpt.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="outputs/tunisian_cpt")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.02, help="held-out fraction for eval_loss")
    ap.add_argument("--val-max", type=int, default=800, help="cap val blocks (keeps eval fast)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from VRAM")
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--quant", choices=["auto", "none", "4bit"], default="auto")
    ap.add_argument("--neftune", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none", help="none | mlflow | tensorboard")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    a = parse()
    set_seed_all(a.seed)
    t0 = time.time()

    hw = autoconfig(a.seq)
    quant = a.quant if a.quant != "auto" else hw["quant"]
    batch = a.batch or hw["batch"]
    accum = a.grad_accum or max(1, hw["grad_accum"] * hw["batch"] // batch)
    use_bf16 = hw["bf16"] and quant == "none"
    print("=" * 72)
    print("GPU        : {}  ({} GB, x{})  -> tier {}".format(
        hw["name"], hw["vram_gb"], hw["gpus"], hw["tier"]))
    print("precision  : {} | quant={} | attn={}".format(
        "bf16" if use_bf16 else "fp16", quant, pick_attention()))
    print("batch      : {} x accum {} x seq {} = {:,} tokens/step".format(
        batch, accum, a.seq, batch * accum * a.seq))
    print("=" * 72)

    # ---------------- data ----------------
    texts = [json.loads(l)["text"] for l in open(a.data, encoding="utf-8") if l.strip()]
    print("CPT lines: {:,}  from {}".format(len(texts), a.data))

    if a.dry_run:
        approx = int(sum(len(t.split()) for t in texts) * 1.6)
        print("[dry-run] approx tokens {:,} -> ~{:.0f} optimizer steps".format(
            approx, approx / (batch * accum * a.seq)))
        write_manifest(a.out, {"stage": "cpt", "dry_run": True, "config": vars(a),
                               "hardware": hw, "data_sha": sha256(a.data, 64),
                               "lines": len(texts), "approx_tokens": approx})
        return

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, default_data_collator)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # pack into fixed-length blocks (standard, efficient CPT)
    ds = Dataset.from_list([{"text": t} for t in texts])
    ds = ds.map(lambda b: tok(b["text"]), batched=True, remove_columns=["text"])
    EOS = tok.eos_token_id

    def group(ex):
        buf = []
        for ids in ex["input_ids"]:
            buf += ids + [EOS]
        n = (len(buf) // a.seq) * a.seq
        blocks = [buf[i:i + a.seq] for i in range(0, n, a.seq)]
        return {"input_ids": blocks,
                "attention_mask": [[1] * a.seq for _ in blocks],
                "labels": [b[:] for b in blocks]}

    packed = ds.map(group, batched=True, batch_size=1000, remove_columns=ds.column_names)
    packed = packed.shuffle(seed=a.seed)
    n_val = min(a.val_max, max(1, int(len(packed) * a.val_frac)))
    val_ds = packed.select(range(n_val))
    train_ds = packed.select(range(n_val, len(packed)))
    total_tokens = len(train_ds) * a.seq
    print("packed blocks: train {:,} | val {:,} | tokens/epoch {:,}".format(
        len(train_ds), n_val, total_tokens))

    # ---------------- model ----------------
    kw = {"device_map": {"": 0}, "attn_implementation": pick_attention()}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if hw["bf16"] else torch.float16,
            bnb_4bit_use_double_quant=True)
    kw["torch_dtype"] = torch.bfloat16 if hw["bf16"] else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.config.use_cache = False
    if quant == "4bit":
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=a.lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

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
        neftune_noise_alpha=a.neftune, report_to=a.report_to, seed=a.seed,
        label_names=["labels"], remove_unused_columns=False)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=default_data_collator)
    res = trainer.train(resume_from_checkpoint=a.resume)
    metrics = trainer.evaluate()
    ppl = math.exp(min(20, metrics["eval_loss"]))

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    write_manifest(a.out, {
        "stage": "cpt", "config": vars(a), "hardware": hw,
        "effective": {"batch": batch, "grad_accum": accum, "quant": quant,
                      "precision": "bf16" if use_bf16 else "fp16"},
        "data": {"file": a.data, "sha": sha256(a.data, 64), "lines": len(texts),
                 "train_blocks": len(train_ds), "val_blocks": n_val,
                 "tokens_per_epoch": total_tokens},
        "metrics": {"train_loss": res.training_loss, "eval_loss": metrics["eval_loss"],
                    "val_perplexity": round(ppl, 2)},
        "runtime_hours": round((time.time() - t0) / 3600, 2)})
    print("\nDONE. val_loss={:.4f} (ppl {:.1f}) -> {}".format(metrics["eval_loss"], ppl, a.out))


if __name__ == "__main__":
    main()
