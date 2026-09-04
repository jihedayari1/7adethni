"""Shared helpers for the Azure training suite: hardware auto-config, seeding, manifests.

Everything here exists to make a run REPRODUCIBLE and AUDITABLE:
  * one seed controls python/numpy/torch
  * the exact data file is hashed, so you can prove which data produced which model
  * a manifest.json is written next to every adapter (config + hashes + metrics + timing)
"""
import hashlib, json, os, random, subprocess, sys, time
from pathlib import Path


def set_seed_all(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sha256(path, limit_mb=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit_mb and n > limit_mb * (1 << 20):
                break
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpus": 0, "name": "cpu", "vram_gb": 0, "bf16": False}
        p = torch.cuda.get_device_properties(0)
        return {"gpus": torch.cuda.device_count(), "name": p.name,
                "vram_gb": round(p.total_memory / 1e9, 1),
                "bf16": bool(torch.cuda.is_bf16_supported())}
    except Exception as e:
        return {"gpus": 0, "name": f"unknown({e})", "vram_gb": 0, "bf16": False}


def autoconfig(seq_len, target_eff_seqs=32):
    """Pick precision / quantization / batch from the actual GPU. Azure-friendly."""
    g = gpu_info()
    v = g["vram_gb"]
    if v >= 70:     tier, quant, batch = "A100-80GB / H100", "none", 8
    elif v >= 38:   tier, quant, batch = "A100-40GB", "none", 4
    elif v >= 30:   tier, quant, batch = "V100-32GB / L40", "none", 2
    elif v >= 22:   tier, quant, batch = "A10 / L4 24GB", "4bit", 2
    elif v >= 14:   tier, quant, batch = "T4 / V100-16GB", "4bit", 1
    else:           tier, quant, batch = "cpu/small", "4bit", 1
    if seq_len >= 4096:
        batch = max(1, batch // 2)
    return {"tier": tier, "quant": quant, "batch": batch,
            "grad_accum": max(1, target_eff_seqs // batch),
            "bf16": g["bf16"] and quant == "none",
            "fp16": (not g["bf16"]) or quant != "none",
            **g}


def pick_attention():
    """flash-attn if installed (big speedup on A100/H100), else PyTorch SDPA."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def training_args(**kw):
    """Build TrainingArguments, tolerating the eval_strategy/evaluation_strategy rename."""
    from transformers import TrainingArguments
    try:
        return TrainingArguments(**kw)
    except TypeError:
        if "eval_strategy" in kw:
            kw["evaluation_strategy"] = kw.pop("eval_strategy")
        return TrainingArguments(**kw)


def write_manifest(out_dir, payload):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(), **payload}
    (out / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RUN MANIFEST ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])
    return payload
