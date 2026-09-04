"""
Run the result-trusted training suite on Modal serverless GPU.

Azure gave you 0 GPU quota; Modal gives you a real GPU on the free monthly credits you
already have. The training scripts are unchanged (training/azure/*.py) - this only supplies
the GPU, the data volume, and the model cache.

Cost on A10G (~$1.10/h):  CPT ~4h + SFT ~1.5h + ablation ~1.5h + eval ~0.5h  =  ~$8
                          -> comfortably inside the $30/month free credits.

----------------------------------------------------------------------------------
ONE-TIME SETUP (from the repo root)
    pip install modal && modal setup
    modal volume create 7adethni-data
    modal volume put 7adethni-data training/cpt.jsonl                     /cpt.jsonl
    modal volume put 7adethni-data training/sft_real.jsonl                /sft_real.jsonl
    modal volume put 7adethni-data rag/lexicon.jsonl                      /lexicon.jsonl
    modal volume put 7adethni-data dataset/eval_set.jsonl                 /eval_set.jsonl
    modal volume put 7adethni-data dataset/tunbench_comprehension.jsonl   /comprehension.jsonl

RUN (each command returns when that stage finishes)
    modal run training/modal/train_modal.py::cpt
    modal run training/modal/train_modal.py::sft
    modal run training/modal/train_modal.py::sft_ablation
    modal run training/modal/train_modal.py::evaluate

GET RESULTS
    modal volume get 7adethni-out /tunisian_final ./outputs/
    modal volume get 7adethni-out /eval          ./outputs/
----------------------------------------------------------------------------------
"""
import os
import modal

GPU = os.environ.get("MODAL_GPU", "A10G")     # A10G | A100 | A100-80GB | L4
HOURS = 60 * 60

app = modal.App("7adethni-training")

data_vol = modal.Volume.from_name("7adethni-data", create_if_missing=True)
out_vol = modal.Volume.from_name("7adethni-out", create_if_missing=True)
hf_vol = modal.Volume.from_name("7adethni-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers==4.51.3", "peft>=0.12", "accelerate>=0.30",
        "datasets>=2.19", "bitsandbytes>=0.43", "sentencepiece", "rapidfuzz>=3",
    )
    .env({"HF_HOME": "/cache/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # the training suite + the lexicon tooling it imports (structure must be preserved)
    .add_local_dir("training/azure", "/root/repo/training/azure")
    .add_local_dir("rag", "/root/repo/rag", ignore=["*.jsonl", "__pycache__"])
    .add_local_dir("dataset/tools", "/root/repo/dataset/tools", ignore=["__pycache__"])
)

VOLUMES = {"/data": data_vol, "/outputs": out_vol, "/cache": hf_vol}


def _run(script, args):
    """Run one of the training scripts inside the container and persist the outputs."""
    import shutil, subprocess, sys
    from pathlib import Path
    # the scripts resolve sibling data via the repo layout; lexicon lives on the volume
    Path("/root/repo/rag").mkdir(parents=True, exist_ok=True)
    if not Path("/root/repo/rag/lexicon.jsonl").exists():
        shutil.copy("/data/lexicon.jsonl", "/root/repo/rag/lexicon.jsonl")
    cmd = [sys.executable, f"/root/repo/training/azure/{script}"] + args
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root/repo")
    out_vol.commit()
    hf_vol.commit()


@app.function(gpu=GPU, timeout=8 * HOURS, volumes=VOLUMES)
def cpt(extra: str = ""):
    """Stage 1 - continued pre-training on Tunisian text."""
    _run("train_cpt.py", [
        "--data", "/data/cpt.jsonl",
        "--out", "/outputs/tunisian_cpt",
        "--seq", "1024", "--epochs", "1", "--lora-r", "64",
    ] + (extra.split() if extra else []))


@app.function(gpu=GPU, timeout=6 * HOURS, volumes=VOLUMES)
def sft(extra: str = ""):
    """Stage 2 - instruction tuning on top of the CPT adapter."""
    _run("train_sft.py", [
        "--data", "/data/sft_real.jsonl",
        "--cpt", "/outputs/tunisian_cpt",
        "--out", "/outputs/tunisian_final",
        "--seq", "1024", "--epochs", "2",
    ] + (extra.split() if extra else []))


@app.function(gpu=GPU, timeout=6 * HOURS, volumes=VOLUMES)
def sft_ablation(extra: str = ""):
    """Control group: same SFT WITHOUT CPT -> proves whether CPT was worth it."""
    _run("train_sft.py", [
        "--data", "/data/sft_real.jsonl",
        "--cpt", "none",
        "--out", "/outputs/sft_only",
        "--seq", "1024", "--epochs", "2",
    ] + (extra.split() if extra else []))


@app.function(gpu=GPU, timeout=3 * HOURS, volumes=VOLUMES)
def evaluate(extra: str = ""):
    """TunBench: greedy decoding, ablation table, bootstrap confidence intervals."""
    _run("evaluate.py", [
        "--eval", "/data/eval_set.jsonl",
        "--comprehension", "/data/comprehension.jsonl",
        "--adapters", "base=none",
        "cpt=/outputs/tunisian_cpt",
        "final=/outputs/tunisian_final",
        "sft_only=/outputs/sft_only",
        "--out", "/outputs/eval",
    ] + (extra.split() if extra else []))


@app.function(gpu=GPU, timeout=1 * HOURS, volumes=VOLUMES)
def check():
    """Cheap sanity check: is the GPU what we expect, and is the data in place?"""
    import subprocess, sys, os
    subprocess.run(["nvidia-smi"], check=False)
    print("\n/data:", os.listdir("/data"))
    print("/outputs:", os.listdir("/outputs") if os.path.exists("/outputs") else "empty")
    _run("train_cpt.py", ["--data", "/data/cpt.jsonl", "--out", "/outputs/_dry", "--dry-run"])
