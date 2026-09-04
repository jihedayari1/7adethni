#!/usr/bin/env python3
"""Builds training/cpt_kaggle.ipynb — Stage 1: QLoRA continued-pretraining on Tunisian text."""
import json
from pathlib import Path

def md(*s):   return {"cell_type": "markdown", "metadata": {}, "source": list(s)}
def code(*s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(s)}

cells = []

cells.append(md(
"# 🇹🇳 Stage 1 — CPT (continued pre-training) on Tunisian\n",
"\n",
"Teaches the base model to **speak natural Arabizi + understand Tunisian** by training on raw\n",
"real text (TUNIZI + transliterated corpus + Arabic-script Tunisian). This is the step that was\n",
"missing before. Output = a LoRA adapter you then fine-tune in Stage 2 (SFT).\n",
"\n",
"## Setup\n",
"1. **+ Add Data → New Dataset** → upload `training/cpt.jsonl` (from `build_cpt_corpus.py`).\n",
"2. Accelerator **GPU T4 x2**, Internet **ON**.\n",
"3. **Run All** (~6–8 h for ~7M tokens). Download `/kaggle/working/tunisian_cpt` after.\n"
))

cells.append(md("## 1. Install"))
cells.append(code(
"import importlib.metadata as _md\n",
"TARGET_TF = '4.51.3'      # Kaggle's default transformers can be broken/mismatched\n",
"try: _cur = _md.version('transformers')\n",
"except Exception: _cur = None\n",
"print('transformers found:', _cur)\n",
"if _cur != TARGET_TF:\n",
"    !pip uninstall -y -q transformers tokenizers\n",
"    !pip install -q 'transformers==4.51.3' 'peft>=0.12' 'bitsandbytes>=0.43' accelerate datasets\n",
"    print('*** INSTALLED - KERNEL IS RESTARTING. When it stops, click Run All again. ***')\n",
"    import IPython; IPython.Application.instance().kernel.do_shutdown(True)\n",
"else:\n",
"    !pip install -q 'peft>=0.12' 'bitsandbytes>=0.43' accelerate datasets\n",
"    print('env OK - continuing')\n"
))

cells.append(md("## 2. Config"))
cells.append(code(
"import os\n",
"os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # ONE GPU: T4x2 model-split breaks the loss\n",
"os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # stop VRAM fragmentation\n",
"MODEL_NAME  = 'Qwen/Qwen2.5-7B-Instruct'   # keep chat ability; low LR won't wreck it\n",
"MAX_SEQ_LEN = 1024        # packed blocks (2048 OOMs: Qwen vocab is 152k -> huge logits)\n",
"EPOCHS      = 1           # 1 pass over the corpus (CPT is about exposure, not memorization)\n",
"LR          = 2e-4\n",
"LORA_R      = 32          # bigger adapter for CPT (more capacity to absorb the language)\n",
"BATCH       = 1\n",
"GRAD_ACCUM  = 16\n",
"NEFTUNE     = 5           # noised-embedding -> better generalization\n",
"print('CPT config ready ->', MODEL_NAME)\n"
))

cells.append(md("## 3. Load the CPT text"))
cells.append(code(
"import json, glob\n",
"def find(name):\n",
"    h = glob.glob(f'/kaggle/input/**/{name}', recursive=True)\n",
"    if not h: raise FileNotFoundError(f'{name} not found under /kaggle/input')\n",
"    return h[0]\n",
"texts = [json.loads(l)['text'] for l in open(find('cpt.jsonl'), encoding='utf-8') if l.strip()]\n",
"print('CPT lines:', len(texts), '| ~tokens:', int(sum(len(t.split()) for t in texts)*1.4))\n"
))

cells.append(md("## 4. Load model in 4-bit + LoRA (all linear layers)"))
cells.append(code(
"import gc, torch\n",
"# free any model left on the GPU by a previous run (Jupyter keeps it alive)\n",
"for _v in ['trainer','model','base','packed','train_ds']:\n",
"    if _v in globals(): del globals()[_v]\n",
"gc.collect(); torch.cuda.empty_cache()\n",
"print(f'free VRAM before load: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB of {torch.cuda.mem_get_info()[1]/1e9:.1f} GB')\n",
"assert torch.cuda.mem_get_info()[0]/1e9 > 10, 'GPU not empty -> Run > Factory reset, then Run All'\n",
"from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
"from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
"bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',\n",
"                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)\n",
"tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
"if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token\n",
"model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,\n",
"                                             device_map={'': 0}, torch_dtype=torch.float16)\n",
"model.config.use_cache = False\n",
"model = prepare_model_for_kbit_training(model)\n",
"lora = LoraConfig(r=LORA_R, lora_alpha=LORA_R*2, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',\n",
"    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])\n",
"model = get_peft_model(model, lora); model.print_trainable_parameters()\n"
))

cells.append(md("## 5. Pack the corpus into fixed-length blocks (efficient CPT)"))
cells.append(code(
"from datasets import Dataset\n",
"ds = Dataset.from_list([{'text': t} for t in texts])\n",
"ds = ds.map(lambda b: tokenizer(b['text']), batched=True, remove_columns=['text'])\n",
"EOS = tokenizer.eos_token_id\n",
"def group(ex):\n",
"    buf = []\n",
"    for ids in ex['input_ids']: buf += ids + [EOS]\n",
"    n = (len(buf)//MAX_SEQ_LEN)*MAX_SEQ_LEN\n",
"    blocks = [buf[i:i+MAX_SEQ_LEN] for i in range(0, n, MAX_SEQ_LEN)]\n",
"    return {'input_ids': blocks, 'attention_mask': [[1]*MAX_SEQ_LEN for _ in blocks],\n",
"            'labels': [b[:] for b in blocks]}\n",
"packed = ds.map(group, batched=True, batch_size=1000, remove_columns=ds.column_names)\n",
"steps = max(1, len(packed)//(BATCH*GRAD_ACCUM))\n",
"print(f'packed blocks: {len(packed)} => ~{len(packed)*MAX_SEQ_LEN:,} tokens/epoch')\n",
"print(f'optimizer steps: {steps}   rough T4 estimate: {len(packed)*MAX_SEQ_LEN/300/3600:.1f} h')\n"
))

cells.append(md("## 6. Train (QLoRA CPT)"))
cells.append(code(
"from transformers import Trainer, TrainingArguments, default_data_collator\n",
"args = TrainingArguments(\n",
"    output_dir='cpt_out', per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,\n",
"    num_train_epochs=EPOCHS, learning_rate=LR, warmup_ratio=0.03, logging_steps=25,\n",
"    fp16=True, optim='paged_adamw_8bit', weight_decay=0.01, lr_scheduler_type='cosine',\n",
"    gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},\n",
"    neftune_noise_alpha=NEFTUNE, report_to='none', seed=42,\n",
"    save_strategy='steps', save_steps=50, save_total_limit=2, label_names=['labels'], remove_unused_columns=False)  # crash-safe: partial adapter survives\n",
"trainer = Trainer(model=model, args=args, train_dataset=packed, data_collator=default_data_collator)\n",
"trainer.train()\n"
))

cells.append(md("## 7. Save the CPT adapter"))
cells.append(code(
"model.save_pretrained('/kaggle/working/tunisian_cpt')\n",
"tokenizer.save_pretrained('/kaggle/working/tunisian_cpt')\n",
"print('saved -> /kaggle/working/tunisian_cpt  (upload this as a Kaggle Dataset for Stage 2)')\n"
))

cells.append(md("## 8. Quick smell-test (does it produce natural Arabizi?)"))
cells.append(code(
"model.eval()\n",
"def gen(p, n=60):\n",
"    ids = tokenizer(p, return_tensors='pt').to('cuda')\n",
"    out = model.generate(**ids, max_new_tokens=n, do_sample=True, temperature=0.8, top_p=0.9,\n",
"                         repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)\n",
"    return tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)\n",
"for p in ['3aslema, ', 'el 7keya mta3 el yom ', 'chnowa ']:\n",
"    print(repr(p), '->', gen(p), '\\n')\n"
))

cells.append(md(
"## Next\n",
"CPT is *not* the finished model — it's language exposure. Now run **Stage 2 (SFT)**:\n",
"upload `/kaggle/working/tunisian_cpt` as a dataset + `sft_real.jsonl`, then run `sft_kaggle.ipynb`.\n"
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "cpt_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
