#!/usr/bin/env python3
"""Builds training/tunisian_finetune_kaggle.ipynb — standard HF QLoRA (NO Unsloth)."""
import json
from pathlib import Path

def md(*s):   return {"cell_type": "markdown", "metadata": {}, "source": list(s)}
def code(*s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(s)}

cells = []

cells.append(md(
"# 🇹🇳 Tunisian Arabizi Assistant — Fine-tune + Evaluate (Kaggle, no-Unsloth)\n",
"\n",
"Standard HuggingFace **QLoRA** (transformers + peft + bitsandbytes). Robust — avoids the\n",
"Unsloth training-step bug. Trains, measures **dialect-rate before vs after**, and lets you chat.\n",
"\n",
"## Setup (one-time)\n",
"1. **+ Add Data → New Dataset** → upload `cs_pairs.jsonl`, `parallel_pairs.jsonl`, `eval_set.jsonl`.\n",
"2. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON**.\n",
"3. **Run All**. Default model is Qwen2.5-3B (fast + reliable on a T4); ~30–60 min.\n"
))

cells.append(md("## 1. Install (pinned, no Unsloth)"))
cells.append(code(
"%%capture\n",
"!pip install -q -U \"transformers<5\" \"peft>=0.12\" \"bitsandbytes>=0.43\" accelerate datasets\n"
))

cells.append(md("## 2. Config"))
cells.append(code(
"MODEL_NAME      = 'Qwen/Qwen2.5-3B-Instruct'\n",
"#  bigger (set BATCH=1, GRAD_ACCUM=16): 'Qwen/Qwen2.5-7B-Instruct' | 'Qwen/Qwen3-8B'\n",
"MAX_SEQ_LEN     = 1024\n",
"\n",
"CONV_UPSAMPLE   = 3       # repeat the hand-written conversational pairs (the priority)\n",
"PARALLEL_SAMPLE = 5000    # how many translation/comprehension pairs to mix in\n",
"\n",
"EPOCHS          = 1\n",
"LR              = 2e-4\n",
"BATCH           = 2\n",
"GRAD_ACCUM      = 4\n",
"EVAL_MAX_NEW    = 160\n",
"\n",
"SYSTEM = ('Enti assistant tunsi (service client w 7adith 3am). Jaweb DIMA bel derja tounsiya '\n",
"          'bel arabizi (7ourouf latiniya w arqam), b tari9a tabi3iya w 9sira. Ken el user yekteb '\n",
"          'bel 3arbi wala faransi wala anglais, efhem w jaweb bel arabizi. Ken talbou translation, '\n",
"          'a3mel el li talbou.')\n",
"print('config ready ->', MODEL_NAME)\n"
))

cells.append(md("## 3. Load + mix the data"))
cells.append(code(
"import json, glob, random\n",
"random.seed(42)\n",
"def find(name):\n",
"    h = glob.glob(f'/kaggle/input/**/{name}', recursive=True)\n",
"    if not h: raise FileNotFoundError(f'{name} not found under /kaggle/input — upload your dataset')\n",
"    return h[0]\n",
"def load_jsonl(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]\n",
"\n",
"conv = load_jsonl(find('cs_pairs.jsonl'))\n",
"par  = load_jsonl(find('parallel_pairs.jsonl')); random.shuffle(par); par = par[:PARALLEL_SAMPLE]\n",
"train_rows = [{'instruction': r['instruction'], 'output': r['output']} for r in (conv*CONV_UPSAMPLE + par)]\n",
"random.shuffle(train_rows)\n",
"print(f'conversational {len(conv)} x{CONV_UPSAMPLE} | parallel {len(par)} | TOTAL train {len(train_rows)}')\n"
))

cells.append(md("## 4. Load model in 4-bit + attach LoRA"))
cells.append(code(
"import torch\n",
"from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
"from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
"\n",
"bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',\n",
"                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)\n",
"tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
"if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token\n",
"tokenizer.padding_side = 'right'\n",
"\n",
"model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,\n",
"                                             device_map='auto', torch_dtype=torch.float16)\n",
"model.config.use_cache = False\n",
"model = prepare_model_for_kbit_training(model)\n",
"lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',\n",
"    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])\n",
"model = get_peft_model(model, lora)\n",
"model.print_trainable_parameters()\n"
))

cells.append(md("## 5. Chat helper (used for baseline + final test)"))
cells.append(code(
"def generate(user_msg, system=SYSTEM, max_new_tokens=EVAL_MAX_NEW, temperature=0.7):\n",
"    model.eval()\n",
"    msgs = [{'role':'system','content':system},{'role':'user','content':user_msg}]\n",
"    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to('cuda')\n",
"    with torch.no_grad():\n",
"        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),\n",
"            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.9,\n",
"            repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)\n",
"    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()\n"
))

cells.append(md("## 6. Dialect-rate metric (MSA-leakage detector)"))
cells.append(code(
"import re\n",
"TUN = set('barcha famma chnowa chnoua 9addech 9adech kifech 3lech win wa9tech ya5i 5ouya mte3 mta3 '\n",
"  'bch taw tawa ken kima brabi 3aslama 3aslema marhba chwaya barka fissa3 zin behi bahi mli7 3andi '\n",
"  '3andou 3andek hethi hetha hakka haka sa7a 3aychek yezzi 9a3ed mch mouch ma3andich n7eb t7eb 9olli '\n",
"  'ya3ni fel lel el enti ena houwa hia 9ahwa ghodwa lyoum wala 5dma 5edma dar bnin tounsi'.split())\n",
"FR = set('livraison prix commande merci bonjour stock weekend promo garantie couleur taille'.split())\n",
"MSA = set('hadha hadhihi alladhi allati sawfa laysa kayfa limadha 3indama ladhalika lakinna jiddan '\n",
"  'kathiran yumkinu yajibu na7nu inna sayakun dhalika tilka hunaka faqat aydan ladayna lan lam qad'.split())\n",
"MSA_AR = ['الذي','التي','سوف','ليس','كيف','عندما','لذلك','يمكن','يجب','نحن','جدا','هذا','هذه','ذلك']\n",
"_AR=re.compile(r'[\\u0600-\\u06ff]'); _NUM=re.compile(r\"[a-z][3-9'][a-z]\",re.I); _W=re.compile(r\"[a-z0-9'7359]+\",re.I)\n",
"def score_text(t):\n",
"    t=(t or '').strip(); toks=set(w.lower() for w in _W.findall(t))\n",
"    tun=len(toks&TUN)+len(toks&FR)+len(_NUM.findall(t.lower())); msa=len(toks&MSA)+sum(t.count(m) for m in MSA_AR)\n",
"    if _AR.search(t) and not (toks&TUN): return 'arabic_script'\n",
"    if tun==0 and msa==0: return 'unknown'\n",
"    if msa>0 and tun==0: return 'msa_leak'\n",
"    if tun>0 and msa==0: return 'tunisian'\n",
"    return 'tunisian' if tun>=2*msa else 'mixed'\n",
"def dialect_report(preds, tag=''):\n",
"    from collections import Counter\n",
"    c=Counter(score_text(p) for p in preds); n=len(preds) or 1\n",
"    print(f'--- {tag} (n={len(preds)}) ---')\n",
"    for k in ['tunisian','mixed','msa_leak','arabic_script','unknown']: print(f'  {k:14}: {c.get(k,0):4} ({c.get(k,0)/n:.0%})')\n",
"    r=c.get('tunisian',0)/n; print(f'  >> DIALECT RATE: {r:.0%}'); return r\n"
))

cells.append(md("## 7. BASELINE — dialect rate *before* training (expect very low)"))
cells.append(code(
"eval_rows = load_jsonl(find('eval_set.jsonl'))\n",
"print('eval items:', len(eval_rows))\n",
"base_preds = [generate(r['instruction']) for r in eval_rows]\n",
"base_rate = dialect_report(base_preds, 'BASELINE (before fine-tuning)')\n"
))

cells.append(md("## 8. Train (QLoRA, standard HF Trainer)"))
cells.append(code(
"from datasets import Dataset\n",
"from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq\n",
"\n",
"def fmt(r):\n",
"    text = tokenizer.apply_chat_template(\n",
"        [{'role':'system','content':SYSTEM},{'role':'user','content':r['instruction']},\n",
"         {'role':'assistant','content':r['output']}], tokenize=False)\n",
"    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)\n",
"    enc['labels'] = enc['input_ids'].copy()\n",
"    return enc\n",
"\n",
"base_ds = Dataset.from_list(train_rows)\n",
"train_ds = base_ds.map(fmt, remove_columns=base_ds.column_names)\n",
"collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)\n",
"\n",
"args = TrainingArguments(\n",
"    output_dir='outputs', per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,\n",
"    num_train_epochs=EPOCHS, learning_rate=LR, warmup_steps=10, logging_steps=20,\n",
"    fp16=True, optim='paged_adamw_8bit', weight_decay=0.01, lr_scheduler_type='cosine',\n",
"    gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},\n",
"    save_strategy='no', report_to='none', seed=42,\n",
")\n",
"trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)\n",
"model.config.use_cache = False\n",
"trainer.train()\n"
))

cells.append(md("## 9. Save the LoRA adapter (download from the Output tab)"))
cells.append(code(
"model.save_pretrained('/kaggle/working/tunisian_lora')\n",
"tokenizer.save_pretrained('/kaggle/working/tunisian_lora')\n",
"print('saved -> /kaggle/working/tunisian_lora')\n"
))

cells.append(md("## 10. AFTER — dialect rate *after* training (the dataset-efficiency result)"))
cells.append(code(
"after_preds = [generate(r['instruction']) for r in eval_rows]\n",
"after_rate = dialect_report(after_preds, 'AFTER fine-tuning')\n",
"print(f'\\n=== DATASET EFFICIENCY ===')\n",
"print(f'dialect rate  BEFORE: {base_rate:.0%}   AFTER: {after_rate:.0%}   (+{(after_rate-base_rate)*100:.0f} pts)')\n",
"print('\\n--- 8 sample before/after ---')\n",
"for r,b,a in list(zip(eval_rows, base_preds, after_preds))[:8]:\n",
"    print('Q  :', r['instruction'][:70]); print('OLD:', b[:90]); print('NEW:', a[:90]); print()\n"
))

cells.append(md("## 11. Talk to your model 🇹🇳 (understands Arabizi, Arabic letters, French, English)"))
cells.append(code(
"for msg in ['3aslema chna7welek?','قداش تمن التوصيل لصفاقس؟','give me a healthy breakfast idea',\n",
"            '9olli nokta tdha7ek','a7kili 7keya 9sira 3la el sabr']:\n",
"    print('🧑', msg); print('🤖', generate(msg), '\\n')\n"
))
cells.append(code(
"# >>> your turn: change this and re-run <<<\n",
"print(generate('3andi mochkla fel commande mte3i, chnowa na3mel?'))\n"
))

cells.append(md(
"## 12. Read the result\n",
"- The **BEFORE→AFTER dialect rate** is your dataset efficiency. Big jump = the data works.\n",
"- Scan the sample answers: natural Tunisian? note weak categories → add more pairs there.\n",
"- **Download** `/kaggle/working/tunisian_lora` (Output tab).\n",
"- For the real v1: `MODEL_NAME='Qwen/Qwen2.5-7B-Instruct'` (or `Qwen/Qwen3-8B`), `BATCH=1`,\n",
"  `GRAD_ACCUM=16`, and grow the conversational data.\n"
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "tunisian_finetune_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
