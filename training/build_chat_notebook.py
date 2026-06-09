#!/usr/bin/env python3
"""Builds training/tunisian_chat_kaggle.ipynb — load the trained LoRA + chat (for testers)."""
import json
from pathlib import Path

def md(*s):   return {"cell_type": "markdown", "metadata": {}, "source": list(s)}
def code(*s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(s)}

cells = []
cells.append(md(
"# 🇹🇳 Chat with the Tunisian model (testers' notebook)\n",
"\n",
"Loads the **already-trained** LoRA adapter and lets you chat. No training, fast to start.\n",
"\n",
"## Setup\n",
"1. **+ Add Data** → add the LoRA dataset the owner shared (the `tunisian_lora` folder,\n",
"   uploaded as a Kaggle Dataset — it contains `adapter_config.json`).\n",
"2. Right panel: **Accelerator = GPU T4**, **Internet = ON** (to download the base model).\n",
"3. **Run All**, then use the chat cell at the bottom.\n"
))

cells.append(md("## 1. Install"))
cells.append(code(
"%%capture\n",
"!pip install -q -U \"transformers<5\" \"peft>=0.12\" \"bitsandbytes>=0.43\" accelerate\n"
))

cells.append(md("## 2. Config — base model MUST match what was trained"))
cells.append(code(
"BASE_MODEL = 'Qwen/Qwen2.5-3B-Instruct'   # change if the owner trained on 7B/8B\n",
"SYSTEM = ('Enti assistant tunsi (service client w 7adith 3am). Jaweb DIMA bel derja tounsiya '\n",
"          'bel arabizi (7ourouf latiniya w arqam), b tari9a tabi3iya w 9sira. Ken el user yekteb '\n",
"          'bel 3arbi wala faransi wala anglais, efhem w jaweb bel arabizi.')\n",
"import glob, os\n",
"def find_adapter():\n",
"    h = glob.glob('/kaggle/input/**/adapter_config.json', recursive=True)\n",
"    if not h: raise FileNotFoundError('LoRA adapter not found — add the shared tunisian_lora dataset')\n",
"    return os.path.dirname(h[0])\n",
"ADAPTER = find_adapter(); print('adapter:', ADAPTER)\n"
))

cells.append(md("## 3. Load base model (4-bit) + the trained adapter"))
cells.append(code(
"import torch\n",
"from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
"from peft import PeftModel\n",
"\n",
"bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',\n",
"                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)\n",
"try:\n",
"    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)\n",
"except Exception:\n",
"    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)\n",
"if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token\n",
"\n",
"base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,\n",
"                                            device_map='auto', torch_dtype=torch.float16)\n",
"model = PeftModel.from_pretrained(base, ADAPTER)\n",
"model.eval()\n",
"print('ready ✅')\n"
))

cells.append(md("## 4. Chat helper"))
cells.append(code(
"def generate(user_msg, system=SYSTEM, max_new_tokens=200, temperature=0.7):\n",
"    msgs = [{'role':'system','content':system},{'role':'user','content':user_msg}]\n",
"    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to('cuda')\n",
"    with torch.no_grad():\n",
"        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),\n",
"            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.9,\n",
"            repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)\n",
"    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()\n"
))

cells.append(md("## 5. Quick examples"))
cells.append(code(
"for m in ['3aslema chna7welek?','قداش تمن التوصيل لتونس؟','give me a dinner idea','9olli nokta']:\n",
"    print('🧑', m); print('🤖', generate(m), '\\n')\n"
))

cells.append(md("## 6. 💬 Live chat — run this cell and type. Type `quit` to stop."))
cells.append(code(
"print('Chat bel tounsi! ekteb \"quit\" bch to5rej.\\n')\n",
"while True:\n",
"    try:\n",
"        msg = input('🧑 enti: ')\n",
"    except (EOFError, KeyboardInterrupt):\n",
"        break\n",
"    if msg.strip().lower() in ('quit','exit','q',''): print('بسلامة 👋'); break\n",
"    print('🤖', generate(msg), '\\n')\n"
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "tunisian_chat_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
