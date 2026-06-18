"""
7adethni serving API — loads the base model + your trained LoRA adapter and generates
Tunisian Arabizi for each writing feature. The Chrome extension calls POST /generate.

Run (on a GPU box: RunPod / Vast / a server with an NVIDIA GPU):
    pip install -r requirements.txt
    export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"      # must match what you trained on
    export ADAPTER_DIR="./tunisian_lora"               # the downloaded LoRA folder
    export API_KEYS="testkey123"                        # comma-separated allowed keys (optional)
    uvicorn app:app --host 0.0.0.0 --port 8000

Then point the extension's API URL at  http://<host>:8000
"""
import os
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompts import build_messages, FEATURE_LABELS

BASE_MODEL  = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./tunisian_lora")
API_KEYS    = set(k for k in os.environ.get("API_KEYS", "").split(",") if k)

app = FastAPI(title="7adethni API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = _tokenizer = None

def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    try:
        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    except Exception:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                                device_map="auto", torch_dtype=torch.float16)
    _model = PeftModel.from_pretrained(base, ADAPTER_DIR) if os.path.isdir(ADAPTER_DIR) else base
    _model.eval()


class GenReq(BaseModel):
    feature: str = "idea"      # post | reply | rewrite | translate | caption | idea
    text: str
    tone: str = "normal"
    max_new_tokens: int = 220
    temperature: float = 0.75


@app.on_event("startup")
def _startup():
    _load()


@app.get("/health")
def health():
    return {"ok": _model is not None, "base": BASE_MODEL, "features": list(FEATURE_LABELS)}


@app.post("/generate")
def generate(req: GenReq, x_api_key: str | None = Header(default=None)):
    if API_KEYS and x_api_key not in API_KEYS:
        raise HTTPException(401, "invalid or missing API key")
    if not (req.text or "").strip():
        raise HTTPException(400, "text is empty")
    _load()
    msgs = build_messages(req.feature, req.text, req.tone)
    ids = _tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=min(req.max_new_tokens, 400), do_sample=True,
            temperature=req.temperature, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id)
    text = _tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return {"feature": req.feature, "tone": req.tone, "output": text}
