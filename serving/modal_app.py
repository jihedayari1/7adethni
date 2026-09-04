"""
7adethni model server on Modal (serverless GPU, scale-to-zero).

Why Modal: $30/mo free credits ≈ ~25 A10G-hours ≈ 15-20k generations — serves the first
hundreds of users for $0/mo. Scale-to-zero means you pay nothing while idle; cold start
~60s with the weights cached in a Volume (the UI shows a warm-up message).

Model: Qwen2.5-7B-Instruct-AWQ (4-bit, official checkpoint) on an A10G via vLLM.
Best-of-2 is done with ONE vLLM call (SamplingParams n=2) — same latency class as n=1.
The grounding layer (lexicon scoring, meanings, canonicalization) runs in-container.

Deploy (once, from the repo root):
    pip install modal
    modal setup                                   # login (free account)
    modal secret create 7adethni-keys MODEL_API_KEY=<pick-a-key>   # required (use "" to disable auth)
    modal deploy serving/modal_app.py
    # -> prints a URL like https://<you>--7adethni-serving-server-api.modal.run

Then point the gateway at it:
    fly secrets set MODEL_API_URL=https://...modal.run MODEL_API_KEY=<same-key>

Same API contract as serving/app.py:  GET /health,  POST /generate.
"""
import os
import modal

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
MINUTES = 60

app = modal.App("7adethni-serving")

MODEL_FP16 = "Qwen/Qwen2.5-7B-Instruct"   # base used when a LoRA adapter is attached
# where you upload your trained adapter (Stage-2 `tunisian_final`):
#   modal volume put 7adethni-adapter ./tunisian_final /tunisian_final
ADAPTER_DIR = "/adapter/tunisian_final"

# HF weights cached across cold starts (first boot downloads ~5.6GB, later boots ~60s)
hf_cache = modal.Volume.from_name("7adethni-hf-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("7adethni-adapter", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "rapidfuzz>=3", "fastapi[standard]")
    .env({"HF_HOME": "/cache/hf", "VLLM_NO_USAGE_STATS": "1"})
    # ship the grounding stack into the container
    .add_local_file("rag/normalizer.py",  "/root/rag/normalizer.py")
    .add_local_file("rag/retriever.py",   "/root/rag/retriever.py")
    .add_local_file("rag/lexicon.jsonl",  "/root/rag/lexicon.jsonl")
    .add_local_file("serving/prompts.py",   "/root/serving/prompts.py")
    .add_local_file("serving/grounding.py", "/root/serving/grounding.py")
)


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache, "/adapter": adapter_vol},
    scaledown_window=4 * MINUTES,          # scale to zero after 4 idle minutes
    timeout=10 * MINUTES,
    secrets=[modal.Secret.from_name("7adethni-keys")],   # holds MODEL_API_KEY
)
class Server:
    @modal.enter()
    def load(self):
        import os, sys
        sys.path.insert(0, "/root/serving")
        sys.path.insert(0, "/root/rag")
        from vllm import LLM, SamplingParams          # noqa
        self.SamplingParams = SamplingParams
        # If a trained adapter is present in the volume -> serve base(fp16)+LoRA.
        # Otherwise -> serve the AWQ base (smaller/faster). Same API either way.
        self.lora = None
        if os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
            from vllm.lora.request import LoRARequest
            self.llm = LLM(model=MODEL_FP16, enable_lora=True, max_lora_rank=64,
                           gpu_memory_utilization=0.92, max_model_len=2048, download_dir="/cache/hf")
            self.lora = LoRARequest("tunisian", 1, ADAPTER_DIR)
            self.model_name = f"{MODEL_FP16}+tunisian_final"
        else:
            self.llm = LLM(model=MODEL_ID, quantization="awq",
                           gpu_memory_utilization=0.90, max_model_len=2048, download_dir="/cache/hf")
            self.model_name = MODEL_ID
        self.tok = self.llm.get_tokenizer()
        import grounding, prompts                      # noqa
        self.g, self.p = grounding, prompts
        hf_cache.commit()                              # persist freshly downloaded weights

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel

        API_KEY = os.environ.get("MODEL_API_KEY", "")
        web = FastAPI(title="7adethni model server (Modal)")

        class GenReq(BaseModel):
            feature: str = "translate"
            text: str
            tone: str = "normal"
            max_new_tokens: int = 220
            temperature: float = 0.75

        @web.get("/health")
        def health():
            return {"ok": True, "model": self.model_name, "adapter": self.lora is not None,
                    "backend": "modal+vllm"}

        @web.post("/generate")
        def generate(req: GenReq, x_api_key: str | None = Header(default=None)):
            if API_KEY and x_api_key != API_KEY:
                raise HTTPException(401, "invalid or missing API key")
            if not (req.text or "").strip():
                raise HTTPException(400, "text is empty")

            # comprehension grounding: gloss confident slang in the INPUT
            context = self.g.input_meanings(req.text)
            msgs = self.p.build_messages(req.feature, req.text, req.tone, context=context)
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            # best-of-2 in a single batched call, then keep the most coherent
            sp = self.SamplingParams(n=2, temperature=req.temperature, top_p=0.9,
                                     repetition_penalty=1.1,
                                     max_tokens=min(req.max_new_tokens, 400))
            gen_kw = {"lora_request": self.lora} if self.lora else {}
            cands = [o.text.strip() for o in self.llm.generate([prompt], sp, **gen_kw)[0].outputs]
            idx, _ = self.g.pick_best(cands)
            text = self.g.canonicalize(cands[idx])
            q = self.g.quality(text)
            return {"feature": req.feature, "tone": req.tone, "output": text,
                    "quality": {"real_word_rate": round(q["rate"], 3), "oov": q["oov"]}}

        return web


@app.local_entrypoint()
def test():
    """`modal run serving/modal_app.py` — one warm test generation before deploying."""
    import json, urllib.request
    url = Server().api.get_web_url()  # type: ignore[attr-defined]
    body = json.dumps({"feature": "translate", "text": "Comment ça va mon ami ?"}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    print(urllib.request.urlopen(req, timeout=600).read().decode())
