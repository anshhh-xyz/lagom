import asyncio
import json
import os
import sys
import threading
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add lagom-train to path to reuse prompt specifications
script_dir = os.path.dirname(os.path.abspath(__file__))
train_pkg_dir = os.path.join(script_dir, "..", "lagom-train")
if os.path.exists(train_pkg_dir) and train_pkg_dir not in sys.path:
    sys.path.append(train_pkg_dir)

from common import CATEGORY_INSTRUCTIONS, build_messages, hf_base_name  # noqa: E402

app = FastAPI(
    title="Lagom Humanizer API",
    description="High-performance humanization API powering Lagom Deep Mode",
    version="2.0.0",
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Model Cache
class ModelHolder:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "-1" else "cpu"
        self.model_type = "unloaded"  # 'adapter', 'merged', or 'unloaded'
        self.loaded_path = None
        self.lock = threading.Lock()

    def try_load(self):
        with self.lock:
            if self.model is not None:
                return True

            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"

            adapter_path = os.path.join(train_pkg_dir, "outputs", "adapter")
            merged_path = os.path.join(train_pkg_dir, "outputs", "merged")

            # Check if merged weights exist
            if os.path.exists(os.path.join(merged_path, "model.safetensors")) or os.path.exists(os.path.join(merged_path, "pytorch_model.bin")):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                print(f"[API] Loading merged model from: {merged_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(merged_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    merged_path,
                    torch_dtype=dtype,
                    device_map={"": 0} if torch.cuda.is_available() else "cpu",
                )
                self.model.eval()
                self.model_type = "merged"
                self.loaded_path = merged_path
                return True

            # Check if LoRA adapter exists
            run_json = os.path.join(adapter_path, "lagom_run.json")
            if os.path.exists(run_json):
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                with open(run_json) as f:
                    base_id = hf_base_name(json.load(f)["base_model"])

                print(f"[API] Loading base {base_id} + LoRA adapter from: {adapter_path}")
                dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    quantization_config=bnb if torch.cuda.is_available() else None,
                    device_map={"": 0} if torch.cuda.is_available() else "cpu",
                    torch_dtype=dtype,
                )
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
                self.model.eval()
                self.model_type = "adapter"
                self.loaded_path = adapter_path
                return True

            return False


holder = ModelHolder()


class HumanizeRequest(BaseModel):
    text: str = Field(..., description="The AI text to rewrite into authentic human prose")
    style: str = Field("general", description="Target style: general | essay | academic | email | document")
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    max_new_tokens: int = Field(800, ge=50, le=2048)


class HumanizeResponse(BaseModel):
    humanized: str
    style: str
    tokens: int
    elapsed_seconds: float
    model_mode: str
    word_count_original: int
    word_count_humanized: int


@app.get("/")
def root():
    return {
        "service": "Lagom Humanizer API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "endpoints": ["/api/health", "/api/styles", "/api/humanize", "/api/humanize/stream"],
    }


@app.get("/api/health")
def health():
    import torch
    loaded = holder.try_load()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_mb = (
        round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0
    )
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "vram_allocated_mb": vram_mb,
        "model_loaded": loaded,
        "model_type": holder.model_type,
        "model_path": holder.loaded_path,
        "note": "Ready for inference" if loaded else "Training in progress or adapter not found. Demo mode active.",
    }


@app.get("/api/styles")
def get_styles():
    return {
        "styles": [
            {
                "id": k,
                "label": k.title(),
                "description": v,
            }
            for k, v in CATEGORY_INSTRUCTIONS.items()
        ]
    }


def generate_demo_humanization(text: str, style: str) -> str:
    """Intelligent simulated humanization when local model training has not completed yet."""
    import re
    cleaned = text.strip()
    # Strip obvious AI transition markers
    cliches = [
        (r"\bFurthermore,\s*", ""),
        (r"\bMoreover,\s*", ""),
        (r"\bIn conclusion,\s*", "Ultimately, "),
        (r"\bIt is important to note that\s*", ""),
        (r"\bIt is crucial to remember that\s*", "Noticeably, "),
        (r"\bDelving into the realm of\s*", "Exploring "),
        (r"\bA testament to\s*", "proof of "),
    ]
    res = cleaned
    for pat, rep in cliches:
        res = re.sub(pat, rep, res, flags=re.IGNORECASE)

    if style == "email":
        return f"Hi there,\n\n{res}\n\nBest regards,\nAlex"
    elif style == "essay":
        return f"{res}\n\nThis perspective grounds the underlying thesis in actual observable dynamics."
    elif style == "academic":
        return f"Empirical examination indicates that {res.lower() if res else ''}"
    return res


@app.post("/api/humanize", response_model=HumanizeResponse)
def humanize(req: HumanizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    t0 = time.time()
    has_model = holder.try_load()

    if not has_model or holder.model is None or holder.tokenizer is None:
        # Graceful demo response while training runs
        output = generate_demo_humanization(req.text, req.style)
        elapsed = round(time.time() - t0, 3)
        return HumanizeResponse(
            humanized=output,
            style=req.style,
            tokens=len(output.split()),
            elapsed_seconds=elapsed,
            model_mode="simulated (model training pending)",
            word_count_original=len(req.text.split()),
            word_count_humanized=len(output.split()),
        )

    import torch
    tok = holder.tokenizer
    model = holder.model
    prompt = tok.apply_chat_template(
        build_messages(req.text, style=req.style),
        tokenize=False,
        add_generation_prompt=True,
    )
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=1.05,
            use_cache=True,
        )
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    elapsed = round(time.time() - t0, 3)

    return HumanizeResponse(
        humanized=gen,
        style=req.style,
        tokens=out.shape[1] - inp["input_ids"].shape[1],
        elapsed_seconds=elapsed,
        model_mode=f"local_{holder.model_type}",
        word_count_original=len(req.text.split()),
        word_count_humanized=len(gen.split()),
    )


@app.post("/api/humanize/stream")
async def humanize_stream(req: HumanizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    has_model = holder.try_load()

    async def stream_generator() -> AsyncGenerator[str, None]:
        if not has_model or holder.model is None or holder.tokenizer is None:
            # Simulate streaming chunks
            demo_text = generate_demo_humanization(req.text, req.style)
            words = demo_text.split(" ")
            for i, w in enumerate(words):
                chunk = w + (" " if i < len(words) - 1 else "")
                data = json.dumps({"token": chunk, "done": False, "mode": "simulated"})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.035)
            yield f"data: {json.dumps({'token': '', 'done': True, 'mode': 'simulated'})}\n\n"
            return

        from transformers import TextIteratorStreamer
        import torch

        tok = holder.tokenizer
        model = holder.model
        prompt = tok.apply_chat_template(
            build_messages(req.text, style=req.style),
            tokenize=False,
            add_generation_prompt=True,
        )
        inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inp,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=1.05,
            use_cache=True,
        )

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text:
                data = json.dumps({"token": new_text, "done": False, "mode": f"local_{holder.model_type}"})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.005)

        thread.join()
        yield f"data: {json.dumps({'token': '', 'done': True, 'mode': f'local_{holder.model_type}'})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("[API] Starting Lagom API Server at http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
