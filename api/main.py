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

from common import CATEGORY_INSTRUCTIONS, build_messages  # noqa: E402

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
        self.is_loading = False
        self.lock = threading.Lock()

    def find_model_dir(self):
        """Scans for valid trained adapter or merged model checkpoints."""
        candidates = []
        if os.environ.get("LAGOM_MODEL_PATH"):
            candidates.append(os.environ.get("LAGOM_MODEL_PATH"))
        if os.environ.get("LAGOM_ADAPTER_PATH"):
            candidates.append(os.environ.get("LAGOM_ADAPTER_PATH"))
        if os.environ.get("LAGOM_MERGED_PATH"):
            candidates.append(os.environ.get("LAGOM_MERGED_PATH"))

        candidates.extend([
            os.path.join(train_pkg_dir, "outputs", "merged"),
            os.path.join(script_dir, "..", "outputs", "merged"),
            os.path.join(train_pkg_dir, "outputs", "adapter"),
            os.path.join(script_dir, "..", "outputs", "adapter"),
            os.path.join(train_pkg_dir, "outputs_smoke", "adapter"),
            os.path.join(script_dir, "..", "outputs_smoke", "adapter"),
        ])

        for path in candidates:
            if not path or not os.path.exists(path):
                continue

            # Check for merged model
            if (
                os.path.exists(os.path.join(path, "model.safetensors"))
                or os.path.exists(os.path.join(path, "pytorch_model.bin"))
                or (os.path.exists(os.path.join(path, "config.json")) and not os.path.exists(os.path.join(path, "adapter_config.json")))
            ):
                return "merged", os.path.abspath(path)

            # Check for LoRA adapter
            if (
                os.path.exists(os.path.join(path, "adapter_config.json"))
                or os.path.exists(os.path.join(path, "lagom_run.json"))
            ):
                return "adapter", os.path.abspath(path)

        return None, None

    def try_load(self):
        with self.lock:
            if self.model is not None:
                return True

            kind, model_dir = self.find_model_dir()
            if not kind or not model_dir:
                return False

            self.is_loading = True
            try:
                import torch
                dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                device_target = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

                if kind == "merged":
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    print(f"[API] Found merged weights at: {model_dir}. Loading...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=dtype,
                        device_map=device_target,
                    )
                    self.model.eval()
                    self.model_type = "merged"
                    self.loaded_path = model_dir
                    print(f"[API] Merged model successfully loaded on {device_target}!")
                    return True

                if kind == "adapter":
                    from peft import PeftModel
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                    base_id = "Qwen/Qwen2.5-3B-Instruct"
                    run_meta_path = os.path.join(model_dir, "lagom_run.json")
                    adapter_cfg_path = os.path.join(model_dir, "adapter_config.json")

                    if os.path.exists(run_meta_path):
                        try:
                            with open(run_meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                if "base_model" in meta:
                                    base_id = meta["base_model"]
                        except Exception:
                            pass
                    elif os.path.exists(adapter_cfg_path):
                        try:
                            with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                                if "base_model_name_or_path" in cfg:
                                    base_id = cfg["base_model_name_or_path"]
                        except Exception:
                            pass

                    print(f"[API] Found adapter at: {model_dir}. Loading base {base_id} (4-bit NF4) + LoRA...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "right"

                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype,
                    ) if torch.cuda.is_available() else None

                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_id,
                        quantization_config=bnb,
                        device_map=device_target,
                        torch_dtype=dtype,
                    )
                    self.model = PeftModel.from_pretrained(base_model, model_dir)
                    self.model.eval()
                    self.model_type = "adapter"
                    self.loaded_path = model_dir
                    print(f"[API] Adapter model successfully loaded and active!")
                    return True

            except Exception as e:
                print(f"[API] Failed to load model from {model_dir}: {e}")
                self.model = None
                self.tokenizer = None
                self.model_type = "unloaded"
                return False
            finally:
                self.is_loading = False

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
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    vram_mb = (
        round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        if gpu_available
        else 0
    )
    return {
        "status": "ready" if loaded else "training_pending",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_allocated_mb": vram_mb,
        "model_loaded": loaded,
        "model_type": holder.model_type,
        "model_path": holder.loaded_path,
        "is_loading": holder.is_loading,
        "message": (
            "Model is loaded and ready for inference."
            if loaded
            else "Training pending. Model weights not found in outputs/adapter or outputs/merged. Once training finishes, the model will be loaded automatically."
        ),
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


@app.post("/api/humanize", response_model=HumanizeResponse)
def humanize(req: HumanizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    has_model = holder.try_load()
    if not has_model or holder.model is None or holder.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not ready. Training has not been completed yet (no weights found in "
                "outputs/adapter or outputs/merged). Please run the training script in lagom-train first."
            ),
        )

    import torch
    t0 = time.time()
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
    if not has_model or holder.model is None or holder.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not ready. Training has not been completed yet (no weights found in "
                "outputs/adapter or outputs/merged). Please run the training script in lagom-train first."
            ),
        )

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

    async def stream_generator() -> AsyncGenerator[str, None]:
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

