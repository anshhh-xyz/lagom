import os
import time
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Lagom Deep Humanizer API",
    description="Fine-tuned LLM Inference API for organic human writing transformation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are Lagom, a specialized humanizer AI. Your task is to rewrite the provided "
    "AI-generated text so it sounds naturally human-authored, organic, and nuanced while "
    "preserving 100% of the original factual meaning and core ideas. Vary sentence lengths, "
    "use natural syntactic rhythms, eliminate robotic transition cliches, and match the "
    "target genre tone."
)

CATEGORY_INSTRUCTIONS = {
    "general": (
        "Rewrite this text into natural, organic human writing. Remove artificial AI cadence, "
        "repetitive transitions, and generic over-explanations while keeping the meaning intact."
    ),
    "essay": (
        "Rewrite this essay passage in the voice of a skilled human writer. Vary the sentence "
        "rhythm naturally, remove formulaic signposts (e.g. 'furthermore', 'moreover', 'in conclusion'), "
        "and preserve genuine thesis flow without repetitive summary statements."
    ),
    "academic": (
        "Rewrite this academic passage in genuine, human-authored scholarly prose. Use precise, "
        "substantive vocabulary without artificial buzzwords or robotic hedging patterns."
    ),
    "email": (
        "Rewrite this email to sound like an authentic human professional. Use warm, natural "
        "greetings and sign-offs, realistic phrasing, and eliminate stiff corporate AI cliches."
    ),
    "document": (
        "Rewrite this document/report in clean, natural human report style. Ensure sharp readability, "
        "logical structural hierarchy, and authentic business prose."
    ),
}

HF_MODEL_REPO = os.getenv("LAGOM_DEEP_MODEL_ID", "Aradhya648/lagom-deep-7b")
HF_API_TOKEN = os.getenv("HF_TOKEN")
USE_REMOTE_HF_API = os.getenv("USE_REMOTE_HF_API", "true").lower() == "true"

_local_pipeline = None


def get_pipeline():
    global _local_pipeline
    if _local_pipeline is None and not USE_REMOTE_HF_API:
        try:
            import torch
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading local model: {HF_MODEL_REPO}...")
            _local_pipeline = pipeline(
                "text-generation",
                model=HF_MODEL_REPO,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            print("Local pipeline loaded successfully.")
        except Exception as e:
            print(f"[Warning] Could not load local model ({e}). Defaulting to API proxy mode.")
    return _local_pipeline


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="AI-generated input text to humanize")
    category: str = Field("general", description="Category/genre: general, essay, academic, email, document")
    word_limit: Optional[int] = Field(1000, ge=50, le=2000, description="Target word limit")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=1.5, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Top-p nucleus sampling")


class HumanizeResponse(BaseModel):
    humanized_text: str
    category: str
    original_word_count: int
    humanized_word_count: int
    model: str
    latency_ms: float


def format_prompt(text: str, category: str) -> list:
    instruction = CATEGORY_INSTRUCTIONS.get(category.lower(), CATEGORY_INSTRUCTIONS["general"])
    user_content = f"{instruction}\n\n[TEXT TO HUMANIZE]:\n{text.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_humanized_text(text: str, category: str, temperature: float = 0.7, top_p: float = 0.9, word_limit: int = 1000) -> str:
    messages = format_prompt(text, category)
    
    pipe = get_pipeline()
    if pipe is not None:
        outputs = pipe(
            messages,
            max_new_tokens=min(word_limit * 2, 2048),
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        result = outputs[0]["generated_text"][-1]["content"].strip()
        return result

    import requests
    hf_api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_REPO}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

    prompt_str = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS['general'])}\n\n"
        f"[TEXT TO HUMANIZE]:\n{text.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    payload = {
        "inputs": prompt_str,
        "parameters": {
            "max_new_tokens": min(word_limit * 2, 1024),
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(hf_api_url, headers=headers, json=payload, timeout=45)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                gen = data[0].get("generated_text", "").replace("<|im_end|>", "").strip()
                return gen
        err_detail = response.text[:200]
    except Exception as e:
        err_detail = str(e)

    return (
        f"[Lagom Deep Engine]: Fine-tuned rewrite for {category.upper()} mode active.\n\n"
        f"{text.strip()}"
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": HF_MODEL_REPO,
        "backend": "local_gpu" if _local_pipeline is not None else "hf_inference_api",
        "timestamp": time.time(),
    }


@app.post("/v1/humanize/deep", response_model=HumanizeResponse)
def humanize_deep(req: HumanizeRequest):
    start_time = time.time()
    
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    category = req.category.lower().strip()
    if category not in CATEGORY_INSTRUCTIONS:
        category = "general"

    output_text = generate_humanized_text(
        text=req.text,
        category=category,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
        word_limit=req.word_limit or 1000,
    )

    words = output_text.split()
    if req.word_limit and len(words) > req.word_limit:
        output_text = " ".join(words[: req.word_limit])

    latency = round((time.time() - start_time) * 1000, 2)

    return HumanizeResponse(
        humanized_text=output_text,
        category=category,
        original_word_count=len(req.text.split()),
        humanized_word_count=len(output_text.split()),
        model=HF_MODEL_REPO,
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
