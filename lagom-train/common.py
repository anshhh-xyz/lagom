"""Shared prompts + presets. Prompts are identical to deep_mode/server/app.py and
deep_mode/prompts/deep_prompts.ts, so the trained model plugs straight into your server."""

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


def build_messages(text: str, category: str = "general", style: str | None = None):
    """Chat messages WITHOUT the assistant turn (used for training prompts and inference).
    Format:
        Target Style: {style}
        Guidelines: {instruction}

        [TEXT TO HUMANIZE]:
        {text}
    """
    chosen_style = (style or category or "general").strip().lower()
    instruction = CATEGORY_INSTRUCTIONS.get(chosen_style, CATEGORY_INSTRUCTIONS["general"])
    user_content = (
        f"Target Style: {chosen_style.title()}\n"
        f"Guidelines: {instruction}\n\n"
        f"[TEXT TO HUMANIZE]:\n{text.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --- presets -------------------------------------------------------------------------
# local8gb : RTX 5050 8GB. 7B QLoRA does not leave enough headroom for ~1.5k-token
#            sequences on 8GB, so the preset uses a 3B model.
# kaggle   : T4 16GB (single GPU is used). 7B fits at bs=2.
PRESETS = {
    "local8gb": dict(
        model="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_len=1536, batch_size=1, grad_accum=16, lora_r=32, lora_alpha=32,
    ),
    "kaggle": dict(
        model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_len=1536, batch_size=2, grad_accum=8, lora_r=32, lora_alpha=32,
    ),
}


def hf_base_name(name: str) -> str:
    """'unsloth/Qwen2.5-3B-Instruct-bnb-4bit' -> 'Qwen/Qwen2.5-3B-Instruct'."""
    n = name
    if n.startswith("unsloth/"):
        n = n[len("unsloth/"):]
        if n.endswith("-bnb-4bit"):
            n = n[: -len("-bnb-4bit")]
        if n.startswith("Qwen"):
            n = "Qwen/" + n
    return n
