import time
from config import AIIFY_PROMPTS, KEY_PAIRS
from clients import (
    get_gemini_client,
    get_groq_client,
)

GEMINI_MODELS = [
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]


def aiify_gemini(text, category, pair_idx=0):
    client = get_gemini_client(pair_idx)
    prompt = AIIFY_PROMPTS[category].format(text=text)
    
    for m in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=prompt,
            )
            if resp.text:
                return resp.text.strip(), m
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "NOT_FOUND" in err_msg or "404" in err_msg:
                continue
            raise e
    raise RuntimeError(f"Gemini quota exhausted / unavailable on {KEY_PAIRS[pair_idx]['name']}.")


def aiify_groq(text, category, pair_idx=0):
    client = get_groq_client(pair_idx)
    prompt = AIIFY_PROMPTS[category].format(text=text)
    
    for model_name in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            content = resp.choices[0].message.content.strip()
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            if content:
                return content, model_name
        except Exception:
            continue
    raise RuntimeError(f"Groq quota exhausted / unavailable on {KEY_PAIRS[pair_idx]['name']}.")


def aiify_call(text, category, model="groq", pair_idx=0):
    providers = [model, "gemini" if model == "groq" else "groq"]
    
    for prov in providers:
        try:
            if prov == "gemini":
                return aiify_gemini(text, category, pair_idx=pair_idx)
            elif prov == "groq":
                return aiify_groq(text, category, pair_idx=pair_idx)
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "exhausted" in err_str.lower() or "unavailable" in err_str.lower():
                continue
            for attempt in range(2):
                try:
                    time.sleep(1)
                    if prov == "gemini":
                        return aiify_gemini(text, category, pair_idx=pair_idx)
                    elif prov == "groq":
                        return aiify_groq(text, category, pair_idx=pair_idx)
                except Exception:
                    pass

    raise RuntimeError(f"Both Gemini and Groq exhausted/failed on {KEY_PAIRS[pair_idx]['name']}.")
