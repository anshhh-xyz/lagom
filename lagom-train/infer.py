#!/usr/bin/env python
"""Humanize text with the trained adapter (4-bit base + LoRA) or a merged model.
Features paragraph-aware chunking and 0.70-0.95 length ratio inference guard.

    python infer.py --text "AI text ..." --category essay
    python infer.py --file input.txt --category academic
    python infer.py --merged outputs/merged --file input.txt      # after merge.py
"""
import argparse
import json
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import build_messages, clean_humanized_output


def chunk_paragraphs(text: str, min_words: int = 150, max_words: int = 300) -> list[str]:
    """Split input on blank lines, merging short paragraphs to ~150-300 words."""
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]
    if not raw_paras:
        return [text] if text.strip() else []

    chunks = []
    curr = []
    curr_words = 0

    for p in raw_paras:
        pw = len(p.split())
        if curr and (curr_words + pw > max_words) and (curr_words >= min_words):
            chunks.append("\n\n".join(curr))
            curr = [p]
            curr_words = pw
        else:
            curr.append(p)
            curr_words += pw

    if curr:
        chunks.append("\n\n".join(curr))

    return chunks


def generate_single_pass(model, tok, prompt: str, args) -> str:
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def humanize_chunk_with_guard(
    model,
    tok,
    chunk: str,
    style: str,
    args,
    min_ratio: float = 0.70,
    max_ratio: float = 0.95,
    target_ratio: float = 0.85,
    max_attempts: int = 3,
) -> str:
    """Inference guard: ensures output/input words ratio is within 0.70-0.95, resampling up to 3 times."""
    in_words = len(chunk.split())
    if in_words == 0:
        return ""

    prompt = tok.apply_chat_template(
        build_messages(chunk, style=style),
        tokenize=False,
        add_generation_prompt=True,
    )

    candidates = []
    for attempt in range(max_attempts):
        gen = generate_single_pass(model, tok, prompt, args)
        cleaned = clean_humanized_output(gen, chunk)
        out_words = len(cleaned.split())
        ratio = out_words / max(in_words, 1)
        candidates.append((cleaned, ratio, abs(ratio - target_ratio)))

        if min_ratio <= ratio <= max_ratio:
            return cleaned

    # Return the candidate closest to target_ratio (0.85)
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/adapter")
    p.add_argument("--merged", default=None, help="Path to merged 16-bit model (skips adapter)")
    p.add_argument("--text")
    p.add_argument("--file")
    p.add_argument("--style", "--category", default="general", dest="style",
                   help="Target humanization style: general | essay | academic | email | document")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--min-ratio", type=float, default=0.70)
    p.add_argument("--max-ratio", type=float, default=0.95)
    p.add_argument("--target-ratio", type=float, default=0.85)
    p.add_argument("--resample-attempts", type=int, default=3)
    a = p.parse_args()

    if a.text:
        text = a.text
    elif a.file:
        with open(a.file, encoding="utf-8") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Enter/paste your AI text to humanize (press Enter, then Ctrl+Z on Windows / Ctrl+D on Linux and Enter):")
        text = sys.stdin.read()

    if not text.strip():
        raise SystemExit("No input text provided.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if a.merged:
        tok = AutoTokenizer.from_pretrained(a.merged)
        model = AutoModelForCausalLM.from_pretrained(a.merged, torch_dtype=dtype, device_map={"": 0})
    else:
        from peft import PeftModel
        run_file = os.path.join(a.adapter, "lagom_run.json")
        base = "Qwen/Qwen2.5-3B-Instruct"
        if os.path.exists(run_file):
            with open(run_file) as f:
                base = json.load(f).get("base_model", base)

        tok = AutoTokenizer.from_pretrained(a.adapter)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0}, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, a.adapter)

    model.eval()

    # Paragraph-aware chunking
    chunks = chunk_paragraphs(text, min_words=150, max_words=300)
    humanized_chunks = []

    for i, chunk in enumerate(chunks):
        out_chunk = humanize_chunk_with_guard(
            model,
            tok,
            chunk,
            style=a.style,
            args=a,
            min_ratio=a.min_ratio,
            max_ratio=a.max_ratio,
            target_ratio=a.target_ratio,
            max_attempts=a.resample_attempts,
        )
        humanized_chunks.append(out_chunk)

    final_output = "\n\n".join(humanized_chunks)
    print(final_output)


if __name__ == "__main__":
    main()
