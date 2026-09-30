#!/usr/bin/env python
"""Humanize text with the trained adapter (4-bit base + LoRA) or a merged model.

    python infer.py --text "AI text ..." --category essay
    python infer.py --file input.txt --category academic
    python infer.py --merged outputs/merged --file input.txt      # after merge.py
"""
import argparse
import json
import os
import sys

from common import build_messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/adapter")
    p.add_argument("--merged", default=None, help="Path to merged 16-bit model (skips adapter)")
    p.add_argument("--text"); p.add_argument("--file")
    p.add_argument("--style", "--category", default="general", dest="style",
                   help="Target humanization style: general | essay | academic | email | document")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-new-tokens", type=int, default=900)
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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if a.merged:
        tok = AutoTokenizer.from_pretrained(a.merged)
        model = AutoModelForCausalLM.from_pretrained(a.merged, torch_dtype=dtype, device_map={"": 0})
    else:
        from peft import PeftModel
        with open(os.path.join(a.adapter, "lagom_run.json")) as f:
            base = json.load(f)["base_model"]
        tok = AutoTokenizer.from_pretrained(a.adapter)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0}, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    prompt = tok.apply_chat_template(build_messages(text, style=a.style), tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temperature,
                             top_p=a.top_p, repetition_penalty=a.repetition_penalty, use_cache=True)
    print(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
