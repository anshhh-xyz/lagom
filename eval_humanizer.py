#!/usr/bin/env python
"""
Evaluation script for Lagom Humanizer (eval_humanizer.py).
Evaluates model outputs against held-out input texts from a JSONL file.
Computes and reports:
- Length ratio (output_words / input_words) & distribution across bins (<0.70, 0.70-0.80, 0.80-0.90, 0.90-0.95, >0.95)
- Burstiness (sentence-length standard deviation in words)
- Contraction count per 100 words
- AI-cliché density per 1000 words
- Copy rate vs input (fraction of input sentences whose max cosine sim > 0.90 in output)
- Sentence count ratio (output_sentences / input_sentences)
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure common is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(script_dir, "lagom-train")
if os.path.exists(train_dir) and train_dir not in sys.path:
    sys.path.append(train_dir)

try:
    from common import build_messages, clean_humanized_output
except ImportError:
    from lagom_train.common import build_messages, clean_humanized_output

try:
    import pysbd
    SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except ImportError:
    SEGMENTER = None

CLICHE_WORDS = [
    "furthermore", "moreover", "consequently", "in conclusion",
    "landscape", "tapestry", "multifaceted", "paramount",
    "delve", "testament", "pivotal", "foster", "harness",
    "navigate", "realm", "crucial", "profound", "underscore",
    "symbiotic", "holistic", "unprecedented", "paradigm",
    "catalyst", "transformative", "intricate", "seamless",
    "vibrant", "ecosystem", "synergy", "steadfast",
    "cornerstone", "ultimately"
]

CLICHE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in CLICHE_WORDS) + r")\b",
    re.IGNORECASE
)
CONTRACTION_PATTERN = re.compile(
    r"\b([a-zA-Z]+)('re|'ve|'d|'ll|'t|'s|n't)\b",
    re.IGNORECASE
)


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    if SEGMENTER is not None:
        try:
            return [s.strip() for s in SEGMENTER.segment(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text.strip()) if s.strip()]


def chunk_paragraphs(text: str, min_words: int = 150, max_words: int = 300) -> list[str]:
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


def humanize_chunk(model, tok, chunk: str, style: str, args, min_ratio=0.70, max_ratio=0.95, target_ratio=0.85, max_attempts=3) -> str:
    in_words = len(chunk.split())
    if in_words == 0:
        return ""
    prompt = tok.apply_chat_template(build_messages(chunk, style=style), tokenize=False, add_generation_prompt=True)
    candidates = []
    for _ in range(max_attempts):
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
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cleaned = clean_humanized_output(gen, chunk)
        out_words = len(cleaned.split())
        ratio = out_words / max(in_words, 1)
        candidates.append((cleaned, ratio, abs(ratio - target_ratio)))
        if min_ratio <= ratio <= max_ratio:
            return cleaned
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0]


def evaluate(args):
    # 1. Load test data
    inputs = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                inputs.append(item)

    print(f"Loaded {len(inputs)} evaluation examples from {args.input}")

    # 2. Load model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.merged:
        tok = AutoTokenizer.from_pretrained(args.merged)
        model = AutoModelForCausalLM.from_pretrained(args.merged, torch_dtype=dtype, device_map={"": 0})
    else:
        from peft import PeftModel
        base = "Qwen/Qwen2.5-3B-Instruct"
        run_file = os.path.join(args.adapter, "lagom_run.json")
        if os.path.exists(run_file):
            with open(run_file) as f:
                base = json.load(f).get("base_model", base)
        tok = AutoTokenizer.from_pretrained(args.adapter)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0}, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 3. Load embedding model for copy_rate evaluation
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    results = []
    print("\nRunning inference & metrics...")
    for i, item in enumerate(inputs):
        inp_text = item.get("text") or item.get("AI Text") or item.get("input", "")
        cat = item.get("category", args.style)
        if not inp_text.strip():
            continue

        # Humanize
        chunks = chunk_paragraphs(inp_text)
        out_chunks = [humanize_chunk(model, tok, c, cat, args) for c in chunks]
        out_text = "\n\n".join(out_chunks)

        inp_words = len(inp_text.split())
        out_words = len(out_text.split())
        ratio = round(out_words / max(inp_words, 1), 3)

        # Style metrics
        out_sents = split_sentences(out_text)
        inp_sents = split_sentences(inp_text)

        burst = float(np.std([len(s.split()) for s in out_sents])) if len(out_sents) >= 2 else 0.0
        contr = round(len(CONTRACTION_PATTERN.findall(out_text)) / max(out_words, 1) * 100.0, 2)
        cliche_d = round(len(CLICHE_PATTERN.findall(out_text)) / max(out_words, 1) * 1000.0, 2)
        sc_ratio = round(len(out_sents) / max(len(inp_sents), 1), 3)

        # Copy rate: share of input sentences with max cosine similarity > 0.90 in output
        if inp_sents and out_sents:
            i_embs = emb_model.encode(inp_sents, normalize_embeddings=True)
            o_embs = emb_model.encode(out_sents, normalize_embeddings=True)
            sim_mat = np.dot(i_embs, o_embs.T)
            best_sims = np.max(sim_mat, axis=1)
            copy_rate = round(float(np.mean(best_sims > 0.90)), 3)
            mean_cov = round(float(np.mean(best_sims)), 3)
        else:
            copy_rate = 0.0
            mean_cov = 0.0

        results.append({
            "category": cat,
            "input_words": inp_words,
            "output_words": out_words,
            "ratio": ratio,
            "burstiness": burst,
            "contraction_rate": contr,
            "cliche_density": cliche_d,
            "copy_rate": copy_rate,
            "mean_cov": mean_cov,
            "sent_count_ratio": sc_ratio,
        })
        print(f"[{i+1}/{len(inputs)}] ratio={ratio:.2f} | copy_rate={copy_rate:.2f} | cliches={cliche_d:.1f} | burst={burst:.1f}")

    df = pd.DataFrame(results)

    # 4. Print Summary Report & Ratio Distribution
    print("\n" + "=" * 80)
    print("                     LAGOM HUMANIZER EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Evaluated: {len(df)} texts\n")

    def get_bin(r):
        if r < 0.70: return "<0.70"
        elif 0.70 <= r < 0.80: return "0.70-0.80"
        elif 0.80 <= r < 0.90: return "0.80-0.90"
        elif 0.90 <= r <= 0.95: return "0.90-0.95"
        else: return ">0.95"

    df["ratio_bin"] = df["ratio"].apply(get_bin)
    ratio_dist = df["ratio_bin"].value_counts(normalize=True) * 100.0

    print("LENGTH RATIO DISTRIBUTION:")
    for b in ["<0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", ">0.95"]:
        print(f"  {b:>10}: {ratio_dist.get(b, 0.0):.1f}%")

    print("\nOVERALL METRIC MEANS & MEDIANS:")
    for m in ["ratio", "burstiness", "contraction_rate", "cliche_density", "copy_rate", "mean_cov", "sent_count_ratio"]:
        print(f"  {m:<20}: Mean = {df[m].mean():.2f} | Median = {df[m].median():.2f}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDetailed metrics saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to test.jsonl")
    p.add_argument("--output", default="eval_results.csv", help="Output CSV path")
    p.add_argument("--adapter", default="lagom-train/outputs/adapter")
    p.add_argument("--merged", default=None)
    p.add_argument("--style", default="general")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--max-new-tokens", type=int, default=900)
    evaluate(p.parse_args())
