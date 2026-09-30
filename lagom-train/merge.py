#!/usr/bin/env python
"""Merge the LoRA adapter into full 16-bit weights (for serving via server/app.py, vLLM, HF Hub, GGUF).
Loads the base model in fp16 on CPU, so it needs ~7 GB RAM for 3B / ~16 GB for 7B, not VRAM.

    python merge.py --adapter outputs/adapter --out outputs/merged
"""
import argparse
import json
import os

from common import hf_base_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/adapter")
    p.add_argument("--out", default="outputs/merged")
    a = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(a.adapter, "lagom_run.json")) as f:
        base = hf_base_name(json.load(f)["base_model"])
    print(f"Merging {a.adapter} into {base} ...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print(f"Merged model saved -> {a.out}")


if __name__ == "__main__":
    main()
