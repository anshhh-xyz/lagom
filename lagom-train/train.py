#!/usr/bin/env python
"""Lagom Deep Mode - QLoRA fine-tuning (AI text -> human rewrite).

    python train.py --preset local8gb          # RTX 5050 8GB  (Qwen2.5-3B)
    python train.py --preset kaggle            # Kaggle T4     (Qwen2.5-7B)
    python train.py --preset local8gb --max-samples 300 --epochs 1   # smoke test first!

Backend: Unsloth if importable (faster, less VRAM), otherwise plain transformers+peft+bitsandbytes.
"""
import argparse
import inspect
import json
import math
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from common import PRESETS, hf_base_name, build_messages  # noqa: E402
from data import load_pairs, split_pairs, encode  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), default="local8gb")

    # Auto-resolve dataset paths if present in ../Data/improved_data/ or relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_candidates = [
        os.path.join(script_dir, "..", "Data", "improved_data", "train_normalized.csv"),
        os.path.join(script_dir, "Data", "improved_data", "train_normalized.csv"),
        os.path.join(script_dir, "data", "train_normalized.csv"),
        "Data/improved_data/train_normalized.csv",
    ]
    val_candidates = [
        os.path.join(script_dir, "..", "Data", "improved_data", "validation_normalized.csv"),
        os.path.join(script_dir, "Data", "improved_data", "validation_normalized.csv"),
        os.path.join(script_dir, "data", "validation_normalized.csv"),
        "Data/improved_data/validation_normalized.csv",
    ]
    default_train = next((c for c in train_candidates if os.path.exists(c)), train_candidates[0])
    default_val = next((c for c in val_candidates if os.path.exists(c)), None)

    p.add_argument("--data", default=default_train, help="Path to training CSV")
    p.add_argument("--val-csv", default=default_val, help="Optional separate validation CSV. Auto-detected if present.")
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--model", default=None, help="Override preset base model")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None, help="Use only N training rows (smoke test)")
    p.add_argument("--backend", choices=["auto", "unsloth", "hf"], default="auto")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output-dir")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.model = a.model or PRESETS[a.preset]["model"]
    return a


def main():
    args = parse_args()

    # Unsloth must be imported BEFORE transformers / peft.
    use_unsloth = False
    if args.backend in ("auto", "unsloth"):
        try:
            from unsloth import FastLanguageModel  # noqa: F401
            use_unsloth = True
        except Exception as e:  # ImportError, or GPU/triton problems on Windows
            if args.backend == "unsloth":
                raise
            print(f"[info] Unsloth unavailable ({type(e).__name__}: {str(e)[:120]}) -> using plain HF + PEFT backend")

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint
    from datasets import Dataset

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. On RTX 50-series install PyTorch built for CUDA 12.8+ (see README).")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB | backend={'unsloth' if use_unsloth else 'hf+peft'}")
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    # ---------------- model ----------------
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model, max_seq_length=args.max_seq_len, dtype=None, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth", random_state=args.seed)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        base = hf_base_name(args.model)
        tokenizer = AutoTokenizer.from_pretrained(base)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map={"": 0},
                                                     torch_dtype=dtype, attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.print_trainable_parameters()

    # ---------------- data ----------------
    pairs, dropped = load_pairs(args.data)
    print(f"Loaded {len(pairs):,} usable pairs from {args.data}  (dropped: {dict(dropped)})")
    if args.val_csv:
        train_pairs = pairs
        val_pairs, _ = load_pairs(args.val_csv)
        val_pairs = val_pairs[: args.val_size]
    else:
        train_pairs, val_pairs = split_pairs(pairs, args.val_size, args.seed)
    if args.max_samples:
        train_pairs = train_pairs[: args.max_samples]

    train_rows, st = encode(train_pairs, tokenizer, args.max_seq_len)
    val_rows, _ = encode(val_pairs, tokenizer, args.max_seq_len)
    print(f"Train examples: {st['kept']:,} (dropped {st['dropped_too_long']} longer than {args.max_seq_len} tokens) | val: {len(val_rows)}")
    print(f"Token lengths: mean={st['mean_tokens']:.0f} p50={st['p50']} p95={st['p95']} p99={st['p99']} max={st['max']}")
    if st["dropped_too_long"] > 0.03 * len(train_pairs):
        print("[warn] >3% of examples exceed --max-seq-len; consider raising it if VRAM allows.")
    train_ds, val_ds = Dataset.from_list(train_rows), Dataset.from_list(val_rows)

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        ids = torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch])
        att = torch.tensor([b["attention_mask"] + [0] * (m - len(b["attention_mask"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    # ---------------- trainer ----------------
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    ta = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)), weight_decay=0.01, max_grad_norm=1.0,
        optim="paged_adamw_8bit", bf16=bf16, fp16=not bf16,
        logging_steps=10, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_steps=args.save_steps, prediction_loss_only=True,
        group_by_length=True, report_to="none", seed=args.seed,
        gradient_checkpointing=not use_unsloth,
        remove_unused_columns=False,
    )
    if not use_unsloth:
        ta["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    ta[key] = "steps"
    trainer = Trainer(model=model, args=TrainingArguments(**ta), train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate)

    ckpt = get_last_checkpoint(args.output_dir) if (args.resume and os.path.isdir(args.output_dir)) else None
    print(f"\nStarting training: {total_steps} optimizer steps (effective batch {args.batch_size * args.grad_accum}), resume={ckpt}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=ckpt)
    print(f"Training finished in {(time.time() - t0) / 60:.1f} min")

    # ---------------- save + quick sample ----------------
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "lagom_run.json"), "w") as f:
        json.dump({"base_model": args.model, "max_seq_len": args.max_seq_len, "epochs": args.epochs,
                   "lr": args.lr, "lora_r": args.lora_r, "train_examples": len(train_ds)}, f, indent=2)
    print(f"Adapter saved -> {adapter_dir}")

    if val_pairs:
        if use_unsloth:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        else:
            model.config.use_cache = True
        model.eval()
        p = val_pairs[0]
        prompt = tokenizer.apply_chat_template(build_messages(p["ai"], p["category"]), tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=700, do_sample=True, temperature=0.7, top_p=0.9,
                                 repetition_penalty=1.05, use_cache=True)
        gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print("\n" + "=" * 70 + f"\n[{p['category']}] AI INPUT:\n{p['ai'][:700]}\n\n--- MODEL OUTPUT:\n{gen[:1200]}\n\n--- REFERENCE HUMAN:\n{p['human'][:700]}\n" + "=" * 70)


if __name__ == "__main__":
    main()
