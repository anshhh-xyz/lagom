"""CSV -> cleaned pairs -> tokenized examples with prompt tokens masked (loss on the human rewrite only)."""
import random
from collections import Counter

import pandas as pd

from common import CATEGORY_INSTRUCTIONS, build_messages

AI_COL, HUMAN_COL, CAT_COL = "AI Text", "Human Text", "category"


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    lines = [ln.strip() for ln in text.strip().split("\n")]
    out, prev_blank = [], False
    for ln in lines:
        if not ln:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(ln)
            prev_blank = False
    return "\n".join(out).strip()


def load_pairs(csv_path: str, min_words: int = 25):
    df = pd.read_csv(csv_path, usecols=[AI_COL, HUMAN_COL, CAT_COL])
    pairs, seen, dropped = [], set(), Counter()
    for ai, hu, cat in zip(df[AI_COL], df[HUMAN_COL], df[CAT_COL]):
        ai, hu = clean_text(ai), clean_text(hu)
        if not ai or not hu:
            dropped["empty"] += 1
            continue
        if len(ai.split()) < min_words or len(hu.split()) < min_words:
            dropped["too_short"] += 1
            continue
        if ai in seen:
            dropped["duplicate_ai_text"] += 1
            continue
        seen.add(ai)
        cat = str(cat).strip().lower()
        if cat not in CATEGORY_INSTRUCTIONS:
            cat = "general"
        pairs.append({"ai": ai, "human": hu, "category": cat})
    return pairs, dropped


def split_pairs(pairs, val_size: int, seed: int = 42):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    return pairs[val_size:], pairs[:val_size]


def encode(pairs, tokenizer, max_len: int, eos_text: str = "<|im_end|>\n", chunk: int = 256):
    """Returns (rows, stats). Examples longer than max_len are DROPPED, never truncated
    (a truncated target would teach the model to stop mid-sentence / never emit EOS)."""
    rows, lengths, too_long = [], [], 0
    for i in range(0, len(pairs), chunk):
        batch = pairs[i:i + chunk]
        prompts = [
            tokenizer.apply_chat_template(build_messages(p["ai"], p["category"]),
                                          tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        comps = [p["human"] + eos_text for p in batch]
        p_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(comps, add_special_tokens=False)["input_ids"]
        for pi, ci in zip(p_ids, c_ids):
            n = len(pi) + len(ci)
            lengths.append(n)
            if n > max_len:
                too_long += 1
                continue
            rows.append({
                "input_ids": pi + ci,
                "attention_mask": [1] * n,
                "labels": [-100] * len(pi) + ci,
            })
    lengths.sort()
    q = lambda f: lengths[min(len(lengths) - 1, int(len(lengths) * f))] if lengths else 0
    stats = {"kept": len(rows), "dropped_too_long": too_long, "mean_tokens": (sum(lengths) / max(len(lengths), 1)),
             "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": lengths[-1] if lengths else 0}
    return rows, stats
