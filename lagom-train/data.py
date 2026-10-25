"""CSV -> cleaned pairs -> tokenized examples with prompt tokens masked (loss on the human rewrite only)."""
import random
from collections import Counter

import pandas as pd

from common import CATEGORY_INSTRUCTIONS, build_messages

AI_COL, HUMAN_COL, CAT_COL = "AI Text", "Human Text", "category"
TARGET_WORDS_COL = "target_words"
AI_WORDS_COL = "ai_words"
HUMAN_WORDS_COL = "human_words"


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


def _compute_target_words(row: dict) -> int:
    """Derive target_words for a row.

    Priority order:
      1. Explicit target_words column (from train_filtered.csv).
      2. Compute from human_words column: round(human_words, -1).
      3. Compute from human text length: round(len(human.split()), -1).
    """
    tw = row.get(TARGET_WORDS_COL)
    if tw is not None and not (isinstance(tw, float) and tw != tw):  # not NaN
        return int(round(float(tw), -1))

    hw = row.get(HUMAN_WORDS_COL)
    if hw is not None and not (isinstance(hw, float) and hw != hw):
        return int(round(float(hw), -1))

    # Fallback: count words in the already-cleaned human text stored in row
    human_text = row.get("human", "")
    if human_text:
        return int(round(len(human_text.split()), -1))

    return None  # caller may handle None as "use default"


def load_pairs(csv_path: str, min_words: int = 25):
    # Read optional columns gracefully
    required = [AI_COL, HUMAN_COL, CAT_COL]
    optional = [TARGET_WORDS_COL, AI_WORDS_COL, HUMAN_WORDS_COL]

    # Load required cols + whatever optional ones exist
    all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = required + [c for c in optional if c in all_cols]

    df = pd.read_csv(csv_path, usecols=usecols)

    pairs, seen, dropped = [], set(), Counter()
    for row in df.to_dict("records"):
        ai = clean_text(row.get(AI_COL, ""))
        hu = clean_text(row.get(HUMAN_COL, ""))
        cat = str(row.get(CAT_COL, "general")).strip().lower()

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
        if cat not in CATEGORY_INSTRUCTIONS:
            cat = "general"

        # Assemble pair dict including human text so _compute_target_words can use it
        pair = {"ai": ai, "human": hu, "category": cat}

        # Attach all optional fields to the row dict for target_words lookup
        full_row = dict(row)
        full_row["human"] = hu  # cleaned human text for word count fallback
        pair["target_words"] = _compute_target_words(full_row)

        pairs.append(pair)

    return pairs, dropped


def split_pairs(pairs, val_size: int, seed: int = 42):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    return pairs[val_size:], pairs[:val_size]


def encode(pairs, tokenizer, max_len: int, eos_text: str = "<|im_end|>\n", chunk: int = 256):
    """Returns (rows, stats). Examples longer than max_len are DROPPED, never truncated
    (a truncated target would teach the model to stop mid-sentence / never emit EOS).

    Reads pair["target_words"] and passes it to build_messages so training and inference
    prompts are identical in format (both include "Target length: about N words").
    If target_words is None, build_messages falls back to round(0.85 * input_words, -1).
    """
    rows, lengths, too_long = [], [], 0
    for i in range(0, len(pairs), chunk):
        batch = pairs[i:i + chunk]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(p["ai"], p["category"], target_words=p.get("target_words")),
                tokenize=False,
                add_generation_prompt=True,
            )
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
    stats = {
        "kept": len(rows),
        "dropped_too_long": too_long,
        "mean_tokens": (sum(lengths) / max(len(lengths), 1)),
        "p50": q(0.5),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": lengths[-1] if lengths else 0,
    }
    return rows, stats
