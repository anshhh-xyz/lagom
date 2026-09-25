#!/usr/bin/env python3
"""Lagom preprocessing step 1: deterministic cleaning + style/quality signals.

This file deliberately does NOT run SBERT, TF-IDF, or NLI. Those are separate
steps so each signal can be debugged independently.

Default input resolution:
  1. data/improved_data/all_scored.csv if present
  2. data/improved_data/base_clean.csv if present
  3. the only CSV in data/Collection

Default output directory: data/improved_data
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import choose_input_csv, clean_text, detect_pair_columns, ensure_dirs, resolve_data_dir, word_count

MILD_REPEAT = 3
STRONG_REPEAT = 4
EXTREME_REPEAT = 6

INFORMAL_WORDS = {
    "lol","lmao","lmfao","rofl","omg","btw","idk","imo","imho","tbh","ngl","ikr",
    "bruh","bro","dude","yep","yup","nah","nope","gonna","wanna","gotta","kinda","sorta",
    "ain't","lemme","gimme","cuz","coz","tho","thx","pls","plz","haha","hehe","hahaha",
}
INTERNET_SLANG = {"fr","rn","sus","af","ong","istg","smh","fyi","wbu","hbu","wyd","wym","tf","lmk","asap"}
PROFANITY = {"fuck","fucking","fucked","shit","bullshit","bitch","asshole","damn","crap","dick","piss"}


def repeated_letter_runs(text: str):
    return re.findall(r"([A-Za-z])\1{2,}", text)


def max_letter_run(text: str) -> int:
    runs = re.findall(r"(.)\1+", text, flags=re.UNICODE)
    return max((len(x) for x in runs), default=1)


def elongation_level(text: str):
    m = max_letter_run(text)
    if m >= EXTREME_REPEAT:
        return "extreme", m
    if m >= STRONG_REPEAT:
        return "strong", m
    if m >= MILD_REPEAT:
        return "mild", m
    return "none", m


def repeated_punctuation(text: str) -> int:
    return len(re.findall(r"[!?.,;:]{3,}", text))


def token_counts(text: str):
    tokens = re.findall(r"\b[\w’'-]+\b", text.lower(), flags=re.UNICODE)
    informal = sum(t in INFORMAL_WORDS for t in tokens)
    slang = sum(t in INTERNET_SLANG for t in tokens)
    profanity = sum(t in PROFANITY for t in tokens)
    return len(tokens), informal, slang, profanity


def backup_outputs(out: Path, filenames: list[str]):
    existing = [out / f for f in filenames if (out / f).exists()]
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = out / "_backups" / f"before_improvements2_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for p in existing:
        shutil.copy2(p, backup / p.name)
    return backup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Input CSV. If omitted, auto-detected.")
    ap.add_argument("--data-dir", default=None, help="Project data directory; default is ../data relative to this file.")
    ap.add_argument("--keep-c", type=float, default=0.02, help="Fraction of C/noisy examples retained as optional noise.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(args.data_dir, root)
    out, _ = ensure_dirs(data_dir)
    src = choose_input_csv(data_dir, args.input)
    print(f"[improvements2] input : {src}")
    print(f"[improvements2] output: {out}")

    df = pd.read_csv(src)
    ai_col, human_col, category_col = detect_pair_columns(df)
    df[ai_col] = df[ai_col].map(clean_text)
    df[human_col] = df[human_col].map(clean_text)

    df = df[(df[ai_col] != "") & (df[human_col] != "")].copy()
    before = len(df)
    df["_pair_key"] = (df[ai_col].str.lower().str.replace(r"\s+", " ", regex=True) + "|||" +
                        df[human_col].str.lower().str.replace(r"\s+", " ", regex=True))
    df = df.drop_duplicates("_pair_key").copy()
    exact_dupes_removed = before - len(df)

    metrics = df.apply(lambda r: pd.Series({
        "ai_words": word_count(r[ai_col]),
        "human_words": word_count(r[human_col]),
        "ai_human_word_ratio": word_count(r[human_col]) / max(word_count(r[ai_col]), 1),
        "human_elongation_level": elongation_level(r[human_col])[0],
        "human_max_letter_run": elongation_level(r[human_col])[1],
        "human_repeated_punctuation": repeated_punctuation(r[human_col]),
        "human_informal_count": token_counts(r[human_col])[1],
        "human_slang_count": token_counts(r[human_col])[2],
        "human_profanity_count": token_counts(r[human_col])[3],
    }), axis=1)
    df = pd.concat([df, metrics], axis=1)

    def classify(row):
        elong = row["human_elongation_level"]
        if elong == "extreme": return "C"
        if row["human_max_letter_run"] >= STRONG_REPEAT: return "C"
        if row["human_repeated_punctuation"] >= 2: return "C"
        if row["human_profanity_count"] >= 2: return "C"
        if row["human_slang_count"] >= 3: return "B"
        if row["human_informal_count"] >= 3: return "B"
        ratio = row["ai_human_word_ratio"]
        if ratio < 0.40 or ratio > 1.60: return "B"
        return "A"

    df["quality_tier"] = df.apply(classify, axis=1)
    df["semantic_status"] = "NOT_RUN"
    df["semantic_risk"] = False
    df["semantic_score"] = np.nan

    # Never put extreme/strong elongations into the core dataset. Keep a tiny C sample as noise.
    c = df[df["quality_tier"] == "C"]
    keep_c = c.sample(frac=min(max(args.keep_c, 0.0), 1.0), random_state=args.seed) if len(c) else c
    keep_keys = set(keep_c["_pair_key"])
    core = df[(df["quality_tier"].isin(["A", "B"])) | (df["_pair_key"].isin(keep_keys))].copy()
    core["is_noise_style_example"] = core["_pair_key"].isin(keep_keys)

    review = df[(df["quality_tier"] == "C") & (~df["_pair_key"].isin(keep_keys))].copy()
    review["review_reason"] = review.apply(lambda r: ",".join(x for x, ok in [
        ("extreme_elongation", r["human_elongation_level"] == "extreme"),
        ("strong_elongation", r["human_elongation_level"] == "strong"),
        ("repeated_punctuation", r["human_repeated_punctuation"] >= 2),
        ("profanity", r["human_profanity_count"] >= 2),
    ] if ok) or "style_noise", axis=1)

    # Re-split only after cleaning. Semantic steps later can add risk flags but should not silently alter split membership.
    if len(core) >= 10:
        train, temp = train_test_split(core, test_size=0.20, random_state=args.seed)
        validation, test = train_test_split(temp, test_size=0.50, random_state=args.seed)
    else:
        train = core.copy(); validation = core.iloc[0:0].copy(); test = core.iloc[0:0].copy()

    drop_internal = ["_pair_key"]
    files = ["base_clean.csv", "train.csv", "validation.csv", "test.csv", "manual_review.csv", "all_scored.csv", "report.txt"]
    backup = backup_outputs(out, files)
    for frame, name in [(core,"base_clean.csv"),(train,"train.csv"),(validation,"validation.csv"),(test,"test.csv"),(review,"manual_review.csv"),(df,"all_scored.csv")]:
        frame.drop(columns=drop_internal, errors="ignore").to_csv(out / name, index=False)

    report = f"""Lagom improvements2 report\n============================\nInput: {src}\nRows after exact duplicate removal: {len(df)}\nExact duplicates removed: {exact_dupes_removed}\nCore rows: {len(core)}\nReview/noise rows excluded from core: {len(review)}\nA rows: {(df.quality_tier == 'A').sum()}\nB rows: {(df.quality_tier == 'B').sum()}\nC rows: {(df.quality_tier == 'C').sum()}\nCore noise retained: {len(keep_c)}\n\nSemantic models are intentionally NOT run here. Run sbert_semantic.py, tfidf_similarity.py, and nli_cross_encoder.py next.\nBackup: {backup if backup else 'none'}\n"""
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
