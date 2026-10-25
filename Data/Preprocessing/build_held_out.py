#!/usr/bin/env python
"""
Build eval/held_out.jsonl from test_normalized.csv (40 stratified rows)
plus the 5-paragraph essay from samples.md.

Stratified sampling: proportional to category size, seed=42.
Each JSONL line: {"text": <AI Text>, "category": <category>, "source": "test_set" | "samples_md"}
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Script lives in Data/Preprocessing/ - repo root is two levels up
ROOT = Path(__file__).resolve().parent.parent.parent
TEST_CSV = ROOT / "Data" / "improved_data" / "test_normalized.csv"
SAMPLES_MD = ROOT / "samples.md"
OUT_JSONL = ROOT / "eval" / "held_out.jsonl"

TOTAL_ROWS = 40
SEED = 42


def extract_samples_essay(samples_path: Path) -> str:
    """Extract the AI essay text from samples.md (the part before 'Humanized By Lagom')."""
    text = samples_path.read_text(encoding="utf-8")
    match = re.search(r"AI\(essay\)-(.*?)(?:Huma[mn]ized By Lagom-|\Z)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    parts = re.split(r"Huma[mn]ized By Lagom-", text, flags=re.DOTALL)
    return parts[0].replace("AI(essay)-", "").strip() if parts else text.strip()


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows stratified by category, proportional to category sizes."""
    cats = df["category"].value_counts()
    total = len(df)

    quotas = {}
    for cat in cats.index:
        quotas[cat] = max(1, round(n * cats[cat] / total))

    quota_total = sum(quotas.values())
    while quota_total > n:
        biggest = max(quotas, key=quotas.get)
        quotas[biggest] -= 1
        quota_total -= 1
    while quota_total < n:
        biggest = max(quotas, key=lambda c: cats[c] / max(quotas.get(c, 1), 1))
        quotas[biggest] += 1
        quota_total += 1

    sampled = []
    for cat, quota in quotas.items():
        cat_df = df[df["category"] == cat]
        take = min(len(cat_df), quota)
        sampled.append(cat_df.sample(n=take, random_state=seed))

    result = pd.concat(sampled, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    print(f"Repo root  : {ROOT}")
    print(f"Loading    : {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    print(f"  {len(df)} rows | categories: {dict(df['category'].value_counts())}")

    sample = stratified_sample(df, TOTAL_ROWS, SEED)
    print(f"\nSampled {len(sample)} rows (stratified):")
    print(f"  {dict(sample['category'].value_counts())}")

    records = []
    for _, row in sample.iterrows():
        ai_text = str(row["AI Text"]).strip()
        ai_w = int(row["ai_words"]) if "ai_words" in row and pd.notna(row.get("ai_words")) else len(ai_text.split())
        records.append({
            "text": ai_text,
            "category": str(row["category"]).strip(),
            "source": "test_set",
            "human_reference": str(row["Human Text"]).strip(),
            "ai_words": ai_w,
            "target_words": int(round(ai_w * 0.85, -1)),
        })

    print(f"\nExtracting essay from: {SAMPLES_MD}")
    essay_text = extract_samples_essay(SAMPLES_MD)
    essay_words = len(essay_text.split())
    print(f"  Essay length: {essay_words} words")
    records.append({
        "text": essay_text,
        "category": "essay",
        "source": "samples_md",
        "human_reference": None,
        "ai_words": essay_words,
        "target_words": int(round(essay_words * 0.85, -1)),
    })

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} records to {OUT_JSONL}")
    print(f"  - {TOTAL_ROWS} stratified test rows")
    print(f"  - 1 samples.md essay (essay, {essay_words} words)")
    print("\nCategory distribution:")
    cats = Counter(r["category"] for r in records)
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:12s}: {cnt}")


if __name__ == "__main__":
    main()
