#!/usr/bin/env python
"""Audit dataset length ratios and extract pairs that are too small for re-evaluation and re-generation.

Usage:
    python Data/Preprocessing/audit_length_ratios.py --min-ratio 0.70
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def audit_datasets(data_dir: Path, min_ratio: float = 0.70):
    train_path = data_dir / "train_normalized.csv"
    val_path = data_dir / "validation_normalized.csv"
    test_path = data_dir / "test_normalized.csv"

    reeval_rows = []

    for name, path in [("train", train_path), ("validation", val_path), ("test", test_path)]:
        if not path.exists():
            print(f"Skipping {name} (not found: {path})")
            continue

        df = pd.read_csv(path)
        print(f"\n========================================================")
        print(f"Auditing {name.upper()}: {path} ({len(df):,} total rows)")
        print(f"========================================================")

        # Compute accurate word counts from actual text
        ai_texts = df["AI Text"].fillna("").astype(str)
        hu_texts = df["Human Text"].fillna("").astype(str)
        categories = df["category"].fillna("general").astype(str)

        ai_words = ai_texts.apply(lambda x: len(x.split()))
        hu_words = hu_texts.apply(lambda x: len(x.split()))
        ratios = hu_words / ai_words.clip(lower=1)

        print(f"Word Count AI:    mean={ai_words.mean():.1f}, min={ai_words.min()}, max={ai_words.max()}")
        print(f"Word Count Human: mean={hu_words.mean():.1f}, min={hu_words.min()}, max={hu_words.max()}")
        print(f"Length Ratio:     mean={ratios.mean():.2f}, median={ratios.median():.2f}")
        print(f"Percentiles:      p10={ratios.quantile(0.10):.2f}, p25={ratios.quantile(0.25):.2f}, p50={ratios.quantile(0.50):.2f}, p75={ratios.quantile(0.75):.2f}")

        # Breakdown by category
        print(f"\nCategory breakdown for {name}:")
        for cat in sorted(categories.unique()):
            cat_mask = categories == cat
            cat_ratios = ratios[cat_mask]
            cat_too_small = (cat_ratios < min_ratio).sum()
            print(f"  - {cat:<12}: total={cat_mask.sum():5d} | mean_ratio={cat_ratios.mean():.2f} | <{min_ratio:.2f}: {cat_too_small:4d} ({cat_too_small/cat_mask.sum()*100:.1f}%)")

        # Collect flagged rows
        flagged_mask = ratios < min_ratio
        print(f"\nFlagged {flagged_mask.sum():,} pairs ({flagged_mask.mean()*100:.1f}%) with length ratio < {min_ratio:.2f}")

        for idx in df[flagged_mask].index:
            reeval_rows.append({
                "split": name,
                "original_index": idx,
                "category": categories.loc[idx],
                "ai_words": int(ai_words.loc[idx]),
                "human_words": int(hu_words.loc[idx]),
                "length_ratio": round(float(ratios.loc[idx]), 3),
                "reason": f"Human text is only {ratios.loc[idx]:.1%} of AI text length (< {min_ratio:.0%})",
                "AI Text": ai_texts.loc[idx],
                "Human Text": hu_texts.loc[idx],
            })

    reeval_df = pd.DataFrame(reeval_rows)
    out_path = data_dir / "pairs_for_reevaluation.csv"
    reeval_df.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved {len(reeval_df):,} total pairs across all splits flagged for re-evaluation to:\n  {out_path}")

    # Summary table by category
    print("\nSummary of flagged pairs by category:")
    print(reeval_df.groupby(["split", "category"]).size().unstack(fill_value=0))
    return reeval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ratio", type=float, default=0.70, help="Minimum human/AI word ratio threshold")
    parser.add_argument("--data-dir", default=None, help="Directory containing improved_data")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "Data" / "improved_data"

    audit_datasets(data_dir, min_ratio=args.min_ratio)
