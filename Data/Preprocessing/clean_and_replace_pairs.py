#!/usr/bin/env python
"""Apply all regenerated 1:1 pairs into train_normalized.csv and purge any remaining
un-regenerated truncated pairs (< 0.70 ratio) so the training data has ZERO summarization bias.

Usage:
    python Data/Preprocessing/clean_and_replace_pairs.py --min-ratio 0.70
"""

import argparse
from pathlib import Path
import pandas as pd


def clean_and_replace(data_dir: Path, min_ratio: float = 0.70, backup: bool = True, target_splits: list[str] | None = None):
    splits = target_splits or ["train", "validation", "test"]
    cache_path = data_dir / "regenerated_cache.csv"
    cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()

    for split_name in splits:
        file_path = data_dir / f"{split_name}_normalized.csv"
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        initial_len = len(df)
        print(f"\n========================================================")
        print(f"Cleaning & Replacing: {file_path.name} ({initial_len:,} rows)")
        print(f"========================================================")

        if backup:
            b_path = data_dir / f"{split_name}_normalized.backup_{initial_len}.csv"
            if not b_path.exists():
                df.to_csv(b_path, index=False)
                print(f"Created safety backup at {b_path.name}")

        replaced = 0
        if not cache_df.empty:
            better_items = cache_df[(cache_df["new_ratio"] >= 0.85) & (cache_df["split"] == split_name)]
            existing_ai_map = {str(txt).strip(): idx for idx, txt in enumerate(df["AI Text"].fillna(""))}
            appended_rows = []

            for _, item in better_items.iterrows():
                ai_clean = str(item["AI Text"]).strip()
                new_human = str(item["New Human Text"])
                new_w = int(item["new_human_words"])
                new_ratio = float(item["new_ratio"])
                sbert_sim = float(item.get("sbert_similarity", 0.88))
                sbert_flag = str(item.get("sbert_semantic_flag", "HIGH"))
                src_model = str(item.get("source_model", "llm_regenerated_1_to_1"))
                cat = str(item["category"])

                if ai_clean in existing_ai_map:
                    t_idx = existing_ai_map[ai_clean]
                    df.loc[t_idx, "Human Text"] = new_human
                    df.loc[t_idx, "human_words"] = new_w
                    df.loc[t_idx, "ai_human_word_ratio"] = new_ratio
                    df.loc[t_idx, "sbert_similarity"] = sbert_sim
                    df.loc[t_idx, "sbert_semantic_flag"] = sbert_flag
                    df.loc[t_idx, "semantic_status"] = "REGENERATED_1_TO_1"
                    df.loc[t_idx, "quality_tier"] = "A"
                    replaced += 1
                else:
                    new_row = {col: None for col in df.columns}
                    new_row.update({
                        "AI Text": ai_clean,
                        "Human Text": new_human,
                        "category": cat,
                        "source_model": src_model,
                        "word_count": new_w,
                        "human_source": "llm_regenerated_1_to_1",
                        "contributor": "lagom_pipeline",
                        "reviewed": True,
                        "ai_words": int(item["ai_words"]),
                        "human_words": new_w,
                        "ai_human_word_ratio": new_ratio,
                        "quality_tier": "A",
                        "semantic_status": "REGENERATED_1_TO_1",
                        "sbert_similarity": sbert_sim,
                        "sbert_semantic_flag": sbert_flag,
                    })
                    appended_rows.append(new_row)
                    existing_ai_map[ai_clean] = len(df) + len(appended_rows) - 1

            if appended_rows:
                df = pd.concat([df, pd.DataFrame(appended_rows)], ignore_index=True)
            print(f"[REPLACED] Injected {replaced:,} in-place, appended {len(appended_rows):,} newly regenerated rows.")

        # Recalculate true ratios
        ai_w = df["AI Text"].fillna("").astype(str).apply(lambda s: len(s.split()))
        hu_w = df["Human Text"].fillna("").astype(str).apply(lambda s: len(s.split()))
        df["ai_words"] = ai_w
        df["human_words"] = hu_w
        df["ai_human_word_ratio"] = (hu_w / ai_w.clip(lower=1)).round(4)

        bad_mask = df["ai_human_word_ratio"] < min_ratio
        dropped_count = bad_mask.sum()
        clean_df = df[~bad_mask].copy().reset_index(drop=True)
        clean_df.to_csv(file_path, index=False)
        print(f"[DONE] Saved {file_path.name}: {len(clean_df):,} clean rows (purged {dropped_count:,} truncated rows, mean ratio: {clean_df['ai_human_word_ratio'].mean():.2f}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ratio", type=float, default=0.70, help="Minimum acceptable ratio (default: 0.70)")
    parser.add_argument("--split", default=None, choices=["train", "validation", "test", "all"], help="Specific split or all")
    parser.add_argument("--data-dir", default=None, help="Directory containing improved_data")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a backup file")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "Data" / "improved_data"

    target_splits = None if (not args.split or args.split == "all") else [args.split]
    clean_and_replace(data_dir, min_ratio=args.min_ratio, backup=not args.no_backup, target_splits=target_splits)
