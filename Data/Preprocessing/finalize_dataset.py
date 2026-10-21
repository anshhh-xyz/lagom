from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import resolve_data_dir


def _compute_semantic_score(df: pd.DataFrame) -> pd.Series:
    """Composite semantic quality score in [0, 1].

    Weighted blend:
      - SBERT similarity (0–1)          weight 0.50  — primary semantic alignment
      - NLI entailment probability      weight 0.35  — factual preservation
      - TF-IDF lexical similarity       weight 0.15  — surface lexical overlap
    Missing sub-scores fall back to the column mean so NaN never propagates.
    """
    sbert = df["sbert_similarity"].fillna(df["sbert_similarity"].mean())
    entail = df["nli_entailment_prob"].fillna(df["nli_entailment_prob"].mean())
    tfidf = df["tfidf_similarity"].fillna(df["tfidf_similarity"].mean()) if "tfidf_similarity" in df.columns else pd.Series(0.5, index=df.index)
    return (0.50 * sbert + 0.35 * entail + 0.15 * tfidf).clip(0.0, 1.0)


def _recompute_quality_tier(df: pd.DataFrame) -> pd.Series:
    """Rebuild quality tier labels using actual scoring signals.

    Tier A: high semantic confidence, balanced length, clean prose.
    Tier B: moderate issues — tolerable for training with reduced weight.
    Tier C: kept as optional noise only (should not reach this after filtering).

    Criteria (applied in priority order):
      C → sbert < 0.72  OR  contradiction_prob > 0.25  OR  elongation extreme/strong
      B → sbert in [0.72, 0.80)  OR  ratio outside (0.65, 1.40)  OR  informal_count >= 3
      A → everything else
    """
    sbert = df["sbert_similarity"].fillna(0.0)
    contra = df["nli_contradiction_prob"].fillna(0.0)
    ratio = df["ai_human_word_ratio"].fillna(1.0)
    elong = df["human_elongation_level"].fillna("none") if "human_elongation_level" in df.columns else pd.Series("none", index=df.index)
    informal = df["human_informal_count"].fillna(0) if "human_informal_count" in df.columns else pd.Series(0, index=df.index)

    is_c = (sbert < 0.72) | (contra > 0.25) | elong.isin(["extreme", "strong"])
    is_b = (~is_c) & ((sbert < 0.80) | (ratio < 0.65) | (ratio > 1.40) | (informal >= 3))

    return np.select([is_c, is_b], ["C", "B"], default="A")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sbert-low", type=float, default=0.70)
    ap.add_argument("--nli-contradiction", type=float, default=0.35)
    ap.add_argument("--min-ratio", type=float, default=0.60)
    ap.add_argument("--max-ratio", type=float, default=1.60)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(args.data_dir, root)
    out = data_dir / "improved_data"

    scored_path = out / "final_scored.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing {scored_path}")

    df = pd.read_csv(scored_path)

    # Remove rows where AI input ends in ellipsis — these are truncated source excerpts
    ai_text = df["AI Text"].fillna("").astype(str).str.strip()
    truncated_ai = ai_text.str.endswith("...") | ai_text.str.endswith("\u2026")
    df = df[~truncated_ai].copy()

    ai_w = df["AI Text"].fillna("").apply(lambda s: len(str(s).split()))
    hu_w = df["Human Text"].fillna("").apply(lambda s: len(str(s).split()))
    ratios = hu_w / ai_w.clip(lower=1)
    df["ai_human_word_ratio"] = ratios

    low_sbert = df["sbert_similarity"] < args.sbert_low
    high_contradiction = df["nli_contradiction_prob"] > args.nli_contradiction
    bad_ratio = (ratios < args.min_ratio) | (ratios > args.max_ratio)
    too_short = (ai_w < 30) | (hu_w < 30)

    df["semantic_risk"] = low_sbert | high_contradiction | bad_ratio | too_short

    df["semantic_status"] = np.select(
        [
            high_contradiction,
            low_sbert,
            bad_ratio | too_short,
            (df["sbert_similarity"] >= 0.80) & (~high_contradiction),
        ],
        [
            "CONTRADICTION",
            "LOW_SIMILARITY",
            "LENGTH_MISMATCH",
            "HIGH_CONFIDENCE",
        ],
        default="ACCEPTABLE",
    )

    # Populate composite semantic_score (was previously all-NaN placeholder)
    df["semantic_score"] = _compute_semantic_score(df)

    # Rebuild quality_tier using real scoring signals (was 99.3% A without signal basis)
    df["quality_tier"] = _recompute_quality_tier(df)

    final_pool = df[~df["semantic_risk"]].copy()

    hu_lower = final_pool["Human Text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True)
    final_pool = final_pool.loc[~hu_lower.duplicated()].copy()

    if len(final_pool) >= 10:
        train, temp = train_test_split(
            final_pool,
            test_size=0.20,
            random_state=args.seed,
            stratify=final_pool["category"] if final_pool["category"].nunique() > 1 else None,
        )
        validation, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=args.seed,
            stratify=temp["category"] if temp["category"].nunique() > 1 else None,
        )
    else:
        train = final_pool.copy()
        validation = final_pool.iloc[0:0].copy()
        test = final_pool.iloc[0:0].copy()

    outputs = [
        "final_scored.csv",
        "semantic_risk.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "train_normalized.csv",
        "validation_normalized.csv",
        "test_normalized.csv",
        "final_report.txt",
    ]

    existing = [out / f for f in outputs if (out / f).exists()]
    backup = None

    if existing:
        backup = (
            out
            / "_backups"
            / f"before_finalize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        backup.mkdir(parents=True, exist_ok=False)
        for p in existing:
            shutil.copy2(p, backup / p.name)

    df.to_csv(out / "final_scored.csv", index=False)
    df[df["semantic_risk"]].to_csv(out / "semantic_risk.csv", index=False)

    train.to_csv(out / "train.csv", index=False)
    validation.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    train.to_csv(out / "train_normalized.csv", index=False)
    validation.to_csv(out / "validation_normalized.csv", index=False)
    test.to_csv(out / "test_normalized.csv", index=False)

    tier_counts = final_pool["quality_tier"].value_counts()
    report = f"""Lagom V2 finalization report
===========================

Input rows: {len(df)}
Pruned semantic/length risk rows: {int(df["semantic_risk"].sum())}
Final clean training pool: {len(final_pool)}
Train (80%): {len(train)}
Validation (10%): {len(validation)}
Test (10%): {len(test)}

Pruning breakdown:
- SBERT low (< {args.sbert_low}): {int(low_sbert.sum())}
- NLI contradiction signal (> {args.nli_contradiction}): {int(high_contradiction.sum())}
- Length ratio outside [{args.min_ratio}, {args.max_ratio}]: {int(bad_ratio.sum())}
- Dropped duplicates: {len(df[~df["semantic_risk"]]) - len(final_pool)}

Quality tier distribution (rebuilt from scoring signals):
{tier_counts.to_string()}

Semantic score stats (composite SBERT/NLI/TF-IDF):
  mean:  {df['semantic_score'].mean():.4f}
  median:{df['semantic_score'].median():.4f}
  min:   {df['semantic_score'].min():.4f}
  max:   {df['semantic_score'].max():.4f}

Category balance in final train pool:
{final_pool['category'].value_counts().to_string()}

Backup: {backup if backup else 'none'}
"""
    (out / "final_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
