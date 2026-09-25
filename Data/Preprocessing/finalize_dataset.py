#!/usr/bin/env python3
"""Lagom preprocessing step 5: merge semantic signals and build final splits.

Important design:
- SBERT and TF-IDF are used as the primary semantic screening signals.
- NLI is diagnostic/informational. Low NLI entailment is NOT a rejection rule.
- NLI contradiction can still flag an example because contradiction is a stronger
  warning that the rewrite may have changed meaning.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import resolve_data_dir


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)

    # Semantic screening thresholds.
    # NLI entailment is intentionally NOT used as a hard filter.
    ap.add_argument("--sbert-review", type=float, default=0.68)
    ap.add_argument("--sbert-low", type=float, default=0.55)
    ap.add_argument("--tfidf-low", type=float, default=0.15)
    ap.add_argument("--nli-contradiction", type=float, default=0.50)

    # Optional: retain rows marked REVIEW in the training pool.
    # By default REVIEW rows are excluded from the training pool.
    ap.add_argument(
        "--keep-review",
        action="store_true",
        help="Keep semantic-review rows in the training pool instead of excluding them.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(args.data_dir, root)
    out = data_dir / "improved_data"

    required_files = [
        "base_clean.csv",
        "sbert_scored.csv",
        "tfidf_scored.csv",
        "nli_scored.csv",
    ]
    missing_files = [name for name in required_files if not (out / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required scored files in "
            f"{out}:\n  - " + "\n  - ".join(missing_files)
        )

    base = pd.read_csv(out / "base_clean.csv")
    sb = pd.read_csv(out / "sbert_scored.csv")
    tf = pd.read_csv(out / "tfidf_scored.csv")
    nli = pd.read_csv(out / "nli_scored.csv")

    if not (len(sb) == len(tf) == len(nli) == len(base)):
        raise ValueError(
            "Row-count mismatch:\n"
            f"  base_clean.csv: {len(base)}\n"
            f"  sbert_scored.csv: {len(sb)}\n"
            f"  tfidf_scored.csv: {len(tf)}\n"
            f"  nli_scored.csv: {len(nli)}"
        )

    require_columns(sb, ["sbert_similarity"], "sbert_scored.csv")
    require_columns(tf, ["tfidf_similarity"], "tfidf_scored.csv")
    require_columns(
        nli,
        ["nli_entailment_prob", "nli_neutral_prob", "nli_contradiction_prob"],
        "nli_scored.csv",
    )

    # Only bring each scorer's own columns into the merged dataframe.
    # This avoids the previous bug where SBERT was accidentally asked for
    # TF-IDF/NLI columns.
    sb_cols = [c for c in sb.columns if c.startswith("sbert_")]
    tf_cols = [c for c in tf.columns if c.startswith("tfidf_")]
    nli_cols = [c for c in nli.columns if c.startswith("nli_")]

    df = pd.concat(
        [
            base.reset_index(drop=True),
            sb[sb_cols].reset_index(drop=True),
            tf[tf_cols].reset_index(drop=True),
            nli[nli_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    # ------------------------------------------------------------
    # Semantic screening
    #
    # IMPORTANT:
    # NLI entailment is NOT a hard filter.
    #
    # A paraphrase can have low NLI entailment while still being a
    # very good rewrite. This was observed in the actual Lagom data.
    # We therefore use NLI mainly to:
    #   1) expose contradiction,
    #   2) provide diagnostic information for later analysis.
    # ------------------------------------------------------------

    very_low_sbert = df["sbert_similarity"] < args.sbert_low

    weak_both = (
        (df["sbert_similarity"] < args.sbert_review)
        & (df["tfidf_similarity"] < args.tfidf_low)
    )

    nli_contradiction = (
        df["nli_contradiction_prob"] >= args.nli_contradiction
    )

    df["semantic_risk"] = (
        very_low_sbert
        | weak_both
        | nli_contradiction
    )

    # Status is descriptive, not a claim of ground truth.
    high_confidence = (
        (df["sbert_similarity"] >= 0.78)
        & (df["tfidf_similarity"] >= 0.15)
        & (~nli_contradiction)
    )

    df["semantic_status"] = np.select(
        [
            nli_contradiction,
            df["semantic_risk"],
            high_confidence,
        ],
        [
            "CONTRADICTION",
            "REVIEW",
            "HIGH_CONFIDENCE",
        ],
        default="ACCEPTABLE",
    )

    # NLI is explicitly retained as diagnostic information.
    df["nli_diagnostic"] = np.select(
        [
            df["nli_contradiction_prob"] >= args.nli_contradiction,
            df["nli_entailment_prob"] >= 0.70,
            df["nli_entailment_prob"] < 0.50,
        ],
        [
            "CONTRADICTION_SIGNAL",
            "ENTAILMENT_SIGNAL",
            "LOW_ENTAILMENT_INFORMATIVE_ONLY",
        ],
        default="MIXED",
    )

    # By default, semantic-review rows are kept out of the training pool,
    # but they remain in final_scored.csv and manual_review.csv.
    final_pool = df if args.keep_review else df[~df["semantic_risk"]].copy()

    dedupe_cols = [c for c in ["AI Text", "Human Text", "ai_text", "human_text"] if c in final_pool.columns]

    # Prefer the actual pair columns if present.
    if "AI Text" in final_pool.columns and "Human Text" in final_pool.columns:
        final_pool = final_pool.drop_duplicates(
            subset=["AI Text", "Human Text"]
        )
    elif "ai_text" in final_pool.columns and "human_text" in final_pool.columns:
        final_pool = final_pool.drop_duplicates(
            subset=["ai_text", "human_text"]
        )

    if len(final_pool) >= 10:
        train, temp = train_test_split(
            final_pool,
            test_size=0.20,
            random_state=args.seed,
        )
        validation, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=args.seed,
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
        "manual_review.csv",
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
    df[
        df["semantic_status"].isin(["REVIEW", "CONTRADICTION"])
    ].to_csv(out / "manual_review.csv", index=False)

    train.to_csv(out / "train.csv", index=False)
    validation.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    report = f"""Lagom finalization report
===========================

Input/core rows: {len(base)}
Semantic-risk rows: {int(df["semantic_risk"].sum())}
Final training pool: {len(final_pool)}
Train: {len(train)}
Validation: {len(validation)}
Test: {len(test)}

Semantic risk breakdown:
- SBERT very-low signal (< {args.sbert_low}): {int(very_low_sbert.sum())}
- SBERT+TF-IDF weak combined signal: {int(weak_both.sum())}
- NLI contradiction signal (>= {args.nli_contradiction}): {int(nli_contradiction.sum())}

NLI diagnostic:
- High entailment signal (>= 0.70): {int((df["nli_entailment_prob"] >= 0.70).sum())}
- Low entailment, informational only (< 0.50): {int((df["nli_entailment_prob"] < 0.50).sum())}

Signals used:
- SBERT sentence similarity: primary semantic similarity signal
- TF-IDF lexical similarity: supporting lexical-overlap signal
- NLI entailment/neutral/contradiction: diagnostic; contradiction can flag risk
- Deterministic style/quality signals from improvements2.py

IMPORTANT:
NLI low entailment is NOT an automatic rejection rule.
A low NLI entailment score can occur for valid paraphrases.
Thresholds are screening heuristics, not proof of meaning preservation.
Inspect semantic_risk.csv and manual_review.csv before training.

Backup: {backup if backup else 'none'}
"""
    (out / "final_report.txt").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
