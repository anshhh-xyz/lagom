"""
Data Filtering and Stratified Length Control Script for Lagom Humanizer (Phase 2)
Builds train_filtered.csv and validation_filtered.csv with clean, authentic human targets
and balanced length ratio mix (0.70-0.80: 12%, 0.80-0.90: 63%, 0.90-0.95: 20%, >0.95: <=5%).
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer

try:
    import pysbd
    SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except ImportError:
    SEGMENTER = None

# ==============================================================================
# CONFIGURATION DICTIONARY
# ==============================================================================
CONFIG = {
    "min_cov": 0.55,
    "mean_cov": 0.75,
    "copy_rate_strict": 0.60,    # for essay, academic
    "copy_rate_loose": 0.75,     # for general, email, document
    "cliche_multiplier": 2.0,
    "cliche_floor": 4.0,         # Minimum threshold when baseline median is 0.0
    "min_ratio": 0.70,           # Hard rule: drop every row with ratio < 0.70
    "target_mix": {
        "0.70-0.80": 0.12,
        "0.80-0.90": 0.63,
        "0.90-0.95": 0.20,
        ">0.95": 0.05,
    },
    "cliche_words": [
        "furthermore", "moreover", "consequently", "in conclusion",
        "landscape", "tapestry", "multifaceted", "paramount",
        "delve", "testament", "pivotal", "foster", "harness",
        "navigate", "realm", "crucial", "profound", "underscore",
        "symbiotic", "holistic", "unprecedented", "paradigm",
        "catalyst", "transformative", "intricate", "seamless",
        "vibrant", "ecosystem", "synergy", "steadfast",
        "cornerstone", "ultimately"
    ],
}

CLICHE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in CONFIG["cliche_words"]) + r")\b",
    re.IGNORECASE
)
CONTRACTION_PATTERN = re.compile(
    r"\b([a-zA-Z]+)('re|'ve|'d|'ll|'t|'s|n't)\b",
    re.IGNORECASE
)
CITATION_PATTERN = re.compile(r"\([A-Z][^)]*\d{4}\)")
TITLE_LEAK_PATTERN = re.compile(
    r"(?i)^(?:[^\n]{0,80}\s+(?:Essay|Research Paper|Report)|(?:Title|Subject)\s*:[^\n]*)\n+"
)
TRAILING_REF_PATTERN = re.compile(
    r"(?i)\n\s*(?:Works Cited|References|Bibliography)[\s\S]*$"
)


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if SEGMENTER is not None:
        try:
            return [s.strip() for s in SEGMENTER.segment(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text.strip()) if s.strip()]


def strip_artifacts(text: str) -> str:
    """Clean citations, title headers, and trailing bibliographies from Human Text."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    # 1. Strip title leakage prefix
    cleaned = TITLE_LEAK_PATTERN.sub("", cleaned).strip()
    # 2. Strip trailing references
    cleaned = TRAILING_REF_PATTERN.sub("", cleaned).strip()
    # 3. Strip in-text citations like (Smith et al., 2018)
    cleaned = CITATION_PATTERN.sub("", cleaned).strip()
    # Normalize double spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned


def recompute_style(human_text: str, ai_text: str):
    h_sents = split_sentences(human_text)
    h_words = len(human_text.split())
    a_words = len(ai_text.split())
    ratio = round(h_words / max(a_words, 1), 4)
    burst = float(np.std([len(s.split()) for s in h_sents])) if len(h_sents) >= 2 else 0.0
    contr = round((len(CONTRACTION_PATTERN.findall(human_text)) / max(h_words, 1)) * 100.0, 2)
    cliche = round((len(CLICHE_PATTERN.findall(human_text)) / max(h_words, 1)) * 1000.0, 2)
    return h_words, ratio, burst, contr, cliche


def get_ratio_bin(r: float) -> str:
    if r < 0.70:
        return "<0.70"
    elif 0.70 <= r < 0.80:
        return "0.70-0.80"
    elif 0.80 <= r < 0.90:
        return "0.80-0.90"
    elif 0.90 <= r <= 0.95:
        return "0.90-0.95"
    else:
        return ">0.95"


def filter_dataset(
    df: pd.DataFrame,
    emb_model: SentenceTransformer,
    cfg: dict,
    is_train: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply artifact cleaning, quality filters, and stratified resampling."""
    print(f"\n[*] Filtering dataset (Initial rows: {len(df)})...")
    cat_col = "category"
    source_col = "human_source"

    # Compute baseline median cliches for hf_dataset_import
    hf_mask = df[source_col].astype(str).str.contains("hf_dataset_import", case=False, na=False)
    hf_medians = df[hf_mask].groupby(cat_col)["human_cliche_density"].median().to_dict()

    strict_cats = {"essay", "academic"}
    reasons = []
    cleaned_humans = []
    cleaned_ratios = []
    cleaned_h_words = []
    cleaned_bursts = []
    cleaned_contrs = []
    cleaned_cliches = []
    passed_mask = []

    for idx, row in df.iterrows():
        cat = str(row[cat_col]).lower()
        src = str(row[source_col])
        ai_txt = str(row.get("AI Text", "")).strip()
        hu_txt = str(row.get("Human Text", "")).strip()

        # 1. Strip artifacts in Human Text
        cleaned_h = strip_artifacts(hu_txt)
        cleaned_humans.append(cleaned_h)

        # Recompute style with cleaned text
        h_words, ratio, burst, contr, cliche = recompute_style(cleaned_h, ai_txt)
        cleaned_h_words.append(h_words)
        cleaned_ratios.append(ratio)
        cleaned_bursts.append(burst)
        cleaned_contrs.append(contr)
        cleaned_cliches.append(cliche)

        # 2. Check Hard Rule: Ratio < 0.70
        drop_reasons = []
        if ratio < cfg["min_ratio"]:
            drop_reasons.append(f"ratio_too_low({ratio:.2f}<{cfg['min_ratio']})")

        # 3. Check Coverage Rules
        min_cov = float(row.get("min_cov", 0.0))
        mean_cov = float(row.get("mean_cov", 0.0))
        if min_cov < cfg["min_cov"] or mean_cov < cfg["mean_cov"]:
            drop_reasons.append(f"poor_coverage(min={min_cov:.2f},mean={mean_cov:.2f})")

        # 4. Check Copy Rate
        copy_rate = float(row.get("copy_rate", 0.0))
        thresh_copy = cfg["copy_rate_strict"] if cat in strict_cats else cfg["copy_rate_loose"]
        if copy_rate > thresh_copy:
            drop_reasons.append(f"high_copy_rate({copy_rate:.2f}>{thresh_copy})")

        # 5. Check Human Cliché Density
        hf_med = hf_medians.get(cat, 0.0)
        cliche_thresh = max(cfg["cliche_multiplier"] * hf_med, cfg["cliche_floor"])
        if cliche > cliche_thresh:
            drop_reasons.append(f"high_cliches({cliche:.1f}>{cliche_thresh:.1f})")

        # Status
        if drop_reasons:
            reasons.append("; ".join(drop_reasons))
            passed_mask.append(False)
        else:
            reasons.append("PASSED")
            passed_mask.append(True)

    # Attach cleaned columns and reason
    processed_df = df.copy()
    processed_df["Human Text"] = cleaned_humans
    processed_df["human_words"] = cleaned_h_words
    processed_df["ratio"] = cleaned_ratios
    processed_df["burstiness"] = cleaned_bursts
    processed_df["contraction_rate"] = cleaned_contrs
    processed_df["human_cliche_density"] = cleaned_cliches
    processed_df["target_words"] = processed_df["human_words"].apply(lambda w: int(round(w, -1)))
    processed_df["reason_dropped"] = reasons

    surviving_df = processed_df[passed_mask].copy()
    dropped_df = processed_df[~np.array(passed_mask)].copy()

    print(f"  Quality Filter Results: {len(surviving_df)} / {len(df)} passed quality filters ({(len(surviving_df)/len(df))*100:.1f}%)")
    print(f"  Surviving counts by category:\n{surviving_df[cat_col].value_counts().to_string()}")

    # 6. Stratified Resampling for Length Mix (for Train split)
    shortfalls = {}
    if is_train:
        print("\n[*] Applying Stratified Resampling to target ratio mix:")
        print("    Target mix: 0.70-0.80: 12% | 0.80-0.90: 63% | 0.90-0.95: 20% | >0.95: <=5%")
        surviving_df["ratio_bin"] = surviving_df["ratio"].apply(get_ratio_bin)

        resampled_dfs = []
        for cat, c_grp in surviving_df.groupby(cat_col):
            b_counts = c_grp["ratio_bin"].value_counts().to_dict()
            # Determine maximum possible balanced subset
            # Core bin is 0.80-0.90 which needs to be 63%
            core_count = b_counts.get("0.80-0.90", 0)
            target_core_share = cfg["target_mix"]["0.80-0.90"]

            # Compute feasible category total
            total_cat = int(core_count / target_core_share) if target_core_share > 0 else core_count

            cat_samples = []
            for b_name, t_share in cfg["target_mix"].items():
                target_n = int(round(total_cat * t_share))
                available = c_grp[c_grp["ratio_bin"] == b_name]
                avail_n = len(available)

                if avail_n < target_n:
                    shortfalls[f"{cat}_{b_name}"] = {"target": target_n, "available": avail_n, "shortfall": target_n - avail_n}
                    cat_samples.append(available)
                else:
                    cat_samples.append(available.sample(target_n, random_state=42))

            resampled_dfs.append(pd.concat(cat_samples, ignore_index=True))

        final_resampled = pd.concat(resampled_dfs, ignore_index=True)
        print(f"  Final Stratified Training Set: {len(final_resampled)} rows")
        return final_resampled, processed_df, shortfalls
    else:
        return surviving_df, processed_df, shortfalls


def generate_filter_report(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    shortfalls: dict,
    out_md: Path,
):
    """Generate detailed filter_report.md with before/after statistics."""
    cat_col = "category"
    source_col = "human_source"

    md = []
    md.append("# Lagom V3 Data Quality Filtering & Stratified Resampling Report\n")
    md.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("## 1. Dataset Row Counts: Before vs. After Filtering\n")

    md.append("### By Category\n")
    cat_summary = []
    for cat in before_df[cat_col].unique():
        b_cnt = (before_df[cat_col] == cat).sum()
        a_cnt = (after_df[cat_col] == cat).sum()
        retention = (a_cnt / b_cnt) * 100.0 if b_cnt > 0 else 0.0
        cat_summary.append({"Category": cat, "Initial": b_cnt, "Filtered": a_cnt, "Retention (%)": f"{retention:.1f}%"})
    md.append(pd.DataFrame(cat_summary).to_markdown(index=False))

    md.append("\n\n### By Human Source\n")
    src_summary = []
    for src in before_df[source_col].unique():
        b_cnt = (before_df[source_col] == src).sum()
        a_cnt = (after_df[source_col] == src).sum()
        retention = (a_cnt / b_cnt) * 100.0 if b_cnt > 0 else 0.0
        src_summary.append({"Human Source": src, "Initial": b_cnt, "Filtered": a_cnt, "Retention (%)": f"{retention:.1f}%"})
    md.append(pd.DataFrame(src_summary).to_markdown(index=False))

    md.append("\n\n## 2. Stylistic Quality Metrics: Before vs. After\n")
    metrics = ["burstiness", "contraction_rate", "human_cliche_density", "ratio"]
    metric_summary = []
    for m in metrics:
        b_mean = before_df[m].mean()
        b_med = before_df[m].median()
        a_mean = after_df[m].mean()
        a_med = after_df[m].median()
        metric_summary.append({
            "Metric": m,
            "Initial Mean": f"{b_mean:.2f}",
            "Initial Median": f"{b_med:.2f}",
            "Filtered Mean": f"{a_mean:.2f}",
            "Filtered Median": f"{a_med:.2f}",
        })
    md.append(pd.DataFrame(metric_summary).to_markdown(index=False))

    md.append("\n\n## 3. Stratified Length Ratio Mix in Filtered Training Set\n")
    after_df["ratio_bin"] = after_df["ratio"].apply(get_ratio_bin)
    ratio_mix = after_df["ratio_bin"].value_counts(normalize=True) * 100.0
    mix_table = []
    for b_name in ["<0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", ">0.95"]:
        mix_table.append({
            "Length Ratio Bin": b_name,
            "Target Mix": f"{CONFIG['target_mix'].get(b_name, 0.0)*100:.1f}%",
            "Actual Filtered Mix": f"{ratio_mix.get(b_name, 0.0):.1f}%",
            "Row Count": (after_df['ratio_bin'] == b_name).sum()
        })
    md.append(pd.DataFrame(mix_table).to_markdown(index=False))

    if shortfalls:
        md.append("\n\n## 4. Bin Shortfall Report (Quality Preservation)\n")
        md.append("Per the hard rules, quality thresholds were strictly maintained without relaxation. Below are reported shortfalls:\n")
        sf_table = []
        for k, v in shortfalls.items():
            sf_table.append({
                "Category / Bin": k,
                "Target Count": v["target"],
                "Available Clean Rows": v["available"],
                "Shortfall": v["shortfall"],
            })
        md.append(pd.DataFrame(sf_table).to_markdown(index=False))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[+] Filter report saved to {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Lagom Data Quality Filter (Phase 2)")
    parser.add_argument("--audit-csv", default="Data/improved_data/audit_scores.csv")
    parser.add_argument("--train-out", default="Data/improved_data/train_filtered.csv")
    parser.add_argument("--val-out", default="Data/improved_data/validation_filtered.csv")
    parser.add_argument("--report-out", default="Data/improved_data/filter_report.md")
    args = parser.parse_args()

    print("=" * 80)
    print("Loading audit scores from:", args.audit_csv)
    print("=" * 80)
    audit_df = pd.read_csv(args.audit_csv)

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    # Split train and validation
    train_raw = audit_df[audit_df["split"] == "train"].copy()
    val_raw = audit_df[audit_df["split"] == "validation"].copy()

    # Process Train Split
    train_filtered, train_annotated, train_shortfalls = filter_dataset(train_raw, emb_model, CONFIG, is_train=True)
    train_filtered.to_csv(args.train_out, index=False, encoding="utf-8")
    print(f"[+] Saved train_filtered.csv ({len(train_filtered)} rows) to {args.train_out}")

    # Process Validation Split
    val_filtered, val_annotated, val_shortfalls = filter_dataset(val_raw, emb_model, CONFIG, is_train=False)
    val_filtered.to_csv(args.val_out, index=False, encoding="utf-8")
    print(f"[+] Saved validation_filtered.csv ({len(val_filtered)} rows) to {args.val_out}")

    # Generate Report
    combined_annotated = pd.concat([train_annotated, val_annotated], ignore_index=True)
    combined_filtered = pd.concat([train_filtered, val_filtered], ignore_index=True)
    generate_filter_report(combined_annotated, combined_filtered, train_shortfalls, Path(args.report_out))

    print("\nPhase 2 Complete! test_normalized.csv left completely untouched.")


if __name__ == "__main__":
    main()
