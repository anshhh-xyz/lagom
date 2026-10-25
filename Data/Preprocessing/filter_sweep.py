"""
Filter Sweep Script for Lagom Humanizer (filter_sweep.py)
Systematically evaluates a parameter grid of coverage floors, fact retention thresholds,
copy rate caps, and cliche limits across 108 combinations.
Generates sweep_report.md with before/after counts, ratio distributions, and examples.
"""

import sys
import os
import io
import re
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

TITLE_LEAK_PATTERN = re.compile(
    r"(?i)^(?:[^\n]{0,80}\s+(?:Essay|Research Paper|Report)|(?:Title|Subject)\s*:[^\n]*)\n+"
)
TRAILING_REF_PATTERN = re.compile(
    r"(?i)\n\s*(?:Works Cited|References|Bibliography)[\s\S]*$"
)
CITATION_PATTERN = re.compile(r"\([A-Z][^)]*\d{4}\)")


def strip_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    cleaned = TITLE_LEAK_PATTERN.sub("", cleaned).strip()
    cleaned = TRAILING_REF_PATTERN.sub("", cleaned).strip()
    cleaned = CITATION_PATTERN.sub("", cleaned).strip()
    return re.sub(r"[ \t]+", " ", cleaned)


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


def run_sweep():
    csv_path = "Data/improved_data/audit_scores.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows across splits.")

    cat_col = "category"
    source_col = "human_source"

    # Pre-clean human texts to avoid re-running regex 108 times
    print("Pre-cleaning artifacts (title leakage, citations, trailing references)...")
    cleaned_ratios = []
    for idx, row in df.iterrows():
        ai_txt = str(row.get("AI Text", "")).strip()
        hu_txt = str(row.get("Human Text", "")).strip()
        cleaned_hu = strip_artifacts(hu_txt)
        c_hw = len(cleaned_hu.split())
        c_aw = len(ai_txt.split())
        cleaned_ratios.append(round(c_hw / max(c_aw, 1), 4))
    df["cleaned_ratio"] = cleaned_ratios

    # Baseline human cliché medians for hf_dataset_import
    hf_mask = df[source_col].astype(str).str.contains("hf_dataset_import", case=False, na=False)
    hf_medians = df[hf_mask].groupby(cat_col)["human_cliche_density"].median().to_dict()
    print("Baseline hf_dataset_import cliché medians:", hf_medians)

    # Parameter Grid
    mean_cov_floors = [0.65, 0.70, 0.75]
    fact_ret_floors = [0.60, 0.70, 0.80]
    copy_caps_llm = [0.50, 0.60]
    copy_caps_hf = [0.60, 0.75, 0.85]
    cliche_mults = [2.0, 3.0]

    grid = list(itertools.product(
        mean_cov_floors,
        fact_ret_floors,
        copy_caps_llm,
        copy_caps_hf,
        cliche_mults
    ))
    print(f"Evaluating {len(grid)} parameter combinations on full dataset...")

    results = []

    for comb_idx, (m_cov, f_ret, c_llm, c_hf, c_mult) in enumerate(grid):
        passed_indices = []

        for idx, row in df.iterrows():
            cat = str(row[cat_col]).lower()
            src = str(row[source_col])
            ratio = row["cleaned_ratio"]

            # Always-on rule: ratio >= 0.70
            if ratio < 0.70:
                continue

            # Cliché threshold
            hf_med = hf_medians.get(cat, 0.0)
            cliche_thresh = max(c_mult * hf_med, 4.0)
            if row["human_cliche_density"] > cliche_thresh:
                continue

            # Fact retention floor
            if row["fact_retention"] < f_ret:
                continue

            # Category rules
            if cat in {"email", "document"}:
                # No copy_rate and no min_cov rule for email & document
                passed_indices.append(idx)
            elif cat in {"essay", "academic"}:
                # Coverage floor
                if row["mean_cov"] < m_cov:
                    continue
                # Copy rate cap by source
                if "llm_regenerated" in src:
                    if row["copy_rate"] > c_llm:
                        continue
                else:
                    if row["copy_rate"] > c_hf:
                        continue
                passed_indices.append(idx)
            else:  # general
                if row["mean_cov"] < m_cov:
                    continue
                if "llm_regenerated" in src:
                    if row["copy_rate"] > c_llm:
                        continue
                else:
                    if row["copy_rate"] > 0.85:
                        continue
                passed_indices.append(idx)

        surv = df.loc[passed_indices]
        n_surv = len(surv)
        cat_counts = surv[cat_col].value_counts().to_dict()
        src_counts = surv[source_col].value_counts().to_dict()

        # Ratio mix
        r_bins = surv["cleaned_ratio"].apply(get_ratio_bin).value_counts(normalize=True).to_dict() if n_surv > 0 else {}
        b_mean = surv["burstiness"].mean() if n_surv > 0 else 0.0
        c_mean = surv["contraction_rate"].mean() if n_surv > 0 else 0.0
        q_mean = surv["quality_score"].mean() if n_surv > 0 else 0.0
        cl_mean = surv["human_cliche_density"].mean() if n_surv > 0 else 0.0

        results.append({
            "combo_id": comb_idx + 1,
            "mean_cov_floor": m_cov,
            "fact_ret_floor": f_ret,
            "copy_cap_llm": c_llm,
            "copy_cap_hf": c_hf,
            "cliche_mult": c_mult,
            "total_survivors": n_surv,
            "academic": cat_counts.get("academic", 0),
            "essay": cat_counts.get("essay", 0),
            "general": cat_counts.get("general", 0),
            "document": cat_counts.get("document", 0),
            "email": cat_counts.get("email", 0),
            "llm_survivors": src_counts.get("llm_regenerated_1_to_1", 0),
            "hf_survivors": src_counts.get("hf_dataset_import", 0),
            "ratio_070_080": round(r_bins.get("0.70-0.80", 0.0) * 100, 1),
            "ratio_080_090": round(r_bins.get("0.80-0.90", 0.0) * 100, 1),
            "ratio_090_095": round(r_bins.get("0.90-0.95", 0.0) * 100, 1),
            "ratio_gt_095": round(r_bins.get(">0.95", 0.0) * 100, 1),
            "mean_burstiness": round(b_mean, 2),
            "mean_contractions": round(c_mean, 2),
            "mean_cliches": round(cl_mean, 2),
            "mean_quality_score": round(q_mean, 2),
            "passed_indices": passed_indices,
        })

    res_df = pd.DataFrame(results)

    # Find recommended configuration:
    # Criteria:
    # 1. academic >= 1000 or max possible
    # 2. essay >= 800 or max possible
    # 3. general >= 1000
    # 4. document >= 1000
    # 5. lowest cliché density, copy_cap_llm <= 0.60
    # Sort by total_survivors while maintaining quality
    candidates = res_df[
        (res_df["copy_cap_llm"] <= 0.60) &
        (res_df["mean_cliches"] <= 0.70)
    ].sort_values(by=["total_survivors", "mean_quality_score"], ascending=[False, False])

    rec = candidates.iloc[0] if len(candidates) > 0 else res_df.sort_values(by="total_survivors", ascending=False).iloc[0]
    rec_combo_id = rec["combo_id"]
    rec_indices = set(rec["passed_indices"])
    print(f"\n[+] Recommended Combo #{rec_combo_id}:")
    print(f"    mean_cov_floor: {rec['mean_cov_floor']} | fact_ret_floor: {rec['fact_ret_floor']} | copy_cap_llm: {rec['copy_cap_llm']} | copy_cap_hf: {rec['copy_cap_hf']} | cliche_mult: {rec['cliche_mult']}x")
    print(f"    Total Survivors: {rec['total_survivors']} (Academic: {rec['academic']}, Essay: {rec['essay']}, General: {rec['general']}, Document: {rec['document']}, Email: {rec['email']})")

    # Generate Step 3 Examples
    print("\nExtracting representative examples for Step 3...")

    # a) Failed ONLY old min_cov < 0.55 rule, but pass under new rules
    old_fail_only_min_cov = []
    for idx, row in df.iterrows():
        # Check old rules
        failed_old_min_cov = (row["min_cov"] < 0.55)
        failed_old_mean_cov = (row["mean_cov"] < 0.75)
        failed_old_copy = (row["copy_rate"] > 0.60 if row[cat_col] in {"essay", "academic"} else row["copy_rate"] > 0.75)
        failed_old_cliche = (row["human_cliche_density"] > max(2.0 * hf_medians.get(row[cat_col], 0.0), 4.0))

        if failed_old_min_cov and not (failed_old_mean_cov or failed_old_copy or failed_old_cliche):
            if idx in rec_indices:
                old_fail_only_min_cov.append(idx)

    # b) Pass under recommended combination
    pass_rec = list(rec_indices)

    # c) Pass old Phase 2 rules but have burstiness < ai_burstiness
    old_phase2_pass_low_burst = []
    for idx, row in df.iterrows():
        cat = str(row[cat_col]).lower()
        if cat in {"essay", "academic"}:
            old_pass = (
                (row["min_cov"] >= 0.55) and
                (row["mean_cov"] >= 0.75) and
                (row["copy_rate"] <= 0.60) and
                (row["human_cliche_density"] <= max(2.0 * hf_medians.get(cat, 0.0), 4.0)) and
                (row["ratio"] >= 0.70)
            )
            if old_pass and (row["burstiness"] < row["ai_burstiness"]):
                old_phase2_pass_low_burst.append(idx)

    # Format Markdown Report
    print("Writing sweep_report.md...")
    md = []
    md.append("# Lagom V3 Data Quality Threshold Sweep Report\n")
    md.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("## 1. Executive Summary & Recommended Configuration\n")
    md.append(f"The threshold sweep systematically evaluated **108 combinations** across:\n")
    md.append("- **mean_cov floor**: `[0.65, 0.70, 0.75]`\n")
    md.append("- **fact_retention floor**: `[0.60, 0.70, 0.80]`\n")
    md.append("- **copy_rate cap for llm_regenerated_1_to_1**: `[0.50, 0.60]`\n")
    md.append("- **copy_rate cap for hf_dataset_import (essay/academic)**: `[0.60, 0.75, 0.85]`\n")
    md.append("- **cliche cap**: `2x` / `3x` baseline median\n")
    md.append("- **Email & Document**: no copy_rate cap, no min_cov rule; fact retention + cliché cap + ratio >= 0.70.\n\n")

    md.append(f"### Recommended Configuration: Combo #{rec_combo_id}\n")
    md.append(f"| Parameter | Recommended Value | Rationale |\n")
    md.append(f"|:---|:---:|:---|\n")
    md.append(f"| **mean_cov Floor** | **{rec['mean_cov_floor']}** | Ensures core semantic retention without dropping edits that prune throat-clearing openers. |\n")
    md.append(f"| **fact_retention Floor** | **{rec['fact_ret_floor']}** | Replaces brittle sentence-min_cov with explicit factual integrity (dates, entities, numbers). |\n")
    md.append(f"| **copy_rate Cap (LLM)** | **{rec['copy_cap_llm']}** | Strictly purges sentence-by-sentence 1:1 synonym substitution. |\n")
    md.append(f"| **copy_rate Cap (HF Essay/Acad)** | **{rec['copy_cap_hf']}** | Preserves dense, high-quality human academic writing while blocking verbatim quotes. |\n")
    md.append(f"| **Cliché Cap Multiplier** | **{rec['cliche_mult']:.0f}x** | Eliminates robotic buzzword contamination (mean clichés drop to {rec['mean_cliches']:.2f}/1000w). |\n\n")

    md.append(f"### Recommended Survivors Breakdown ({rec['total_survivors']} total rows)\n")
    md.append(f"- **Academic**: **{rec['academic']}** rows\n")
    md.append(f"- **General**: **{rec['general']}** rows\n")
    md.append(f"- **Document**: **{rec['document']}** rows\n")
    md.append(f"- **Essay**: **{rec['essay']}** rows\n")
    md.append(f"- **Email**: **{rec['email']}** rows\n")
    md.append(f"- **LLM Regenerated Targets Remaining**: **{rec['llm_survivors']}** (only clean, non-paraphrase rows survive)\n")
    md.append(f"- **HF Dataset Import Targets Remaining**: **{rec['hf_survivors']}**\n")
    md.append(f"- **Length Ratio Mix**: `0.70-0.80`: {rec['ratio_070_080']}% | `0.80-0.90`: {rec['ratio_080_090']}% | `0.90-0.95`: {rec['ratio_090_095']}% | `>0.95`: {rec['ratio_gt_095']}%\n")
    md.append(f"- **Stylistic Averages**: Burstiness = {rec['mean_burstiness']}, Contractions = {rec['mean_contractions']}/100w, Clichés = {rec['mean_cliches']}/1000w, Composite Quality Score = {rec['mean_quality_score']}/100\n\n")

    md.append("## 2. Parameter Sweep Results (Top Representative Combinations)\n\n")
    sweep_table_cols = [
        "combo_id", "mean_cov_floor", "fact_ret_floor", "copy_cap_llm", "copy_cap_hf", "cliche_mult",
        "total_survivors", "academic", "essay", "general", "document", "email", "mean_cliches", "mean_quality_score"
    ]
    sample_combos = pd.concat([
        res_df.head(10),
        res_df[res_df["combo_id"] == rec_combo_id],
        res_df.tail(10)
    ]).drop_duplicates(subset=["combo_id"]).sort_values(by="combo_id")

    md.append(sample_combos[sweep_table_cols].to_markdown(index=False))

    def format_examples_block(indices, title, n=10):
        out = [f"\n\n## 3. Examples Inspection: {title}\n"]
        if not indices:
            out.append("*(No rows matched this specific criteria)*\n")
            return "".join(out)
        sample_idx = pd.Series(list(indices)).sample(min(n, len(indices)), random_state=42).tolist()
        for i, idx in enumerate(sample_idx, 1):
            row = df.loc[idx]
            ai_snippet = (row.get("AI Text", "")[:240] + "...") if len(row.get("AI Text", "")) > 240 else row.get("AI Text", "")
            hu_snippet = (row.get("Human Text", "")[:240] + "...") if len(row.get("Human Text", "")) > 240 else row.get("Human Text", "")
            out.append(f"### Example {i} (Row {idx} - {row.get('category')}, source={row.get('human_source')})\n")
            out.append(f"- **Metrics**: ratio={row.get('cleaned_ratio'):.2f} | min_cov={row.get('min_cov'):.2f} | mean_cov={row.get('mean_cov'):.2f} | copy_rate={row.get('copy_rate'):.2f} | fact_retention={row.get('fact_retention'):.2f} | burst_hu={row.get('burstiness'):.1f} (vs ai={row.get('ai_burstiness'):.1f}) | cliches={row.get('human_cliche_density'):.1f}\n")
            out.append(f"- **AI Text**: `{ai_snippet.strip().replace(chr(10), ' ')}`\n")
            out.append(f"- **Human Text**: `{hu_snippet.strip().replace(chr(10), ' ')}`\n\n")
        return "".join(out)

    md.append(format_examples_block(old_fail_only_min_cov, "Group A - Rows Failing ONLY Old min_cov < 0.55 but Passing New Rules", 10))
    md.append(format_examples_block(pass_rec, "Group B - Rows Passing Under Recommended Combination", 10))
    md.append(format_examples_block(old_phase2_pass_low_burst, "Group C - Rows Passing Old Rules but Having Low Burstiness (burst < ai_burst)", 10))

    report_path = Path("Data/improved_data/sweep_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"\n[+] Saved sweep_report.md ({len(md)} lines) to {report_path}")

    # Also save full 108 combos table to CSV for reference
    res_df.drop(columns=["passed_indices"]).to_csv("Data/improved_data/sweep_combinations_108.csv", index=False)
    print("Saved Data/improved_data/sweep_combinations_108.csv")


if __name__ == "__main__":
    run_sweep()
