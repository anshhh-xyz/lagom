"""
Sentence-Level Audit Script for Lagom Humanizer Dataset (Phase 1)
Evaluates sentence-by-sentence coverage, copy rate, structural change,
style metrics (burstiness, contractions, cliches), artifacts, and length ratios.
"""

import os
import sys
import re
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch

# Ensure pysbd is available
try:
    import pysbd
    SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except ImportError:
    SEGMENTER = None

from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURATION DICT (All thresholds configurable via CLI or this dict)
# ==============================================================================
CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "copy_threshold": 0.90,          # Cosine similarity considered direct paraphrase
    "weak_match_threshold": 0.60,     # Below this, sentence matches no AI sentence strongly
    "rule_min_cov": 0.55,             # Candidate rule: drop if min_cov < 0.55
    "rule_mean_cov": 0.75,            # Candidate rule: drop if mean_cov < 0.75
    "rule_copy_rate_strict": 0.60,    # Candidate rule: drop if copy_rate > 0.60 (essay, academic)
    "rule_copy_rate_loose": 0.75,     # Candidate rule: drop if copy_rate > 0.75 (others)
    "rule_min_ratio": 0.70,           # Candidate rule: drop if human/ai word ratio < 0.70
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

# Compile regexes once
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
    r"(?i)(?:^|\n)\s*(?:[^\n]{0,80}\s+(?:Essay|Research Paper|Report)|(?:Title|Subject)\s*:)"
)
TRAILING_REF_PATTERN = re.compile(
    r"(?i)(?:\n\s*(?:Works Cited|References|Bibliography)[\s\S]*$)"
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using pysbd with fallback."""
    if not isinstance(text, str) or not text.strip():
        return []
    if SEGMENTER is not None:
        try:
            sents = SEGMENTER.segment(text)
            return [s.strip() for s in sents if s.strip()]
        except Exception:
            pass
    # Fallback to regex split on sentence terminals
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text.strip())
    return [s.strip() for s in sents if s.strip()]


def compute_style_metrics(human_text: str, ai_text: str, human_sents: list[str]):
    """Compute burstiness, contractions, and cliché density."""
    human_words = len(human_text.split())
    ai_words = len(ai_text.split())
    
    # 1. Ratio
    ratio = round(human_words / max(ai_words, 1), 4)

    # 2. Burstiness: sentence length standard deviation in words
    if len(human_sents) >= 2:
        lengths = [len(s.split()) for s in human_sents]
        burstiness = float(np.std(lengths))
    else:
        burstiness = 0.0

    # 3. Contraction count per 100 words in Human Text
    contractions = len(CONTRACTION_PATTERN.findall(human_text))
    contraction_rate = round((contractions / max(human_words, 1)) * 100.0, 2)

    # 4. Cliché density per 1000 words
    h_cliches = len(CLICHE_PATTERN.findall(human_text))
    h_cliche_density = round((h_cliches / max(human_words, 1)) * 1000.0, 2)

    a_cliches = len(CLICHE_PATTERN.findall(ai_text))
    a_cliche_density = round((a_cliches / max(ai_words, 1)) * 1000.0, 2)

    return ratio, burstiness, contraction_rate, h_cliche_density, a_cliche_density


def check_artifacts(text: str):
    """Check for citation, title leakage, and trailing reference artifacts."""
    if not isinstance(text, str):
        return False, False, False
    has_citation = bool(CITATION_PATTERN.search(text))
    has_title = bool(TITLE_LEAK_PATTERN.search(text))
    has_refs = bool(TRAILING_REF_PATTERN.search(text))
    return has_citation, has_title, has_refs


def audit_row_embeddings(
    ai_sents: list[str],
    human_sents: list[str],
    ai_embs: np.ndarray,
    human_embs: np.ndarray,
    copy_thresh: float,
    weak_thresh: float,
):
    """Compute min_cov, mean_cov, copy_rate, sent_count_ratio, struct_change from normalized embeddings."""
    n_ai = len(ai_sents)
    n_human = len(human_sents)

    if n_ai == 0 or n_human == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Cosine similarity matrix (since embeddings are L2 normalized: dot product = cosine similarity)
    # Shape: [n_ai, n_human]
    sim_matrix = np.dot(ai_embs, human_embs.T)

    # For each AI sentence, find best-matching human sentence
    best_human_for_ai = np.max(sim_matrix, axis=1) # shape [n_ai]
    min_cov = float(np.min(best_human_for_ai))
    mean_cov = float(np.mean(best_human_for_ai))
    copy_rate = float(np.mean(best_human_for_ai > copy_thresh))

    # Sentence count ratio
    sent_count_ratio = round(n_human / max(n_ai, 1), 3)

    # Structural change:
    # For each human sentence, find best-matching AI sentence
    best_ai_for_human = np.argmax(sim_matrix, axis=0) # shape [n_human]
    best_ai_scores = np.max(sim_matrix, axis=0)        # shape [n_human]

    # Count occurrences of best-matching AI sentences among human sentences
    ai_target_counts = Counter(best_ai_for_human)

    # A human sentence is structurally altered if:
    # 1. Multiple human sentences map to the same AI sentence (split), OR
    # 2. It matches no AI sentence strongly (score < weak_thresh)
    changed_human_count = 0
    for j in range(n_human):
        mapped_ai = best_ai_for_human[j]
        score = best_ai_scores[j]
        if ai_target_counts[mapped_ai] > 1 or score < weak_thresh:
            changed_human_count += 1

    struct_change = float(changed_human_count / max(n_human, 1))

    return round(min_cov, 4), round(mean_cov, 4), round(copy_rate, 4), sent_count_ratio, round(struct_change, 4)


def process_split(
    df: pd.DataFrame,
    split_name: str,
    model: SentenceTransformer,
    cfg: dict,
) -> pd.DataFrame:
    """Audit all rows in a given split dataframe."""
    print(f"\n[*] Processing split '{split_name}' ({len(df)} rows)...", flush=True)

    # 1. Split all texts into sentences
    print("  -> Segmenting sentences with pysbd...", flush=True)
    all_ai_sents = []
    all_human_sents = []
    style_rows = []
    artifact_rows = []

    for idx, row in df.iterrows():
        ai_txt = str(row.get("AI Text", "")).strip()
        hu_txt = str(row.get("Human Text", "")).strip()

        ai_s = split_sentences(ai_txt)
        hu_s = split_sentences(hu_txt)
        all_ai_sents.append(ai_s)
        all_human_sents.append(hu_s)

        # Style metrics
        ratio, burst, contr, h_cliche, a_cliche = compute_style_metrics(hu_txt, ai_txt, hu_s)
        style_rows.append({
            "ratio": ratio,
            "burstiness": burst,
            "contraction_rate": contr,
            "human_cliche_density": h_cliche,
            "ai_cliche_density": a_cliche,
            "ai_words": len(ai_txt.split()),
            "human_words": len(hu_txt.split()),
            "ai_sents_count": len(ai_s),
            "human_sents_count": len(hu_s),
        })

        # Artifact checks
        has_cite, has_title, has_refs = check_artifacts(hu_txt)
        artifact_rows.append({
            "has_citation": has_cite,
            "has_title_leakage": has_title,
            "has_trailing_references": has_refs,
        })

    # 2. Gather unique sentences to embed efficiently in batches
    print("  -> Embedding sentences with MiniLM-L6-v2 on GPU/CPU...", flush=True)
    # Map sentence text to unique ID for deduped fast encoding
    unique_sents = list(set([s for sents in all_ai_sents for s in sents] + [s for sents in all_human_sents for s in sents]))
    print(f"     Total sentences: {sum(len(s) for s in all_ai_sents) + sum(len(s) for s in all_human_sents)}, Unique: {len(unique_sents)}", flush=True)
    
    t0 = time.time()
    embeddings = model.encode(
        unique_sents,
        batch_size=cfg["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
        device=cfg["device"],
    )
    sent2emb = {s: embeddings[i] for i, s in enumerate(unique_sents)}
    print(f"     Embeddings complete in {time.time() - t0:.1f}s", flush=True)

    # 3. Compute sentence-level audit metrics per row
    print("  -> Calculating sentence coverage and structural change...", flush=True)
    audit_rows = []
    copy_thresh = cfg["copy_threshold"]
    weak_thresh = cfg["weak_match_threshold"]

    for i in range(len(df)):
        ai_s = all_ai_sents[i]
        hu_s = all_human_sents[i]

        if len(ai_s) == 0 or len(hu_s) == 0:
            min_cov, mean_cov, copy_rate, sc_ratio, st_change = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            ai_embs = np.array([sent2emb[s] for s in ai_s])
            hu_embs = np.array([sent2emb[s] for s in hu_s])
            min_cov, mean_cov, copy_rate, sc_ratio, st_change = audit_row_embeddings(
                ai_s, hu_s, ai_embs, hu_embs, copy_thresh, weak_thresh
            )

        audit_rows.append({
            "min_cov": min_cov,
            "mean_cov": mean_cov,
            "copy_rate": copy_rate,
            "sent_count_ratio": sc_ratio,
            "struct_change": st_change,
        })

    # Build output dataframe
    res_df = df.copy()
    res_df["split"] = split_name
    for k, v in pd.DataFrame(style_rows).items():
        res_df[k] = v
    for k, v in pd.DataFrame(artifact_rows).items():
        res_df[k] = v
    for k, v in pd.DataFrame(audit_rows).items():
        res_df[k] = v

    return res_df


def print_audit_report(df: pd.DataFrame, cfg: dict):
    """Generate and print the complete audit report grouped by category x human_source."""
    print("\n" + "=" * 90)
    print("             LAGOM DATASET SENTENCE-LEVEL AUDIT REPORT (PHASE 1)")
    print("=" * 90)

    # Ensure required columns
    cat_col = "category" if "category" in df.columns else "domain"
    source_col = "human_source" if "human_source" in df.columns else "source"

    # Define ratio bins
    def get_ratio_bin(r):
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

    df["ratio_bin"] = df["ratio"].apply(get_ratio_bin)

    # 1. High-level Summary
    print(f"\n[1] Overall Statistics: {len(df)} total rows across splits:")
    print(df["split"].value_counts().to_string())

    # 2. Grouped percentiles
    metrics = [
        "ratio", "min_cov", "mean_cov", "copy_rate",
        "sent_count_ratio", "struct_change", "burstiness",
        "contraction_rate", "human_cliche_density", "ai_cliche_density"
    ]

    print("\n[2] Metric Percentiles (10 / 25 / 50 / 75 / 90) Grouped by Category x Human Source:")
    groups = df.groupby([cat_col, source_col])

    for (cat, src), grp in groups:
        print(f"\n--- Category: {cat.upper()} | Source: {src} (N = {len(grp)}) ---")
        percentiles = grp[metrics].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).T
        percentiles.columns = ["p10", "p25", "p50 (med)", "p75", "p90"]
        print(percentiles.round(3).to_string())

        # Length ratio breakdown
        ratio_counts = grp["ratio_bin"].value_counts(normalize=True) * 100.0
        bin_order = ["<0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", ">0.95"]
        breakdown_str = ", ".join([f"{b}: {ratio_counts.get(b, 0.0):.1f}%" for b in bin_order])
        print(f"  Length Ratio Mix: {breakdown_str}")

    # 3. Reference Human Cliche Medians for hf_dataset_import
    print("\n[3] Baseline Human Cliche Medians (hf_dataset_import):")
    hf_mask = df[source_col].astype(str).str.contains("hf_dataset_import", case=False, na=False)
    hf_medians = df[hf_mask].groupby(cat_col)["human_cliche_density"].median().to_dict()
    for cat, med in hf_medians.items():
        print(f"  {cat}: median {med:.2f} cliches/1000 words -> 2x threshold = {2 * med:.2f}")

    # 4. Impact of Candidate Filter Rules
    print("\n[4] Candidate Filter Rule Evaluation (Per Row Analysis):")
    rule_results = []
    
    strict_cats = {"essay", "academic"}

    for idx, row in df.iterrows():
        cat = str(row[cat_col]).lower()
        src = str(row[source_col])

        # Rule 1: dropped idea or poor coverage
        drop_rule1 = (row["min_cov"] < cfg["rule_min_cov"]) or (row["mean_cov"] < cfg["rule_mean_cov"])

        # Rule 2: copy rate too high
        thresh_copy = cfg["rule_copy_rate_strict"] if cat in strict_cats else cfg["rule_copy_rate_loose"]
        drop_rule2 = (row["copy_rate"] > thresh_copy)

        # Rule 3: human cliche density > 2x hf median
        hf_med = hf_medians.get(cat, 5.0)
        drop_rule3 = (row["human_cliche_density"] > (2.0 * hf_med))

        # Rule 4: ratio < 0.70
        drop_rule4 = (row["ratio"] < cfg["rule_min_ratio"])

        # Rule 5: artifacts
        drop_rule5 = row["has_title_leakage"] or row["has_trailing_references"]

        # Combined drop (excluding or including llm_regenerated)
        any_fail = drop_rule1 or drop_rule2 or drop_rule3 or drop_rule4 or drop_rule5
        is_llm_regen = "llm_regenerated" in src

        rule_results.append({
            "drop_rule1_cov": drop_rule1,
            "drop_rule2_copy": drop_rule2,
            "drop_rule3_cliche": drop_rule3,
            "drop_rule4_ratio": drop_rule4,
            "drop_rule5_artifact": drop_rule5,
            "any_fail": any_fail,
            "is_llm_regen": is_llm_regen,
        })

    rule_df = pd.DataFrame(rule_results)
    total_n = len(df)
    print(f"  Rule 1 (min_cov < {cfg['rule_min_cov']} or mean_cov < {cfg['rule_mean_cov']}): {rule_df['drop_rule1_cov'].sum()} rows ({rule_df['drop_rule1_cov'].mean()*100:.1f}%)")
    print(f"  Rule 2 (copy_rate > {cfg['rule_copy_rate_strict']}/{cfg['rule_copy_rate_loose']}):       {rule_df['drop_rule2_copy'].sum()} rows ({rule_df['drop_rule2_copy'].mean()*100:.1f}%)")
    print(f"  Rule 3 (human cliches > 2x hf median):          {rule_df['drop_rule3_cliche'].sum()} rows ({rule_df['drop_rule3_cliche'].mean()*100:.1f}%)")
    print(f"  Rule 4 (ratio < {cfg['rule_min_ratio']}):                         {rule_df['drop_rule4_ratio'].sum()} rows ({rule_df['drop_rule4_ratio'].mean()*100:.1f}%)")
    print(f"  Rule 5 (title leakage / trailing refs):         {rule_df['drop_rule5_artifact'].sum()} rows ({rule_df['drop_rule5_artifact'].mean()*100:.1f}%)")
    print(f"  TOTAL ROWS DROPPED (Union of all rules):        {rule_df['any_fail'].sum()} rows ({rule_df['any_fail'].mean()*100:.1f}%)")
    print(f"  TOTAL ROWS SURVIVING:                           {(~rule_df['any_fail']).sum()} rows ({(~rule_df['any_fail']).mean()*100:.1f}%)")

    # Essay llm_regenerated survival check
    essay_llm = df[(df[cat_col] == "essay") & (df[source_col].astype(str).str.contains("llm_regenerated"))]
    if len(essay_llm) > 0:
        essay_llm_survived = (~rule_df.loc[essay_llm.index, "any_fail"]).sum()
        print(f"\n  [Essay Specific] llm_regenerated_1_to_1 Essays: {len(essay_llm)} total -> {essay_llm_survived} survive filters ({essay_llm_survived/len(essay_llm)*100:.1f}%)")

    # 5. Dump 10 Random Examples for Each of the 4 Target Bins
    print("\n" + "=" * 90)
    print("                     INSPECTION OF REPRESENTATIVE BINS (10 EXAMPLES EACH)")
    print("=" * 90)

    # Bin 1: copy_rate > 0.8
    b1 = df[df["copy_rate"] > 0.80]
    dump_bin_examples("BIN 1: HEAVY PARAPHRASE / COPYING (copy_rate > 0.80)", b1, 10)

    # Bin 2: min_cov < 0.55
    b2 = df[df["min_cov"] < 0.55]
    dump_bin_examples("BIN 2: DROPPED CONTENT / POOR COVERAGE (min_cov < 0.55)", b2, 10)

    # Bin 3: Cliche density in top 10%
    cliche_p90 = df["human_cliche_density"].quantile(0.90)
    b3 = df[df["human_cliche_density"] >= cliche_p90]
    dump_bin_examples(f"BIN 3: HEAVY AI CLICHÉ DENSITY (Top 10% >= {cliche_p90:.1f}/1000 words)", b3, 10)

    # Bin 4: The "Good" Bin (mean_cov 0.78-0.93, copy_rate < 0.50)
    b4 = df[(df["mean_cov"] >= 0.78) & (df["mean_cov"] <= 0.93) & (df["copy_rate"] < 0.50) & (df["ratio"] >= 0.70)]
    dump_bin_examples("BIN 4: THE 'GOOD' HUMANIZED BIN (mean_cov 0.78-0.93, copy_rate < 0.50, ratio >= 0.70)", b4, 10)


def dump_bin_examples(title: str, sub_df: pd.DataFrame, n: int = 10):
    print(f"\n>>> {title} (Total pool: {len(sub_df)} rows) <<<")
    if len(sub_df) == 0:
        print("  [No rows found in this bin]")
        return

    sample = sub_df.sample(min(n, len(sub_df)), random_state=42)
    for i, (idx, row) in enumerate(sample.iterrows(), 1):
        ai_snippet = (row.get("AI Text", "")[:280] + "...") if len(row.get("AI Text", "")) > 280 else row.get("AI Text", "")
        hu_snippet = (row.get("Human Text", "")[:280] + "...") if len(row.get("Human Text", "")) > 280 else row.get("Human Text", "")
        print(f"\n  [{i}] Row {idx} ({row.get('category')}, source={row.get('human_source')}):")
        print(f"      Metrics: ratio={row.get('ratio'):.2f} | min_cov={row.get('min_cov'):.2f} | mean_cov={row.get('mean_cov'):.2f} | copy_rate={row.get('copy_rate'):.2f} | burst={row.get('burstiness'):.1f} | cliches={row.get('human_cliche_density'):.1f}")
        print(f"      AI Text   : {ai_snippet.strip()}")
        print(f"      Human Text: {hu_snippet.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Lagom Sentence-Level Dataset Audit (Phase 1)")
    parser.add_argument("--data-dir", type=str, default="Data/improved_data", help="Directory containing normalized CSVs")
    parser.add_argument("--output-csv", type=str, default="Data/improved_data/audit_scores.csv", help="Where to save audited scores")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--copy-threshold", type=float, default=CONFIG["copy_threshold"])
    parser.add_argument("--weak-threshold", type=float, default=CONFIG["weak_match_threshold"])
    parser.add_argument("--min-cov", type=float, default=CONFIG["rule_min_cov"])
    parser.add_argument("--mean-cov", type=float, default=CONFIG["rule_mean_cov"])
    parser.add_argument("--copy-strict", type=float, default=CONFIG["rule_copy_rate_strict"])
    parser.add_argument("--copy-loose", type=float, default=CONFIG["rule_copy_rate_loose"])
    parser.add_argument("--min-ratio", type=float, default=CONFIG["rule_min_ratio"])
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["batch_size"] = args.batch_size
    cfg["copy_threshold"] = args.copy_threshold
    cfg["weak_match_threshold"] = args.weak_threshold
    cfg["rule_min_cov"] = args.min_cov
    cfg["rule_mean_cov"] = args.mean_cov
    cfg["rule_copy_rate_strict"] = args.copy_strict
    cfg["rule_copy_rate_loose"] = args.copy_loose
    cfg["rule_min_ratio"] = args.min_ratio

    print("=" * 80)
    print("Initializing Sentence Transformer:", cfg["model_name"])
    print("Device:", cfg["device"])
    print("=" * 80)
    model = SentenceTransformer(cfg["model_name"], device=cfg["device"])

    data_dir = Path(args.data_dir)
    splits = ["train", "validation", "test"]
    processed_dfs = []

    for split in splits:
        p = data_dir / f"{split}_normalized.csv"
        if not p.exists():
            print(f"[!] Warning: {p} not found, skipping...")
            continue
        print(f"Reading {p}...")
        df = pd.read_csv(p)
        audited_df = process_split(df, split, model, cfg)
        processed_dfs.append(audited_df)

    if not processed_dfs:
        print("[!] No datasets found to process.")
        sys.exit(1)

    full_audit_df = pd.concat(processed_dfs, ignore_index=True)
    out_path = Path(args.output_csv)
    full_audit_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[+] Full audit saved to {out_path} ({len(full_audit_df)} rows, {os.path.getsize(out_path):,} bytes).")

    # Generate and print comprehensive report
    print_audit_report(full_audit_df, cfg)


if __name__ == "__main__":
    main()
