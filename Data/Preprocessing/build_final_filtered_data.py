"""
Final Data Filtering Script for Lagom Humanizer (Phase 2 - Approved Settings)
Builds train_filtered.csv and validation_filtered.csv with:
- Combo #5 parameters: mean_cov >= 0.65, fact_retention >= 0.60, copy_cap_llm = 0.50, copy_cap_hf = 0.85
- Hard length bounds: ratio in [0.70, 0.95]
- Opening rules: drop lowercase start, drop fragment (no verb in 1st sentence), drop opening mismatch (cos < 0.40)
- Expanded stem-matched cliché detection
- 15% cap on llm_regenerated_1_to_1 per category (prioritizing highest quality_score)
- target_words = round(human_words, -1)
- Never mix train and validation rows. Leave test_normalized.csv untouched.
"""

import sys
import os
import io
import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import nltk
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import pysbd
    SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except ImportError:
    SEGMENTER = None

TITLE_LEAK_PATTERN = re.compile(
    r"(?i)^(?:[^\n]{0,80}\s+(?:Essay|Research Paper|Report)|(?:Title|Subject)\s*:[^\n]*)\n+"
)
TRAILING_REF_PATTERN = re.compile(
    r"(?i)\n\s*(?:Works Cited|References|Bibliography)[\s\S]*$"
)
CITATION_PATTERN = re.compile(r"\([A-Z][^)]*\d{4}\)")

CLICHE_STEMS = [
    r"navigat\w*", r"rigor\w*", r"furnish\w*", r"accord\w*", r"labor\w*",
    r"endeavor\w*", r"meticul\w*", r"comprehens\w*", r"leverag\w*", r"robust\w*",
    r"nuanc\w*", r"intricat\w*", r"delv\w*", r"underscor\w*", r"pivot\w*",
    r"tapestr\w*", r"realm\w*", r"furthermor\w*", r"moreov\w*", r"consequent\w*",
    r"in conclusion", r"multifacet\w*", r"paramount\w*", r"testament\w*",
    r"foster\w*", r"harness\w*", r"crucial\w*", r"profound\w*", r"symbio\w*",
    r"holist\w*", r"unprecedent\w*", r"paradigm\w*", r"catalyst\w*",
    r"transform\w*", r"seamless\w*", r"vibrant\w*", r"ecosystem\w*",
    r"synerg\w*", r"steadfast\w*", r"cornerston\w*", r"ultimat\w*"
]
CLICHE_STEM_PATTERN = re.compile(r"\b(" + "|".join(CLICHE_STEMS) + r")\b", re.IGNORECASE)
CONTRACTION_PATTERN = re.compile(r"\b([a-zA-Z]+)('re|'ve|'d|'ll|'t|'s|n't)\b", re.IGNORECASE)


def strip_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    cleaned = TITLE_LEAK_PATTERN.sub("", cleaned).strip()
    cleaned = TRAILING_REF_PATTERN.sub("", cleaned).strip()
    cleaned = CITATION_PATTERN.sub("", cleaned).strip()
    return re.sub(r"[ \t]+", " ", cleaned)


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if SEGMENTER is not None:
        try:
            return [s.strip() for s in SEGMENTER.segment(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text.strip()) if s.strip()]


def has_verb_in_sentence(sent: str) -> bool:
    """Check if the sentence contains at least one verb tag."""
    tokens = nltk.word_tokenize(sent)
    if not tokens:
        return False
    tagged = nltk.pos_tag(tokens)
    return any(tag.startswith("VB") for word, tag in tagged)


def compute_expanded_cliches(text: str) -> float:
    words = len(text.split())
    if words == 0:
        return 0.0
    matches = len(CLICHE_STEM_PATTERN.findall(text))
    return round((matches / words) * 1000.0, 2)


def process_and_filter(
    df: pd.DataFrame,
    emb_model: SentenceTransformer,
    split_name: str,
    hf_medians: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Process, evaluate, clean artifacts, and apply Combo #5 + Opening/Fragment rules."""
    print(f"\n=======================================================")
    print(f"  Processing {split_name.upper()} ({len(df)} initial rows)")
    print(f"=======================================================")

    cat_col = "category"
    source_col = "human_source"

    cleaned_humans = []
    cleaned_ratios = []
    cleaned_h_words = []
    cleaned_bursts = []
    cleaned_contrs = []
    cleaned_cliches = []
    target_words_list = []

    drop_reasons = []
    passed_mask = []
    dropped_only_opening_rule = []

    # Pre-extract AI and Human sentences
    all_hu_sents = []
    all_ai_sents = []
    for idx, row in df.iterrows():
        ai_t = str(row.get("AI Text", "")).strip()
        hu_t = str(row.get("Human Text", "")).strip()
        cleaned_hu = strip_artifacts(hu_t)

        h_sents = split_sentences(cleaned_hu)
        ai_sents = split_sentences(ai_t)
        all_hu_sents.append(h_sents)
        all_ai_sents.append(ai_sents)

        hw = len(cleaned_hu.split())
        aw = len(ai_t.split())
        ratio = round(hw / max(aw, 1), 4)

        burst = float(np.std([len(s.split()) for s in h_sents])) if len(h_sents) >= 2 else 0.0
        contr = round((len(CONTRACTION_PATTERN.findall(cleaned_hu)) / max(hw, 1)) * 100.0, 2)
        cliche = compute_expanded_cliches(cleaned_hu)
        target_w = int(round(hw, -1))

        cleaned_humans.append(cleaned_hu)
        cleaned_ratios.append(ratio)
        cleaned_h_words.append(hw)
        cleaned_bursts.append(burst)
        cleaned_contrs.append(contr)
        cleaned_cliches.append(cliche)
        target_words_list.append(target_w)

    print("Checking opening mismatch with MiniLM-L6-v2 embeddings on GPU...")
    # Gather first human sentences and AI sentences
    first_hu_sents = [sents[0] if sents else "" for sents in all_hu_sents]
    # Encode unique first human sentences
    unique_first_hu = list(set([s for s in first_hu_sents if s]))
    hu1_embs = emb_model.encode(unique_first_hu, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    hu1_dict = {s: hu1_embs[i] for i, s in enumerate(unique_first_hu)}

    # Encode unique AI sentences
    unique_ai_sents = list(set([s for sents in all_ai_sents for s in sents]))
    ai_embs = emb_model.encode(unique_ai_sents, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    ai_dict = {s: ai_embs[i] for i, s in enumerate(unique_ai_sents)}

    print("Applying Combo #5 settings and new opening/fragment rules...")
    for idx_num, (orig_idx, row) in enumerate(df.iterrows()):
        cat = str(row[cat_col]).lower()
        src = str(row[source_col])
        cleaned_hu = cleaned_humans[idx_num]
        h_sents = all_hu_sents[idx_num]
        ai_sents = all_ai_sents[idx_num]
        ratio = cleaned_ratios[idx_num]
        cliche = cleaned_cliches[idx_num]
        f_ret = float(row.get("fact_retention", 1.0))
        m_cov = float(row.get("mean_cov", 0.0))
        c_rate = float(row.get("copy_rate", 0.0))

        reasons = []

        # Check 1: Length ratio bounds [0.70, 0.95]
        if ratio < 0.70 or ratio > 0.95:
            reasons.append(f"ratio_outside_bounds({ratio:.2f})")

        # Check 2: Cliché cap (2x median, floor 4.0)
        hf_med = hf_medians.get(cat, 0.0)
        c_thresh = max(2.0 * hf_med, 4.0)
        if cliche > c_thresh:
            reasons.append(f"high_cliches({cliche:.1f}>{c_thresh:.1f})")

        # Check 3: Fact retention floor (0.60)
        if f_ret < 0.60:
            reasons.append(f"low_fact_retention({f_ret:.2f}<0.60)")

        # Check 4: Category coverage and copy rate rules
        if cat in {"email", "document"}:
            # Relaxed
            pass
        elif cat in {"essay", "academic"}:
            if m_cov < 0.65:
                reasons.append(f"low_mean_cov({m_cov:.2f}<0.65)")
            if "llm_regenerated" in src:
                if c_rate > 0.50:
                    reasons.append(f"high_copy_llm({c_rate:.2f}>0.50)")
            else:
                if c_rate > 0.85:
                    reasons.append(f"high_copy_hf({c_rate:.2f}>0.85)")
        else:  # general
            if m_cov < 0.65:
                reasons.append(f"low_mean_cov({m_cov:.2f}<0.65)")
            if "llm_regenerated" in src:
                if c_rate > 0.50:
                    reasons.append(f"high_copy_llm({c_rate:.2f}>0.50)")
            else:
                if c_rate > 0.85:
                    reasons.append(f"high_copy_hf({c_rate:.2f}>0.85)")

        combo5_fail = len(reasons) > 0

        # Check 5: NEW OPENING & FRAGMENT RULES
        opening_reasons = []
        if not cleaned_hu:
            opening_reasons.append("empty_human_text")
        else:
            # 5a. Starts with lowercase letter
            first_char = cleaned_hu[0]
            if first_char.isalpha() and first_char.islower():
                opening_reasons.append(f"starts_lowercase('{first_char}')")

            if not h_sents:
                opening_reasons.append("no_human_sentences")
            else:
                first_h = h_sents[0]
                # 5b. Sentence fragment (no verb in first sentence)
                if not has_verb_in_sentence(first_h):
                    opening_reasons.append("first_sentence_fragment_no_verb")

                # 5c. Opening mismatch (best matching AI sentence cosine < 0.40)
                if first_h in hu1_dict and ai_sents:
                    h1_vec = hu1_dict[first_h]
                    ai_vecs = np.array([ai_dict[s] for s in ai_sents if s in ai_dict])
                    if len(ai_vecs) > 0:
                        opening_cos = float(np.max(np.dot(ai_vecs, h1_vec)))
                        if opening_cos < 0.40:
                            opening_reasons.append(f"opening_mismatch_cos({opening_cos:.2f}<0.40)")

        # Record if dropped ONLY by opening rule
        if not combo5_fail and len(opening_reasons) > 0:
            dropped_only_opening_rule.append({
                "row_index": orig_idx,
                "category": cat,
                "source": src,
                "opening_reason": "; ".join(opening_reasons),
                "ratio": ratio,
                "first_human_sentence": h_sents[0] if h_sents else "",
                "AI Text": row.get("AI Text", ""),
                "Human Text": cleaned_hu,
            })

        reasons.extend(opening_reasons)

        if reasons:
            drop_reasons.append("; ".join(reasons))
            passed_mask.append(False)
        else:
            drop_reasons.append("PASSED")
            passed_mask.append(True)

    annotated_df = df.copy()
    annotated_df["Human Text"] = cleaned_humans
    annotated_df["human_words"] = cleaned_h_words
    annotated_df["ratio"] = cleaned_ratios
    annotated_df["burstiness"] = cleaned_bursts
    annotated_df["contraction_rate"] = cleaned_contrs
    annotated_df["human_cliche_density"] = cleaned_cliches
    annotated_df["target_words"] = target_words_list
    annotated_df["filter_decision"] = drop_reasons

    surviving_df = annotated_df[passed_mask].copy()
    print(f"  Rows passing all quality + opening rules: {len(surviving_df)} / {len(df)} ({(len(surviving_df)/len(df))*100:.1f}%)")

    # Step 5: Cap llm_regenerated_1_to_1 at 15% per category
    print("\nApplying 15% cap on llm_regenerated_1_to_1 per category (prioritizing quality_score)...")
    final_category_dfs = []
    for cat, grp in surviving_df.groupby(cat_col):
        llm_rows = grp[grp[source_col].astype(str).str.contains("llm_regenerated")]
        non_llm_rows = grp[~grp[source_col].astype(str).str.contains("llm_regenerated")]

        n_non_llm = len(non_llm_rows)
        # 15% cap: n_llm / (n_non_llm + n_llm) <= 0.15 => n_llm <= (0.15 / 0.85) * n_non_llm
        max_llm = int(np.floor((0.15 / 0.85) * n_non_llm)) if n_non_llm > 0 else 0

        if len(llm_rows) > max_llm:
            # Keep top max_llm with highest quality_score
            selected_llm = llm_rows.sort_values(by="quality_score", ascending=False).head(max_llm)
            print(f"  {cat.upper()}: Capped LLM from {len(llm_rows)} to {max_llm} (Non-LLM: {n_non_llm}, LLM share: {len(selected_llm)/(n_non_llm+len(selected_llm))*100:.1f}%)")
        else:
            selected_llm = llm_rows
            total_cat = n_non_llm + len(selected_llm)
            share = (len(selected_llm) / total_cat * 100) if total_cat > 0 else 0
            print(f"  {cat.upper()}: Kept all {len(llm_rows)} LLM rows (Non-LLM: {n_non_llm}, LLM share: {share:.1f}%)")

        final_category_dfs.append(pd.concat([non_llm_rows, selected_llm], ignore_index=True))

    final_df = pd.concat(final_category_dfs, ignore_index=True)
    print(f"  Total Final Filtered Rows for {split_name.upper()}: {len(final_df)} rows")

    return final_df, annotated_df, dropped_only_opening_rule


def main():
    print("=" * 80)
    print("  LAGOM V3: BUILDING FINAL FILTERED DATASETS (TRAIN & VALIDATION)")
    print("=" * 80)

    # 1. Load normalized source files
    train_path = Path("Data/improved_data/train_normalized.csv")
    val_path = Path("Data/improved_data/validation_normalized.csv")
    audit_path = Path("Data/improved_data/audit_scores.csv")

    print("Loading audit_scores.csv...")
    audit_df = pd.read_csv(audit_path)

    # Separate train and validation from audit_scores strictly by split
    train_raw = audit_df[audit_df["split"] == "train"].copy()
    val_raw = audit_df[audit_df["split"] == "validation"].copy()
    print(f"Audit split rows: Train = {len(train_raw)}, Validation = {len(val_raw)}")

    # 2. Compute baseline hf cliché medians
    hf_mask = audit_df["human_source"].astype(str).str.contains("hf_dataset_import", case=False, na=False)
    hf_medians = audit_df[hf_mask].groupby("category")["human_cliche_density"].median().to_dict()

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    # 3. Process Train
    train_filtered, train_annotated, train_opening_dropped = process_and_filter(
        train_raw, emb_model, "train", hf_medians
    )
    train_out_path = Path("Data/improved_data/train_filtered.csv")
    train_filtered.to_csv(train_out_path, index=False, encoding="utf-8")
    print(f"[+] Saved {train_out_path} ({len(train_filtered)} rows)")

    # 4. Process Validation
    val_filtered, val_annotated, val_opening_dropped = process_and_filter(
        val_raw, emb_model, "validation", hf_medians
    )
    val_out_path = Path("Data/improved_data/validation_filtered.csv")
    val_filtered.to_csv(val_out_path, index=False, encoding="utf-8")
    print(f"[+] Saved {val_out_path} ({len(val_filtered)} rows)")

    # 5. Generate Comprehensive Markdown Report
    print("\nGenerating final filter report markdown...")
    all_annotated = pd.concat([train_annotated, val_annotated], ignore_index=True)
    all_final = pd.concat([train_filtered, val_filtered], ignore_index=True)
    all_opening_dropped = train_opening_dropped + val_opening_dropped

    def get_ratio_bin(r):
        if r < 0.70: return "<0.70"
        elif 0.70 <= r < 0.80: return "0.70-0.80"
        elif 0.80 <= r < 0.90: return "0.80-0.90"
        elif 0.90 <= r <= 0.95: return "0.90-0.95"
        else: return ">0.95"

    all_final["ratio_bin"] = all_final["ratio"].apply(get_ratio_bin)

    md = []
    md.append("# Lagom V3 Final Quality-Filtered Dataset Report\n")
    md.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("## 1. Executive Summary\n")
    md.append("- **Train Final Rows**: **`" + str(len(train_filtered)) + "`** (built strictly from `train_normalized.csv`).\n")
    md.append("- **Validation Final Rows**: **`" + str(len(val_filtered)) + "`** (built strictly from `validation_normalized.csv`).\n")
    md.append("- **Test Set**: [test_normalized.csv](file:///d:/Projects/Lagom/Data/improved_data/test_normalized.csv) **remains 100% untouched**.\n")
    md.append("- **Length Control**: Strictly restricted to ratio **`[0.70, 0.95]`**. `target_words = round(human_words, -1)` appended.\n")
    md.append("- **LLM Cap**: `llm_regenerated_1_to_1` capped at **$\le 15\%$** per category, prioritizing highest `quality_score`.\n\n")

    md.append("## 2. Before vs. After Breakdown (Category x Human Source)\n\n")
    cat_src_summary = []
    for (cat, src), grp_b in all_annotated.groupby(["category", "human_source"]):
        grp_a = all_final[(all_final["category"] == cat) & (all_final["human_source"] == src)]
        b_cnt = len(grp_b)
        a_cnt = len(grp_a)
        retention = f"{(a_cnt / b_cnt) * 100:.1f}%" if b_cnt > 0 else "0.0%"

        b_ratio = grp_b["ratio"].mean()
        a_ratio = grp_a["ratio"].mean() if a_cnt > 0 else 0.0

        b_burst = grp_b["burstiness"].mean()
        a_burst = grp_a["burstiness"].mean() if a_cnt > 0 else 0.0

        b_contr = grp_b["contraction_rate"].mean()
        a_contr = grp_a["contraction_rate"].mean() if a_cnt > 0 else 0.0

        b_cliche = grp_b["human_cliche_density"].mean()
        a_cliche = grp_a["human_cliche_density"].mean() if a_cnt > 0 else 0.0

        cat_src_summary.append({
            "Category": cat,
            "Source": src,
            "Initial Rows": b_cnt,
            "Filtered Rows": a_cnt,
            "Retention": retention,
            "Ratio (B->A)": f"{b_ratio:.2f}->{a_ratio:.2f}",
            "Burst (B->A)": f"{b_burst:.1f}->{a_burst:.1f}",
            "Contr (B->A)": f"{b_contr:.2f}->{a_contr:.2f}",
            "Cliches (B->A)": f"{b_cliche:.1f}->{a_cliche:.1f}",
        })
    md.append(pd.DataFrame(cat_src_summary).to_markdown(index=False))

    md.append("\n\n## 3. Length Ratio Mix in Final Set\n\n")
    mix_table = []
    r_counts = all_final["ratio_bin"].value_counts(normalize=True) * 100.0
    for b in ["<0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", ">0.95"]:
        mix_table.append({
            "Ratio Bin": b,
            "Share (%)": f"{r_counts.get(b, 0.0):.1f}%",
            "Row Count": (all_final['ratio_bin'] == b).sum()
        })
    md.append(pd.DataFrame(mix_table).to_markdown(index=False))

    md.append("\n\n## 4. LLM Target Share per Category in Final Set\n\n")
    llm_share_table = []
    for cat, grp in all_final.groupby("category"):
        total = len(grp)
        llm_cnt = grp["human_source"].astype(str).str.contains("llm_regenerated").sum()
        share = (llm_cnt / total * 100) if total > 0 else 0.0
        llm_share_table.append({
            "Category": cat,
            "Total Filtered Rows": total,
            "LLM Rows": llm_cnt,
            "Non-LLM Rows": total - llm_cnt,
            "LLM Share (%)": f"{share:.1f}%",
            "Status": "PASS (<=15%)" if share <= 15.01 else "EXCEEDS"
        })
    md.append(pd.DataFrame(llm_share_table).to_markdown(index=False))

    # 6. 30 Random Final Rows
    md.append("\n\n## 5. Inspection of 30 Random Final Filtered Rows\n\n")
    sample_final = all_final.sample(min(30, len(all_final)), random_state=42)
    for i, (orig_i, row) in enumerate(sample_final.iterrows(), 1):
        ai_snip = (row.get("AI Text", "")[:200] + "...") if len(row.get("AI Text", "")) > 200 else row.get("AI Text", "")
        hu_snip = (row.get("Human Text", "")[:200] + "...") if len(row.get("Human Text", "")) > 200 else row.get("Human Text", "")
        md.append(f"### Final Sample {i} (Row {orig_i} - {row.get('category')}, source={row.get('human_source')})\n")
        md.append(f"- **Metrics**: ratio={row.get('ratio'):.2f} | target_words={row.get('target_words')} | mean_cov={row.get('mean_cov'):.2f} | copy_rate={row.get('copy_rate'):.2f} | fact_ret={row.get('fact_retention'):.2f} | burst={row.get('burstiness'):.1f} | cliches={row.get('human_cliche_density'):.1f} | quality={row.get('quality_score'):.1f}\n")
        md.append(f"- **AI Text**: `{ai_snip.strip().replace(chr(10), ' ')}`\n")
        md.append(f"- **Human Text**: `{hu_snip.strip().replace(chr(10), ' ')}`\n\n")

    # 7. 10 Rows dropped ONLY by opening/fragment rule
    md.append("\n## 6. Inspection of 10 Rows Dropped ONLY by Opening / Fragment Rule\n\n")
    if all_opening_dropped:
        sample_dropped = pd.DataFrame(all_opening_dropped).sample(min(10, len(all_opening_dropped)), random_state=42)
        for i, (idx, item) in enumerate(sample_dropped.iterrows(), 1):
            ai_snip = (item.get("AI Text", "")[:200] + "...") if len(item.get("AI Text", "")) > 200 else item.get("AI Text", "")
            hu_snip = (item.get("Human Text", "")[:200] + "...") if len(item.get("Human Text", "")) > 200 else item.get("Human Text", "")
            md.append(f"### Dropped Opening Sample {i} (Row {item.get('row_index')} - {item.get('category')}, source={item.get('source')})\n")
            md.append(f"- **Drop Reason**: `{item.get('opening_reason')}`\n")
            md.append(f"- **First Human Sentence**: `{item.get('first_human_sentence')}`\n")
            md.append(f"- **AI Text**: `{ai_snip.strip().replace(chr(10), ' ')}`\n")
            md.append(f"- **Human Text**: `{hu_snip.strip().replace(chr(10), ' ')}`\n\n")
    else:
        md.append("*(No rows failed exclusively on opening rules)*\n")

    final_report_path = Path("Data/improved_data/final_filtered_report.md")
    with open(final_report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\n[+] Full report saved to {final_report_path}")


if __name__ == "__main__":
    main()
