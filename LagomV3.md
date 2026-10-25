# Lagom V3: Data Quality Overhaul, Sentence-Level Auditing & Length-Controlled Humanization

## Executive Overview
**Lagom** is a QLoRA fine-tuned language model (based on `Qwen/Qwen2.5-3B-Instruct`) engineered to rewrite formulaic, AI-generated text into authentic, natural human prose. 

Prior evaluation identified a critical systemic failure: rather than performing true humanization (re-structuring sentences, pruning verbose filler, eliminating throat-clearing clichés), the model acted as a sentence-by-sentence synonym substitution engine.

### Root Cause Diagnosis
1. **1:1 Paraphrase Target Contamination**:
   - ~26% of training pairs (and ~40% of essay pairs) originated from `llm_regenerated_1_to_1` runs.
   - These pairs had a high sentence-level copy rate (median copy rate 0.42–0.53, with 75th percentile up to 0.86), meaning $>70\%$ of sentences had an exact structural clone in the target.
2. **AI-Cliché Vocabulary**:
   - LLM targets were heavily saturated with stereotypical AI jargon (*landscape, tapestry, multifaceted, paramount, delve, pivotal, testament, foster, harness, transformative, vibrant, ecosystem, synergy*), averaging 9.5 to 17.5 clichés per 1,000 words.
   - In contrast, authentic human texts (`hf_dataset_import`) had a baseline median of 0.00 clichés per 1,000 words.
3. **Length Bloat & Monotonous Ratios**:
   - 60% to 78% of pairs had length ratios $>0.95$ (often exceeding 1.05), training the model to preserve all unnecessary padding.
4. **Passage-Level Truncation in Prior Audits**:
   - Previous SBERT evaluations computed similarity over entire documents, which automatically truncated at 256 tokens and failed to evaluate the back half of essays and reports.

---

## Phase 1: Sentence-Level Audit & Infrastructure (Completed)

Phase 1 implemented a rigorous, sentence-level analysis across the entire dataset without modifying or overwriting any existing data.

### 1. Architectural Implementation
- **Sentence Segmentation**: Utilized `pysbd` (Python Sentence Boundary Disambiguation) to segment both AI and Human texts into individual sentences while preserving academic citations (e.g., `(Smith et al., 2018)`), abbreviations, and lists.
- **Dense Sentence Embeddings**: Employed `sentence-transformers/all-MiniLM-L6-v2` with $L_2$-normalized embeddings on CUDA.
- **Coverage & Alignment Matrix**: For each AI sentence $i$ and human sentence $j$, computed the cosine similarity matrix $S_{ij} = \mathbf{e}_i^{\text{ai}} \cdot \mathbf{e}_j^{\text{human}}$.
  - `min_cov`: Minimum best-match score $\min_i \max_j S_{ij}$ (detects dropped concepts or truncation).
  - `mean_cov`: Average best-match score $\frac{1}{N} \sum_i \max_j S_{ij}$ (measures overall semantic retention).
  - `copy_rate`: Fraction of AI sentences whose best human match exceeds $0.90$ cosine similarity (detects 1:1 synonym swapping).
  - `sent_count_ratio`: $\frac{|\text{human\_sents}|}{|\text{ai\_sents}|}$.
  - `struct_change`: Share of human sentences that map to a shared AI sentence (splitting) or have weak best matches $<0.60$ (merging/structural rewriting).
- **Stylistic Metrics**:
  - `burstiness`: Standard deviation of sentence lengths in words for human text.
  - `contraction_rate`: Number of apostrophe contractions per 100 words.
  - `human_cliche_density`: Count of the 32 designated AI-cliché keywords per 1,000 words in Human Text.
  - `ai_cliche_density`: Count of the 32 designated AI-cliché keywords per 1,000 words in AI Text.
- **Artifact Flagging**: Detected in-text citations `\([A-Z][^)]*\d{4}\)`, title leakage (headers ending in `" Essay"`, `"Research Paper"`, `"Report"`), and trailing reference lists (`"Works Cited"`, `"References"`).

### 2. Files Created & Modified in Phase 1
- `Data/Preprocessing/sentence_level_audit.py`: The complete auditing script with configurable CLI arguments and threshold dictionary.
- `Data/improved_data/audit_scores.csv`: Output database containing all 13,247 audited rows across train, validation, and test splits with all computed metrics.
- Backups created:
  - `Data/improved_data/train_normalized.backup_audit_1792928482.csv`
  - `Data/improved_data/validation_normalized.backup_audit_1792928482.csv`
  - `Data/improved_data/test_normalized.backup_audit_1792928482.csv`
- `lagom-train/infer.py` & `api/main.py`:
  - Upgraded with `chunk_paragraphs` (splits on `\n\n`, groups to 150–300 words).
  - Upgraded with inference length guard: resamples up to 3 times if output/input ratio is outside $[0.70, 0.95]$, returning the candidate closest to $0.85$.
- `eval_humanizer.py`:
  - Standalone evaluation script taking held-out JSONL inputs and generating comprehensive reports for burstiness, contraction rate, cliché density, copy rate, sentence count ratio, and length ratio distributions across `<0.70`, `0.70–0.80`, `0.80–0.90`, `0.90–0.95`, and `>0.95`.

---

## Phase 2: Data Filtering & Stratified Length Control (Attempt 1 - too strict, 564 rows, archived)

> [!NOTE]
> **Threshold Sweep in Progress:** Attempt 1 proved too aggressive (564 rows remaining) primarily because the hard min_cov < 0.55 rule discarded 63.2% of rows, including valid human rewrites that eliminated wordy AI introductions. The min_cov rule is being transitioned to a soft flag, replaced by an explicit act_retention metric. A systematic parameter sweep (ilter_sweep.py) is evaluating coverage floors (0.65/0.70/0.75), fact retention thresholds (0.6/0.7/0.8), and source-specific copy rate caps to reach the target of ~1,000 clean pairs per category. Attempt 1 outputs have been safely archived to Data/improved_data/phase2_attempt1/.


Phase 2 applies strict data-quality filtering and stratified length resampling to construct high-quality training and validation sets (`train_filtered.csv` and `validation_filtered.csv`).

### 1. Filtering Rules & Rationale
1. **Coverage Preservation**: Drop if $\text{min\_cov} < 0.55$ or $\text{mean\_cov} < 0.75$. Prevents models from learning to drop material arguments.
2. **Anti-Paraphrase Filter**:
   - Drop if $\text{copy\_rate} > 0.60$ for strict categories (`essay`, `academic`).
   - Drop if $\text{copy\_rate} > 0.75$ for looser categories (`general`, `email`, `document`).
   - Eliminates sentence-by-sentence synonym swapping.
3. **Anti-Cliché Filter**:
   - Drop if $\text{human\_cliche\_density} > 2\times$ the category median of `hf_dataset_import` (with floor of 4.0/1000w).
   - Purges targets contaminated with robotic buzzwords.
4. **Length Floor**:
   - Hard drop of any row with ratio $< 0.70$.
5. **Artifact Cleaning & Re-Checking**:
   - Rather than dropping rows with title leakage (e.g. `"The Great Gatsby Essay"`) or trailing bibliographies, clean the text in-place using regex, strip whitespace, recompute word counts and coverage metrics, and re-evaluate.
6. **Reason Tracking**:
   - Maintain a dedicated `reason_dropped` column explaining every rejection decision.

### 2. Stratified Resampling (Length Control Mix)
After passing all quality filters, rows within each category are resampled to fit the target length ratio mix:
- **0.70 – 0.80**: ~12%
- **0.80 – 0.90**: ~63%
- **0.90 – 0.95**: ~20%
- **> 0.95**: $\le 5\%$
- **< 0.70**: 0%
- **Target Words Column**: Computes `target_words = round(human_words, -1)` for each training row.
- **Quality Invariance**: Quality thresholds are never lowered to satisfy a bin. Shortfalls are reported directly in `filter_report.md`.

### 3. Deliverables
- `Data/Preprocessing/filter_for_humanization.py`
- `Data/improved_data/train_filtered.csv`
- `Data/improved_data/validation_filtered.csv`
- `Data/improved_data/filter_report.md`
- `test_normalized.csv` left completely untouched as an unbiased test benchmark.


---

## Phase 2 — Final Build: Combo #5 Filtering (Approved)

> **Status: COMPLETE** — `train_filtered.csv` (2,493 rows) and `validation_filtered.csv` (306 rows) built and validated.

### Approved Parameter Settings (Combo #5)

| Parameter | Value |
|-----------|-------|
| `mean_cov` floor | ≥ 0.65 |
| `fact_retention` floor | ≥ 0.60 |
| `copy_rate` cap — `llm_regenerated_1_to_1` | 0.50 |
| `copy_rate` cap — `hf_dataset_import` essay/academic | 0.85 |
| Cliché cap | 2× category median (hf rows), floor 4.0/1000w |
| Email / document copy cap | Relaxed (per sweep) |

### Additional Hard Rules Applied

1. **Source Separation**: `train_filtered.csv` built **only** from `train_normalized.csv` rows; `validation_filtered.csv` **only** from `validation_normalized.csv`. Never mixed. `test_normalized.csv` untouched.
2. **Length Bounds**: Drop rows with `ratio < 0.70` or `ratio > 0.95`. No resampling to a fixed mix. All rows in [0.70, 0.95] kept.
3. **`target_words` column**: `round(human_words, -1)` appended to every filtered row.
4. **Opening & Fragment Rules**:
   - Drop if Human Text starts with a lowercase letter.
   - Drop if first human sentence has no verb (POS tag `VB*`) — fragment check via `nltk.word_tokenize` + `nltk.pos_tag`.
   - Strip title leakage and trailing references before re-checking.
   - Drop if best-matching AI sentence for the first human sentence has cosine < 0.40 (opening mismatch).
5. **Expanded Cliché Stems** (added to the list): `navigat*`, `rigor*`, `furnish*`, `accordingly`, `labor*`, `endeavor*`, `meticulous*`, `comprehensive*`, `leverage*`, `robust*`, `nuanced`, `intricate*`, `delve*`, `underscore*`, `pivotal`, `tapestry`, `realm`. Matched by stem, not exact word.
6. **LLM Source Cap**: `llm_regenerated_1_to_1` rows capped at **≤ 15%** per category; ties broken by highest `quality_score`.

### Results

| Split | Rows |
|-------|------|
| `train_filtered.csv` | **2,493** |
| `validation_filtered.csv` | **306** |

**Category × Source Breakdown (train)**:

| Category | Source | Initial | Filtered | LLM Share |
|---|---|---|---|---|
| academic | hf_dataset_import | 2,316 | 699 | — |
| academic | llm_regenerated_1_to_1 | 350 | 29 | 4.0% ✓ |
| document | hf_dataset_import | 1,926 | 84 | — |
| document | llm_regenerated_1_to_1 | 660 | 14 | 14.3% ✓ |
| email | hf_dataset_import | 1,290 | 318 | — |
| email | llm_regenerated_1_to_1 | 695 | 15 | 4.5% ✓ |
| essay | hf_dataset_import | 1,108 | 433 | — |
| essay | llm_regenerated_1_to_1 | 725 | 3 | 0.7% ✓ |
| general | hf_dataset_import | 2,376 | 1,144 | — |
| general | llm_regenerated_1_to_1 | 679 | 60 | 5.0% ✓ |

All categories pass the ≤ 15% LLM share cap.

**Length Ratio Mix (train)**:

| Bin | Share | Count |
|-----|-------|-------|
| 0.70–0.80 | 56.0% | 1,567 |
| 0.80–0.90 | 32.3% | 903 |
| 0.90–0.95 | 11.8% | 329 |
| < 0.70 | 0.0% | 0 |
| > 0.95 | 0.0% | 0 |

### Files Created / Modified in Phase 2

| File | Purpose |
|------|---------|
| `Data/Preprocessing/build_final_filtered_data.py` | Main Phase 2 filtering script |
| `Data/Preprocessing/filter_sweep.py` | 108-combination threshold sweep |
| `Data/Preprocessing/build_held_out.py` | Builds `eval/held_out.jsonl` |
| `Data/improved_data/train_filtered.csv` | Final training set (2,493 rows) |
| `Data/improved_data/validation_filtered.csv` | Final validation set (306 rows) |
| `Data/improved_data/final_filtered_report.md` | Detailed per-category report with 30 sample rows |
| `Data/improved_data/sweep_report.md` | Sweep results for all 108 combos |
| `Data/improved_data/sweep_combinations_108.csv` | Machine-readable sweep output |
| `Data/improved_data/phase2_attempt1/` | Archived attempt 1 (too strict, 564 rows) |
| `eval/held_out.jsonl` | 41 eval records: 40 stratified test rows + 1 samples.md essay |

---

## Phase 3: Prompt & Training Infrastructure

### 7. Prompt Change — `lagom-train/common.py` `build_messages`

**Status: APPLIED**

`build_messages` now accepts `target_words: int | None`. The user message includes:
```
Target style: {style}
Target length: about {N} words
Task: {instruction}
...
```
- **Training**: `N = target_words` from the CSV column (= `round(human_words, -1)`).
- **Inference**: `N = round(0.85 × input_words, -1)` per chunk (computed inside `build_messages` when `target_words=None`).
- System prompt and category instructions are **unchanged**.

### 8. `data.py` — `load_pairs` / `encode` Updated

**Status: APPLIED**

`load_pairs` now reads `target_words`, `ai_words`, `human_words` columns (gracefully — no error if missing). Priority chain for `target_words`:
1. Explicit `target_words` column from `train_filtered.csv`.
2. `round(human_words, -1)` if `human_words` column present.
3. Count words in cleaned human text: `round(len(human.split()), -1)`.

`encode` passes `target_words=pair["target_words"]` to `build_messages`, making training and inference prompts **format-identical**.

### 9. Inference Guard — `infer.py` / `api/main.py`

**Status: PRE-EXISTING (confirmed unchanged)**

- Resamples up to 3× if `output_words / input_words` is outside `[0.70, 0.95]`.
- Returns candidate closest to `0.85` if all attempts fail the guard.
- `build_messages` called with `target_words=None` → falls back to `round(0.85 × input_words, -1)`.

### 10. Eval Set

**Status: COMPLETE**

`eval/held_out.jsonl` contains **41 records**:
- 40 rows stratified from `test_normalized.csv` by category (general: 11, academic: 10, document: 8, email: 6, essay: 5).
- 1 row from `samples.md` (the 710-word AI essay, category=essay).
- Each record: `{text, category, source, human_reference, ai_words, target_words}`.

Run evaluation (after training) with:
```bash
python eval_humanizer.py --input eval/held_out.jsonl --adapter lagom-train/outputs/adapter
```

---

## Smoke Test (PENDING USER APPROVAL)

After user reviews the above, the first training run will be:
- **Data**: `train_filtered.csv` (2,493 rows), but only 300 rows sampled.
- **Epochs**: 1
- **Purpose**: Verify the pipeline (tokenization, `target_words` in prompt, loss computation) before full training.

```bash
cd lagom-train
python train.py --data ../Data/improved_data/train_filtered.csv --max-rows 300 --epochs 1
```
