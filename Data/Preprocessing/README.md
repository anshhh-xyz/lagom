# Lagom Humanizer 2.0 — Modular Dataset Preprocessing

This pipeline deliberately separates deterministic cleaning from each semantic signal so failures are easy to debug.

## Directory layout

```text
data/
├── Collection/              # raw/source CSV(s)
├── improved_data/           # all generated CSVs
└── preprocessing/           # these scripts
```

Put these scripts in `data/preprocessing/`.

## Pipeline

1. `improvements2.py` — deterministic cleaning, duplicate removal, elongation/slang/punctuation checks, length ratios and quality tiers.
2. `sbert_semantic.py` — SBERT sentence similarity. Uses NVIDIA CUDA automatically when available.
3. `tfidf_similarity.py` — TF-IDF lexical similarity. This is CPU-based because standard scikit-learn TF-IDF does not use CUDA.
4. `nli_cross_encoder.py` — NLI cross-encoder. Uses CUDA automatically when available. Default model: `cross-encoder/nli-deberta-v3-base`.
5. `finalize_dataset.py` — merges the independent signals, creates semantic-risk/manual-review files and final train/validation/test splits.

## Install

Recommended in a fresh virtual environment:

```bash
pip install -r requirements.txt
```

For your RTX 5050, install a CUDA-enabled PyTorch build appropriate to your NVIDIA driver from the official PyTorch selector, then install the remaining packages. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run

From the `data/preprocessing` directory:

```bash
python improvements2.py
python sbert_semantic.py
python tfidf_similarity.py
python nli_cross_encoder.py
python finalize_dataset.py
```

Or run them from anywhere by giving `--data-dir`.

### GPU controls

SBERT:

```bash
python sbert_semantic.py --device cuda --batch-size 64
```

NLI:

```bash
python nli_cross_encoder.py --device cuda --batch-size 16
```

If you get CUDA out-of-memory errors, lower the batch size to 32/16 for SBERT or 8 for NLI. The batch size changes speed/memory use, not the model itself.

## Outputs

Everything is kept under `data/improved_data/`:

- `base_clean.csv`
- `sbert_scored.csv`
- `tfidf_scored.csv`
- `nli_scored.csv`
- `final_scored.csv`
- `semantic_risk.csv`
- `manual_review.csv`
- `train.csv`
- `validation.csv`
- `test.csv`
- reports
- `_backups/` before destructive rewrites

## Important dataset policy

`improvements2.py` excludes strong/extreme elongation and other noisy style examples from the core pool, while retaining a tiny configurable fraction (`--keep-c 0.02`) as style noise.

The semantic scripts do **not** delete rows. They only add independent scores. `finalize_dataset.py` is the step that applies semantic-risk thresholds. This makes debugging much easier: if a row disappears from the final pool, you can inspect exactly which signal caused it to be flagged.

Semantic scores are screening signals, not proof that an AI→human pair preserves every fact. Always inspect `semantic_risk.csv` before training.
