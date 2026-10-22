# Lagom Deep Mode – Training & Inference Package

Fine-tunes a **Qwen2.5 Instruct** model using **QLoRA (4-bit quantization + LoRA)** to rewrite robotic AI text into authentic, natural human writing.

---

## 📂 Architecture & File Map

```
lagom-train/
│
├── common.py           # Shared system prompts, category tones, hardware presets, name helpers
├── data.py             # CSV cleaning, pair loading, tier filtering, prompt masking & tokenization
├── train.py            # Main training pipeline (data prep → QLoRA → training → checkpointing → sample)
├── infer.py            # CLI & interactive text humanizer using trained adapter or merged weights
├── merge.py            # Merges LoRA adapter into full 16-bit standalone model
├── requirements.txt    # Python dependencies
└── README.md           # This comprehensive guide
```

---

## 🔍 In-Depth File-by-File Breakdown

### 1. `common.py` (Prompts, Presets & Helpers)
Acts as the single source of truth for prompts and hardware configurations across training and inference.

* **Lines 4–10 (`SYSTEM_PROMPT`)**: Defines Lagom's core identity and objective: preserving 100% of the factual meaning while varying sentence length, rhythm, and eliminating robotic transition clichés.
* **Lines 12–34 (`CATEGORY_INSTRUCTIONS`)**: Domain-specific rewrite guidelines for 5 categories:
  * `general`: Natural, organic human writing; cleans AI cadence.
  * `essay`: Focuses on thesis flow and syntactic rhythm; strips "furthermore", "moreover", "in conclusion".
  * `academic`: Scholarly prose, precise terminology without artificial buzzwords or robotic hedging.
  * `email`: Warm, authentic professional tone; eliminates stiff corporate jargon.
  * `document`: Clean report style with logical structural hierarchy.
* **Lines 37–56 (`build_messages(text, category, style)`)**: Constructs standard ChatML prompt turns `[{"role": "system", ...}, {"role": "user", ...}]` with explicit style conditioning:
  ```text
  Target Style: {style}
  Guidelines: {category_instructions}

  [TEXT TO HUMANIZE]:
  {text}
  ```
  This explicit `Target Style:` header conditions the LoRA adapter directly on the user's chosen style token (`Email`, `Essay`, `Academic`, `General`, `Document`).

---

### 2. `data.py` (Dataset Loading & Tokenization)
Prepares training pairs and ensures loss is computed **only** on the human rewrite, not on the prompt.

* **Lines 12–25 (`clean_text(text)`)**: Normalizes whitespace, strips trailing spaces per line, and collapses repeated blank lines into single newlines.
* **Lines 28–47 (`load_pairs(csv_path, min_words=25)`)**:
  * Reads `AI Text`, `Human Text`, and `category`.
  * Retains all quality tiers (including Tier C to introduce realistic regularization noise and prevent overfitting).
  * Enforces minimum word count (`min_words=25`) on both AI and human texts to avoid degenerate short phrases.
  * Deduplicates identical AI input texts to prevent training bias.
  * Normalizes categories (defaults unknown categories to `"general"`).
  * Returns usable pairs list and a Counter dict of dropped rows.
* **Lines 66–69 (`split_pairs(pairs, val_size, seed=42)`)**: Deterministically shuffles and holds out `val_size` examples for validation loss calculation.
* **Lines 72–101 (`encode(pairs, tokenizer, max_len, eos_text="<|im_end|>\n", chunk=256)`)**:
  * Formats prompt with chat template: `<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n`.
  * Formats completion: Human rewrite + `<|im_end|>\n`.
  * **Target Masking**: Sets `labels = [-100] * len(prompt_tokens) + completion_tokens`. During backpropagation, PyTorch cross-entropy ignores `-100`, forcing the model to calculate gradients exclusively on the human output.
  * **Strict Dropping (No Mid-Sentence Truncation)**: Pairs exceeding `max_len` (1,536 tokens) are discarded rather than truncated. Truncating targets would teach the model to emit incomplete sentences and omit EOS tokens.
  * Computes percentile token stats (`mean`, `p50`, `p95`, `p99`, `max`).

---

### 3. `train.py` (Fine-Tuning Engine)
Orchestrates QLoRA model loading, PEFT configuration, training execution, checkpointing, and validation.

* **Lines 17–18**: Configures PyTorch CUDA memory allocator (`expandable_segments:True`) to reduce memory fragmentation.
* **Lines 24–58 (`parse_args()`)**:
  * **Automatic Dataset Discovery**: Checks `../Data/improved_data/train_normalized.csv` and `validation_normalized.csv` automatically if available.
  * Parses arguments for preset, learning rate (`2e-4`), epochs (`1.0`), LoRA rank, and output directory.
* **Lines 55–64**: Attempts to load **Unsloth** for speed/VRAM optimization. If running on native Windows where Triton/Unsloth are unavailable, gracefully falls back to plain **Hugging Face + PEFT**.
* **Lines 71–76**: Validates GPU availability and selects precision (`bfloat16` on RTX 5050 / Ampere / Ada / Blackwell, otherwise `float16`).
* **Lines 79–103 (Model Initialization)**:
  * Injects 4-bit NormalFloat (`nf4`) quantization via `BitsAndBytesConfig` with double quantization.
  * Prepares model for k-bit training and enables gradient checkpointing with `use_reentrant=False`.
  * Attaches LoRA adapters across all linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  * Sets `tokenizer.padding_side = "right"` and logs trainable parameters (~20M trainable out of ~3B total).
* **Lines 105–123 (Data Pipeline)**: Loads training and validation datasets, tokenizes with masked labels, and wraps them in Hugging Face `Dataset` objects.
* **Lines 124–130 (`collate(batch)`)**: Dynamic batch padding to the longest sequence in each specific micro-batch, saving significant VRAM over static padding.
* **Lines 132–154 (Trainer Setup)**:
  * Uses `paged_adamw_8bit` optimizer: pages optimizer state to host RAM during attention spikes to prevent CUDA OOM on 8GB VRAM.
  * Cosine learning rate schedule with 3% warmup.
  * Regular checkpointing and evaluation steps.
* **Lines 155–160**: Detects latest checkpoint in `output-dir` if `--resume` is passed, and executes training.
* **Lines 162–169**: Saves final LoRA adapter weights, tokenizer configuration, and run metadata (`lagom_run.json`) to `<output_dir>/adapter/`.
* **Lines 170–183 (Post-training Live Sample)**: Enables `model.config.use_cache = True`, generates a sample humanization from the first validation item, and prints AI Input vs Model Output vs Reference Human text for immediate quality inspection.

---

### 4. `infer.py` (Inference & Text Humanizer)
Allows instant generation using the trained LoRA adapter or a merged model.

* **Lines 16–26**: Parses command line arguments (`--text`, `--file`, `--category`, `--temperature`, `--top-p`, `--max-new-tokens`).
* **Lines 28–39**: Accepts text via command line flag, external file, or interactive terminal prompt (with Windows/Linux submission instructions).
* **Lines 40–49**: Loads the base model in 4-bit NF4 and loads the LoRA adapter from `outputs/adapter` (or directly loads 16-bit merged weights if `--merged` is provided).
* **Lines 50–57**: Applies the chat template and generates the humanized text using nucleus sampling (`top_p=0.9`, `temperature=0.7`, `repetition_penalty=1.05`).

---

### 5. `merge.py` (Adapter Merger)
Merges LoRA delta weights directly into base model weights for production deployment.

* **Lines 20–28**: Loads the base model in `float16` onto system RAM (`device_map={"": "cpu"}`), eliminating GPU VRAM constraints (~7GB system RAM needed for 3B).
* **Lines 28–31**: Applies `PeftModel.merge_and_unload()` and saves standalone weights with `safetensors`. The resulting model can be served with vLLM, Ollama, GGUF, or Hugging Face Transformers without PEFT dependencies.

---

## 🚀 How to Run Locally (RTX 5050 8GB)

### 1. Quick Smoke Test (Verify VRAM & Code Pipeline)
Test on 100 samples to verify that download, quantization, and backprop succeed:
```powershell
cd d:\Projects\Lagom\lagom-train
python train.py --preset local8gb --max-samples 100 --epochs 1 --output-dir outputs_smoke
```

### 2. Full Training Run
Train with LoRA rank `r=32` and `alpha=64`:
```powershell
python train.py --preset local8gb --epochs 3 --output-dir outputs
```
* **LoRA Capacity**: `r=32`, `lora_alpha=64` across all 7 linear projections (`q, k, v, o, gate, up, down`), providing double the representational capacity of r=16 to prevent register collapse.
* **VRAM**: ~5.2 GB – 5.8 GB (well within 8.0 GB limit).
* **Resume if interrupted**:
  ```powershell
  python train.py --preset local8gb --resume --output-dir outputs
  ```

### 3. Dataset Audit & Length Ratio Check
* **Audit & extract all short pairs (< 0.70 ratio):**
  ```powershell
  python ../Data/Preprocessing/audit_length_ratios.py --min-ratio 0.70
  ```
* **Full Batch Regeneration (Processes all 4,706 flagged pairs or 3,741 in train):**
  Uses both **Groq** (`qwen3.8-27b`) and **Gemini** (`gemini-3.5-flash-lite`), rotating through all available API keys (`GROQ_API_KEY_1..4`, `GEMINI_API_KEY_1..3`), with real-time **SBERT cosine similarity scoring** on GPU:
  ```powershell
  # Process all remaining train pairs with auto provider rotation & SBERT validation:
  python ../Data/Preprocessing/regenerate_short_pairs.py --split train

  # Process all 4,706 pairs across all splits (train, validation, test):
  python ../Data/Preprocessing/regenerate_short_pairs.py --split all

  # Fast run focused specifically on essays:
  python ../Data/Preprocessing/regenerate_short_pairs.py --category essay --provider auto
  ```
  *(Checkpoints are saved to `Data/improved_data/regenerated_cache.csv` every 5 pairs, making it completely safe to pause and resume anytime).*

* **Apply regenerated pairs & purge remaining truncated pairs (< 0.70 ratio):**
  ```powershell
  python ../Data/Preprocessing/clean_and_replace_pairs.py --min-ratio 0.70
  ```
  *(Creates a safety backup, merges all regenerated 1:1 rewrites, and purges remaining short rows so training has zero summarization bias).*

---

## 🛠️ Recent Improvements & Architecture Updates (v2.1)

### What Was Done
1. **LoRA Capacity Boost ($r=32, \alpha=64$):**
   - Upgraded adapter rank from $r=16$ to $r=32$ ($\alpha=64$) across all 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Doubled trainable adapter parameters (~20M → ~40M), providing capacity to retain intellectual vocabulary while restructuring syntax.
   - Added CLI arguments `--lora-r` and `--lora-alpha` to `train.py` with runtime logging.
2. **Dataset Length Ratio Audit:**
   - Audited length ratios ($\frac{\text{Human Words}}{\text{AI Words}}$) across train, validation, and test datasets.
   - Discovered that **43.7% of all Essay pairs** and **30.7% of Academic pairs** in the training set had ratios $< 0.70$ (mean ratio $0.78$), causing the model to learn unwanted summarization.
   - Extracted all 4,706 short pairs ($< 0.70$) into `Data/improved_data/pairs_for_reevaluation.csv`.
3. **High-Register, Length-Balanced Pair Generation & Batch Replacement:**
   - Generated curated pairs with strict 1:1 length ratios ($0.97 - 1.12$) focusing on `essay`, `academic`, and `document` styles.
   - Built automated batch regenerator `regenerate_short_pairs.py` to rewrite short pairs via Gemini and replace them when ratio $\ge 0.85$.
   - Built `clean_and_replace_pairs.py` which purged 3,741 remaining truncated rows, leaving 8,050 pristine, non-summarizing training pairs (mean ratio raised from 0.78 to 0.84).

### Why It Was Done
* **Root Cause 1 — Register Collapse:** The previous model dumbed down sophisticated vocabulary (e.g., turning *"sophisticated digital infrastructure"* into generic *"technology"*). Increasing to $r=32$ gives the model sufficient capacity to preserve high-register diction.
* **Root Cause 2 — Unwanted Summarization:** The model dropped 20–30% of content because nearly a third of the training pairs had human rewrites that were truncated or truncated mid-sentence.

### Files Created & Modified
* ✏️ `lagom-train/common.py`: Set default presets to `lora_r=32, lora_alpha=64`.
* ✏️ `lagom-train/train.py`: Added `--lora-alpha` CLI flag and runtime LoRA config logging.
* ➕ `Data/Preprocessing/audit_length_ratios.py`: Script to audit word counts, percentiles, and extract short pairs.
* ➕ `Data/Preprocessing/generate_high_quality_pairs.py`: Script to create and append high-register, 1:1 ratio pairs.
* ➕ `Data/Preprocessing/regenerate_short_pairs.py`: Automated batch regenerator that rewrites short pairs via Gemini and replaces them with 1:1 length ratio rewrites.
* ➕ `Data/Preprocessing/clean_and_replace_pairs.py`: Applies regenerated pairs and purges truncated rows from `train_normalized.csv`.
* 📄 `Data/improved_data/pairs_for_reevaluation.csv`: 4,706 flagged pairs ($< 0.70$ ratio) requiring re-evaluation.
* 📄 `Data/improved_data/regenerated_cache.csv`: Caching file tracking regenerated pairs and new length ratios.
* 📄 `Data/improved_data/new_high_quality_pairs.csv`: Curated 1:1 ratio pairs.
* ✏️ `Data/improved_data/train_normalized.csv`: Cleaned dataset (8,050 rows, zero $<0.70$ pairs, mean ratio 0.84).
* 📄 `Data/improved_data/train_normalized.backup_11791.csv`: Safety backup of original dataset.
* ✏️ `lagom-train/README.md`: Updated documentation and guides.

---

## 🧪 Testing Your Model (Inference)

### 1. Test via CLI Flag
```powershell
python infer.py --adapter outputs/adapter --style essay --text "Furthermore, it is widely believed that modern advancements in automated intelligence are significant. In conclusion, one must acknowledge..."
```

### 2. Test via Text File
```powershell
python infer.py --adapter outputs/adapter --style academic --file input.txt
```

### 3. Interactive Prompt
```powershell
python infer.py --adapter outputs/adapter --style email
```
*(Paste your text, press Enter, then press `Ctrl+Z` and Enter on Windows to run).*

---

## 📦 Exporting Merged Model (Optional)
Merge adapter into standalone 16-bit weights:
```powershell
python merge.py --adapter outputs/adapter --out outputs/merged
python infer.py --merged outputs/merged --text "AI text to rewrite..."
```
