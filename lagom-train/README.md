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
Train on all ~22,400 curated pairs:
```powershell
python train.py --preset local8gb --epochs 1 --output-dir outputs
```
* **VRAM**: ~4.8 GB – 5.5 GB (safe under 8.0 GB).
* **Duration**: ~35–45 minutes for 1 epoch.
* **Resume if interrupted**:
  ```powershell
  python train.py --preset local8gb --resume --output-dir outputs
  ```

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
