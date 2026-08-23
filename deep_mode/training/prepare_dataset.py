import argparse
import csv
import json
import os
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Lagom, a specialized humanizer AI. Your task is to rewrite the provided "
    "AI-generated text so it sounds naturally human-authored, organic, and nuanced while "
    "preserving 100% of the original factual meaning and core ideas. Vary sentence lengths, "
    "use natural syntactic rhythms, eliminate robotic transition cliches, and match the "
    "target genre tone."
)

CATEGORY_INSTRUCTIONS = {
    "general": (
        "Rewrite this text into natural, organic human writing. Remove artificial AI cadence, "
        "repetitive transitions, and generic over-explanations while keeping the meaning intact."
    ),
    "essay": (
        "Rewrite this essay passage in the voice of a skilled human writer. Vary the sentence "
        "rhythm naturally, remove formulaic signposts (e.g. 'furthermore', 'moreover', 'in conclusion'), "
        "and preserve genuine thesis flow without repetitive summary statements."
    ),
    "academic": (
        "Rewrite this academic passage in genuine, human-authored scholarly prose. Use precise, "
        "substantive vocabulary without artificial buzzwords or robotic hedging patterns."
    ),
    "email": (
        "Rewrite this email to sound like an authentic human professional. Use warm, natural "
        "greetings and sign-offs, realistic phrasing, and eliminate stiff corporate AI cliches."
    ),
    "document": (
        "Rewrite this document/report in clean, natural human report style. Ensure sharp readability, "
        "logical structural hierarchy, and authentic business prose."
    ),
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.strip().split("\n")]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


def format_chatml_example(ai_text: str, human_text: str, category: str) -> dict:
    instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
    user_content = f"{instruction}\n\n[TEXT TO HUMANIZE]:\n{ai_text}"
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": human_text},
        ],
        "category": category,
    }


def format_alpaca_example(ai_text: str, human_text: str, category: str) -> dict:
    instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
    return {
        "instruction": instruction,
        "input": ai_text,
        "output": human_text,
        "system": SYSTEM_PROMPT,
        "category": category,
    }


def process_dataset(csv_path: str, output_dir: str, val_split: float = 0.1, seed: int = 42, min_words: int = 25):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Source CSV not found at: {csv_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset from: {csv_path}")
    raw_pairs = []
    seen_ai = set()
    category_stats = {}

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        for row_idx, row in enumerate(reader):
            if len(row) < 3:
                continue
            ai_text = clean_text(row[0])
            human_text = clean_text(row[1])
            category = row[2].strip().lower() if len(row) > 2 else "general"

            if not ai_text or not human_text:
                continue
            if len(ai_text.split()) < min_words or len(human_text.split()) < min_words:
                continue
            if ai_text in seen_ai:
                continue

            seen_ai.add(ai_text)
            raw_pairs.append((ai_text, human_text, category))
            category_stats[category] = category_stats.get(category, 0) + 1

    print(f"\n[Dataset Statistics]")
    print(f"Total Unique Valid Pairs: {len(raw_pairs)}")
    for cat, count in category_stats.items():
        print(f"  - {cat:12}: {count:,} pairs ({count / max(len(raw_pairs), 1) * 100:.1f}%)")

    random.seed(seed)
    random.shuffle(raw_pairs)

    val_size = int(len(raw_pairs) * val_split)
    val_pairs = raw_pairs[:val_size]
    train_pairs = raw_pairs[val_size:]

    print(f"\nTrain Set: {len(train_pairs):,} pairs")
    print(f"Validation Set: {len(val_pairs):,} pairs")

    train_chatml_path = out_dir / "train_chatml.jsonl"
    val_chatml_path = out_dir / "val_chatml.jsonl"

    with open(train_chatml_path, "w", encoding="utf-8") as f:
        for ai_t, h_t, cat in train_pairs:
            f.write(json.dumps(format_chatml_example(ai_t, h_t, cat), ensure_ascii=False) + "\n")

    with open(val_chatml_path, "w", encoding="utf-8") as f:
        for ai_t, h_t, cat in val_pairs:
            f.write(json.dumps(format_chatml_example(ai_t, h_t, cat), ensure_ascii=False) + "\n")

    train_alpaca_path = out_dir / "train_alpaca.jsonl"
    val_alpaca_path = out_dir / "val_alpaca.jsonl"

    with open(train_alpaca_path, "w", encoding="utf-8") as f:
        for ai_t, h_t, cat in train_pairs:
            f.write(json.dumps(format_alpaca_example(ai_t, h_t, cat), ensure_ascii=False) + "\n")

    with open(val_alpaca_path, "w", encoding="utf-8") as f:
        for ai_t, h_t, cat in val_pairs:
            f.write(json.dumps(format_alpaca_example(ai_t, h_t, cat), ensure_ascii=False) + "\n")

    print(f"\n[Saved JSONL Datasets in {output_dir}]")
    print(f"  * ChatML Format : {train_chatml_path.name}, {val_chatml_path.name}")
    print(f"  * Alpaca Format : {train_alpaca_path.name}, {val_alpaca_path.name}")

    return train_chatml_path, val_chatml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare Lagom training dataset for Kaggle fine-tuning.")
    
    default_csv = Path(__file__).resolve().parent.parent.parent / "Data_Collection" / "lagom_pairs.csv"
    if not default_csv.exists():
        default_csv = Path(__file__).resolve().parent.parent / "Data_Collection" / "lagom_pairs.csv"
    if not default_csv.exists():
        default_csv = Path(__file__).resolve().parent / "lagom_pairs.csv"

    parser.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help="Path to raw lagom_pairs.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "processed_data"),
        help="Output directory for processed JSONL files",
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set fraction (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    process_dataset(csv_path=args.csv, output_dir=args.out, val_split=args.val_split, seed=args.seed)


if __name__ == "__main__":
    main()
