import pandas as pd
import re
import html
from pathlib import Path

# Run this script from the project root.
# Expected:
# data/improved_data/train.csv
# data/improved_data/validation.csv
# data/improved_data/test.csv
#
# Output:
# data/improved_data/train_normalized.csv
# data/improved_data/validation_normalized.csv
# data/improved_data/test_normalized.csv

DATA_DIR = Path("../improved_data")

FILES = ["train.csv", "validation.csv", "test.csv"]


def normalize_text(text):
    if pd.isna(text):
        return text

    s = str(text)

    # Decode HTML entities such as &nbsp; and &amp;
    s = html.unescape(s)

    # Remove HTML remnants while preserving a space between surrounding text.
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)

    # Remove broken encoding / invisible characters.
    s = s.replace("\ufffd", "")  # Unicode replacement character: �
    s = re.sub(
        r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]",
        "",
        s,
    )
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    s = re.sub(r"\u00ad", "", s)  # soft hyphen

    # Remove a small set of decorative/box glyphs that appeared as noise.
    # This is intentionally NOT a blanket "remove symbols" rule.
    s = re.sub(r"[\u25a1\u25aa\u25ab\u25cf\u25cb\u25c7\u25c8\u25e6]", " ", s)

    # Normalize whitespace.
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def process_file(path):
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return

    df = pd.read_csv(path)

    text_cols = [c for c in ("AI Text", "Human Text") if c in df.columns]
    if not text_cols:
        print(f"[ERROR] {path}: AI Text/Human Text columns not found")
        return

    changes = 0

    for col in text_cols:
        before = df[col].astype("string")
        df[col] = df[col].map(normalize_text)
        after = df[col].astype("string")
        changes += (before != after).fillna(False).sum()

    output = path.with_name(path.stem + "_normalized.csv")
    df.to_csv(output, index=False)

    print(
        f"[OK] {path.name}: {len(df):,} rows | "
        f"{changes:,} changed text cells -> {output.name}"
    )


if __name__ == "__main__":
    for filename in FILES:
        process_file(DATA_DIR / filename)

    print("\nDone.")
    print("Important: inspect the *_normalized.csv files before replacing originals.")
    print(
        "Preserved intentionally: $, %, +, =, <, >, ^, |, €, £, ¥, "
        "×, ÷, ≥, ≤, ±, →, URLs, slang, spelling mistakes, contractions, "
        "emojis, and informal punctuation."
    )
