import os
import re
from pathlib import Path
import pandas as pd

# Unicode curly/smart quote characters included alongside straight quotes
_CLOSE_QUOTES = '"\u201c\u201d\u2018\u2019'
_OPEN_CAPS = r'[A-Z0-9"\u201c\u201d\u2018\u2019]'
_OPEN_CAPS_DOC = r'[A-Z0-9"\u201c\u201d\u2018\u2019\*\#]'
_TERMINAL = r'[.!?]["\u201c\u201d\u2018\u2019\)]?'


def repair_text(text: str, category: str = "general") -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""

    t = re.sub(r"\b([a-zA-Z]+)\?([a-zA-Z]+)\b", r"\1'\2", t)

    if category == "email":
        t = re.sub(r"(?:\n|\A|\. )\s*-*\s*(?:Original Message|Forwarded by|Forwarded Message|Sent by)[^\n]*[\s\S]*$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"(?:\n|\A)\s*(?:From:|To:|Sent:|Cc:|Subject:)[^\n]*(?:\n\s*(?:From:|To:|Sent:|Cc:|Subject:)[^\n]*)+[\s\S]*$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"(?:\n|\A|\. )\s*(?:This (?:e-?mail|message|communication)|The information (?:contained|in this))[^\n]*(?:confidential|privileged|intended solely)[\s\S]*$", "", t, flags=re.IGNORECASE)

    ref_pat = re.compile(
        r"(?:\n|\. |\A|\s+)(?:References|Works Cited|Bibliography|Works Consulted|Sources)"
        r"(?:\s*[:\n\-]+|\s+(?=[A-Z][a-zA-Z\s,]+(?:\(\d{4}\)|\(\s*n\.\s*d\.\s*\)|,\s*[A-Z]\.))){0,}[\s\S]*$",
        flags=re.IGNORECASE
    )
    t = ref_pat.sub("", t)

    t = re.sub(r"(?:\n|\. )\s*(?:References|Works Cited|Bibliography)\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:\n|\. )\s*(?:Available at|Retrieved from|Retrieved on|Accessed)[^\n]*(?:https?://|www\.|Web\.)[^\n]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+Web\.\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+Web\.\s+Appendix[\s\S]*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+Appendix\s+[A-Z][\s\S]*$", "", t, flags=re.IGNORECASE)

    if category in ("essay", "general"):
        t = re.sub(r"\s*\(\s*(?:https?://|www\.)?[a-zA-Z0-9.\-]+\.(?:com|org|edu|gov|net|co\.uk)\s*(?:editors|news|staff)?[^)]*\)", "", t, flags=re.IGNORECASE)

    if re.match(r"^[a-z]", t):
        m = re.search(r'(?:[\.!\?]\s+|\n\s*)([A-Z0-9"\u201c\u201d\u2018\u2019][^\.!\?]*)', t)
        if m:
            t = t[m.start(1):].strip()
        else:
            return ""

    t = re.sub(r"\s*\([^)]*$", "", t).strip()
    t = re.sub(r"\s+\[[^\]]*$", "", t).strip()

    t = re.sub(r"\s+(?:(?:e\.g|i\.e|cf|app|fig|al|vol|no|pp)\.|\b[A-Za-z]+\s+et\s+al\.)\s*$", "", t, flags=re.IGNORECASE).strip()

    # Match terminal punct optionally followed by straight or curly close quotes/parens
    matches = list(re.finditer(r'[.!?]["\u201c\u201d\u2018\u2019\)]?(?:\s+|\Z)', t))
    if matches:
        last_end = matches[-1].end()
        t = t[:last_end].strip()
    else:
        return ""

    t = re.sub(r"\s*\([^)]*$", "", t).strip()

    return t


def process_dataset(csv_path: str | Path, min_words: int = 30):
    path = Path(csv_path)
    if not path.exists():
        return

    df = pd.read_csv(path)
    if "AI Text" not in df.columns or "Human Text" not in df.columns:
        return

    initial_count = len(df)
    categories = df["category"].astype(str).str.lower() if "category" in df.columns else ["general"] * len(df)

    new_ai = []
    new_hu = []
    for ai, hu, cat in zip(df["AI Text"], df["Human Text"], categories):
        c_ai = repair_text(ai, cat)
        c_hu = repair_text(hu, cat)
        new_ai.append(c_ai)
        new_hu.append(c_hu)

    df["AI Text"] = new_ai
    df["Human Text"] = new_hu

    def is_valid_pair(row):
        ai = str(row["AI Text"]).strip()
        hu = str(row["Human Text"]).strip()
        if not ai or not hu:
            return False

        ai_len = len(ai.split())
        hu_len = len(hu.split())
        if ai_len < min_words or hu_len < min_words:
            return False

        ratio = hu_len / max(ai_len, 1)
        if ratio < 0.55 or ratio > 1.85:
            return False

        # Accept both straight and curly open/close quote chars as valid openers/terminators
        if not re.match(r'^[A-Z0-9"\u201c\u201d\u2018\u2019]', hu) or \
                not re.search(r'[.!?]["\u201c\u201d\u2018\u2019\)]?$', hu):
            return False
        if not re.match(r'^[A-Z0-9"\u201c\u201d\u2018\u2019\*\#]', ai) or \
                not re.search(r'[.!?]["\u201c\u201d\u2018\u2019\)]?$', ai):
            return False

        if re.search(r"\([^)]*$", hu) or re.search(r"\[[^\]]*$", hu):
            return False

        return True

    df = df[df.apply(is_valid_pair, axis=1)].copy()
    df.to_csv(path, index=False)
    print(f"{path.name}: {initial_count:,} -> {len(df):,} pairs cleaned.")


def main():
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "improved_data"
    targets = [
        data_dir / "final_scored.csv",
        data_dir / "base_clean.csv",
        data_dir / "train_normalized.csv",
        data_dir / "validation_normalized.csv",
        data_dir / "test_normalized.csv",
    ]
    for target in targets:
        process_dataset(target)


if __name__ == "__main__":
    main()
