"""Shared utilities for the Lagom preprocessing pipeline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    candidates = list(candidates)
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for col in df.columns:
        name = str(col).strip().lower()
        for candidate in candidates:
            if candidate.lower() in name:
                return col
    if required:
        raise ValueError(f"Could not find required column. Tried {candidates}. Available: {list(df.columns)}")
    return None


def detect_pair_columns(df: pd.DataFrame):
    ai = find_column(df, ["ai_text", "ai", "source", "source_text", "input", "prompt", "generated_text", "machine_text", "original_text"])
    human = find_column(df, ["human_text", "human", "target", "target_text", "output", "rewritten_text", "paraphrase", "humanized_text"])
    category = find_column(df, ["category", "type", "genre", "domain", "style"], required=False)
    return ai, human, category


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def word_tokens(text: str):
    return re.findall(r"\b[\w’'-]+\b", text.lower(), flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(word_tokens(text))


def char_count(text: str) -> int:
    return len(text)


def resolve_data_dir(data_dir_arg: str | None, script_dir: Path) -> Path:
    if data_dir_arg:
        return Path(data_dir_arg).resolve()
    # When script is inside Data/Preprocessing, script_dir.parent is the Data folder
    if (script_dir.parent / "Collection").exists() or (script_dir.parent / "improved_data").exists():
        return script_dir.parent
    if (script_dir.parent / "data").exists():
        return (script_dir.parent / "data").resolve()
    if (script_dir.parent / "Data").exists():
        return (script_dir.parent / "Data").resolve()
    return script_dir.parent


def choose_input_csv(data_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
        return p

    if not (data_dir / "Collection").exists() and (data_dir.parent / "Collection").exists():
        data_dir = data_dir.parent

    candidates = [
        data_dir / "improved_data" / "all_scored.csv",
        data_dir / "improved_data" / "base_clean.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    collection = data_dir / "Collection"
    csvs = sorted(collection.glob("*.csv")) if collection.exists() else []
    if len(csvs) == 1:
        return csvs[0]
    if len(csvs) > 1:
        raise RuntimeError("Multiple CSVs found in data/Collection. Pass --input explicitly so the pipeline uses the intended source.")
    raise FileNotFoundError("No input CSV found. Put the raw dataset in data/Collection or pass --input PATH.")


def ensure_dirs(data_dir: Path):
    if not (data_dir / "Collection").exists() and (data_dir.parent / "Collection").exists():
        data_dir = data_dir.parent
    out = data_dir / "improved_data"
    backup_root = out / "_backups"
    out.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)
    return out, backup_root

