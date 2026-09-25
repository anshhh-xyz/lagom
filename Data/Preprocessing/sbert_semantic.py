#!/usr/bin/env python3
"""Lagom preprocessing step 2: SBERT sentence-level semantic similarity."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from common import detect_pair_columns, resolve_data_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(args.data_dir, root)
    path = Path(args.input) if args.input else data_dir / "improved_data" / "base_clean.csv"
    out = data_dir / "improved_data"
    df = pd.read_csv(path)
    ai, human, _ = detect_pair_columns(df)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    print(f"[SBERT] device={device}; cuda={torch.cuda.is_available()}")
    model = SentenceTransformer(args.model, device=device)
    a = model.encode(df[ai].fillna("").tolist(), batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    h = model.encode(df[human].fillna("").tolist(), batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    sim = np.sum(a * h, axis=1)
    result = pd.DataFrame({"sbert_similarity": sim})
    result["sbert_semantic_flag"] = np.select([sim < 0.55, sim < 0.68, sim >= 0.78], ["LOW", "REVIEW", "HIGH"], default="MEDIUM")
    # Keep the original rows and add only SBERT columns.
    merged = pd.concat([df.reset_index(drop=True), result], axis=1)
    merged.to_csv(out / "sbert_scored.csv", index=False)
    print(f"[SBERT] wrote {out / 'sbert_scored.csv'}")

if __name__ == "__main__": main()
