#!/usr/bin/env python3
"""Lagom preprocessing step 3: TF-IDF lexical overlap.

TF-IDF is intentionally CPU-only here: scikit-learn's standard TF-IDF pipeline
does not use CUDA. It complements the GPU semantic models by measuring lexical
similarity directly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import detect_pair_columns, resolve_data_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ngram-max", type=int, default=2)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    data_dir = resolve_data_dir(args.data_dir, root)
    path = Path(args.input) if args.input else data_dir / "improved_data" / "base_clean.csv"
    out = data_dir / "improved_data"
    df = pd.read_csv(path)
    ai, human, _ = detect_pair_columns(df)
    texts = pd.concat([df[ai].fillna(""), df[human].fillna("")], ignore_index=True)
    vec = TfidfVectorizer(ngram_range=(1, args.ngram_max), min_df=1, sublinear_tf=True, strip_accents="unicode")
    mat = vec.fit_transform(texts)
    n = len(df)
    sim = cosine_similarity(mat[:n], mat[n:]).diagonal()
    result = pd.DataFrame({"tfidf_similarity": sim})
    result["tfidf_lexical_flag"] = np.select([sim < 0.15, sim < 0.30, sim >= 0.55], ["LOW", "REVIEW", "HIGH"], default="MEDIUM")
    pd.concat([df.reset_index(drop=True), result], axis=1).to_csv(out / "tfidf_scored.csv", index=False)
    print(f"[TF-IDF] wrote {out / 'tfidf_scored.csv'}")

if __name__ == "__main__": main()
