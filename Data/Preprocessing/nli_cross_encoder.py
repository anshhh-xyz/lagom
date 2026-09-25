#!/usr/bin/env python3
"""Lagom preprocessing step 4: NLI cross-encoder semantic check.

The model receives the AI text as premise and human text as hypothesis. The
script discovers the model's label IDs from its config instead of assuming a
fixed class ordering.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from common import detect_pair_columns, resolve_data_dir


def normalize_label(s):
    return str(s).strip().lower().replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--batch-size", type=int, default=16)
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
    print(f"[NLI] device={device}; cuda={torch.cuda.is_available()}")
    model = CrossEncoder(args.model, device=device, max_length=512)
    pairs = list(zip(df[ai].fillna(""), df[human].fillna("")))
    probs = model.predict(pairs, batch_size=args.batch_size, apply_softmax=True, show_progress_bar=True)
    probs = np.asarray(probs)
    id2label = getattr(model.model.config, "id2label", {})
    labels = {int(k): normalize_label(v) for k, v in id2label.items()} if id2label else {}
    def col_for(*names):
        for idx, label in labels.items():
            if any(name in label for name in names): return probs[:, idx]
        return np.zeros(len(probs))
    contradiction = col_for("contradiction", "contradict")
    entailment = col_for("entailment", "entails")
    neutral = col_for("neutral")
    if not labels:
        raise RuntimeError("Could not read NLI labels from the model config; refusing to guess class ordering.")
    result = pd.DataFrame({
        "nli_entailment_prob": entailment,
        "nli_neutral_prob": neutral,
        "nli_contradiction_prob": contradiction,
    })
    result["nli_flag"] = np.select(
        [contradiction >= 0.50, entailment >= 0.70, entailment < 0.35],
        ["CONTRADICTION", "ENTAILMENT", "LOW_ENTAILMENT"],
        default="REVIEW"
    )
    pd.concat([df.reset_index(drop=True), result], axis=1).to_csv(out / "nli_scored.csv", index=False)
    print(f"[NLI] labels={labels}")
    print(f"[NLI] wrote {out / 'nli_scored.csv'}")

if __name__ == "__main__": main()
