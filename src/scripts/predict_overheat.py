#!/usr/bin/env python3
"""
Run inference with the persisted logistic regression baseline.

Usage:
    PYTHONPATH=. python src/scripts/predict_overheat.py \
        --input path/to/inverter_labeled_1hz.csv \
        --model models/baseline/logreg_overheat.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="src/notebooks/clean/inverter_labeled_1hz.csv",
        help="Path to CSV containing features (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="models/baseline/logreg_overheat.joblib",
        help="Path to joblib model artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save predictions CSV. Prints head if omitted.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Name of timestamp column to carry through (default: %(default)s)",
    )
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    artifact = load(args.model)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})
    feature_cols = metadata.get("feature_columns")

    df = pd.read_csv(args.input, parse_dates=[args.timestamp_col])
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.endswith("_mean")]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    feature_df = df[feature_cols]
    if feature_df.isna().any().any():
        feature_df = feature_df.ffill().bfill()
        if feature_df.isna().any().any():
            raise ValueError("Unable to impute missing values in feature set.")
    X = feature_df
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    result = pd.DataFrame(
        {
            args.timestamp_col: df[args.timestamp_col],
            "overheat_prob": proba,
            "overheat_pred": preds,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    predictions = run_inference(args)
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / "overheat_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Wrote predictions to {output_path}")
    else:
        print(predictions.head())


if __name__ == "__main__":
    main()
