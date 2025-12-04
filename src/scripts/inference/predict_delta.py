#!/usr/bin/env python3
"""
Predict 30-second temperature deltas using the persisted ridge baseline.

Usage:
    PYTHONPATH=. python src/scripts/predict_delta.py \
        --input src/notebooks/clean/inverter_labeled_1hz.csv \
        --model models/baseline/ridge_deltaT.joblib \
        --output predictions
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
        help="Input CSV with features (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="models/baseline/ridge_deltaT.joblib",
        help="Path to ridge artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path or directory to write predictions (default: print head only)",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column to carry through (default: %(default)s)",
    )
    return parser.parse_args()


def load_artifact(model_path: Path):
    artifact = load(model_path)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})
    feature_cols = metadata.get("feature_columns")
    target = metadata.get("target", "delta_T_30s")
    return model, feature_cols, target


def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    model, feature_cols, target = load_artifact(Path(args.model))
    df = pd.read_csv(args.input, parse_dates=[args.timestamp_col]).sort_values(args.timestamp_col)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.endswith("_mean") and c not in {"inv_hot_spot_temp_mean"}]

    features = df[feature_cols]
    if features.isna().any().any():
        features = features.ffill().bfill()
        if features.isna().any().any():
            raise ValueError("Unable to impute missing values for features.")

    preds = model.predict(features)
    result = pd.DataFrame(
        {
            args.timestamp_col: df[args.timestamp_col],
            "delta_T_30s_pred": preds,
        }
    )
    if target in df.columns:
        result[target] = df[target]
    return result


def main() -> None:
    args = parse_args()
    predictions = run_inference(args)
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir():
            output_path = output_path / "delta_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Wrote delta predictions to {output_path}")
    else:
        print(predictions.head())


if __name__ == "__main__":
    main()
