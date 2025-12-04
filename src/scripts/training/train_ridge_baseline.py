#!/usr/bin/env python3
"""
Train and persist the StandardScaler + Ridge baseline for 30s temperature delta.

Usage:
    PYTHONPATH=. python src/scripts/train_ridge_baseline.py \
        --input src/notebooks/clean/inverter_labeled_1hz.csv \
        --output models/baseline/ridge_deltaT.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import json
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="src/notebooks/clean/inverter_labeled_1hz.csv",
        help="Path to labeled dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="models/baseline/ridge_deltaT.joblib",
        help="Where to store the trained model artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of chronologically sorted rows to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--target",
        default="delta_T_30s",
        help="Target column to regress (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (default: %(default)s)",
    )
    return parser.parse_args()


def build_features(df: pd.DataFrame):
    exclude = {"inv_hot_spot_temp_mean", "inv_hot_spot_temp_future", "delta_T_30s"}
    return [c for c in df.columns if c.endswith("_mean") and c not in exclude]


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str))


def main() -> None:
    args = parse_args()
    data_path = Path(args.input)
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp")
    feature_cols = build_features(df)

    model_df = df.dropna(subset=feature_cols + [args.target])
    split_idx = int(len(model_df) * args.train_fraction)
    train_df = model_df.iloc[:split_idx]
    test_df = model_df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[args.target]
    X_test = test_df[feature_cols]
    y_test = test_df[args.target]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.alpha)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.3f} °C")
    print(f"RMSE: {rmse:.3f} °C")

    metrics = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "mae": mae,
        "rmse": rmse,
        "target": args.target,
        "alpha": args.alpha,
    }

    artifact = {
        "model": pipeline,
        "metadata": {
            "feature_columns": feature_cols,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "data_path": str(data_path),
            "train_fraction": args.train_fraction,
            "target": args.target,
            "alpha": args.alpha,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"Saved ridge artifact to {output_path}")

    metrics_path = Path("metrics/ridge_metrics.json")
    metrics.update(artifact["metadata"])
    save_metrics(metrics_path, metrics)
    print(f"Logged metrics to {metrics_path}")


if __name__ == "__main__":
    main()
