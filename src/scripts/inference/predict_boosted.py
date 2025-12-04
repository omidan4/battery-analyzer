#!/usr/bin/env python3
"""
Run inference using the advanced Gradient Boosting models (Milestone 4).

Usage:
    PYTHONPATH=. python src/scripts/inference/predict_boosted.py \
        --input src/notebooks/clean/inverter_labeled_1hz.csv \
        --classifier models/advanced/gb_overheat.joblib \
        --regressor models/advanced/gbr_deltaT.joblib \
        --output predictions/advanced
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from src.scripts.training.train_boosted_models import add_temporal_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="src/notebooks/clean/inverter_labeled_1hz.csv",
        help="Path to labeled dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--classifier",
        default="models/advanced/gb_overheat.joblib",
        help="Path to saved classifier artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--regressor",
        default="models/advanced/gbr_deltaT.joblib",
        help="Path to saved regressor artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory or file to store predictions; prints head if omitted.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column to carry through (default: %(default)s)",
    )
    return parser.parse_args()


def load_artifact(path: Path):
    artifact = load(path)
    return artifact["model"], artifact.get("metadata", {})


def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    df = add_temporal_features(df)

    clf, clf_meta = load_artifact(Path(args.classifier))
    reg, reg_meta = load_artifact(Path(args.regressor))

    clf_features = clf_meta.get("feature_columns")
    if clf_features is None:
        raise ValueError("Classifier artifact missing feature_columns metadata.")

    reg_features = reg_meta.get("feature_columns")
    if reg_features is None:
        raise ValueError("Regressor artifact missing feature_columns metadata.")

    X_clf = df[clf_features]
    X_reg = df[reg_features]

    clf_probs = clf.predict_proba(X_clf)[:, 1]
    clf_preds = clf.predict(X_clf)
    reg_preds = reg.predict(X_reg)

    result = pd.DataFrame(
        {
            args.timestamp_col: df[args.timestamp_col],
            "gb_overheat_prob": clf_probs,
            "gb_overheat_pred": clf_preds,
            "gb_delta_T_pred": reg_preds,
        }
    )

    if "overheat_label" in df.columns:
        result["overheat_label"] = df["overheat_label"]
    if "delta_T_30s" in df.columns:
        result["delta_T_30s"] = df["delta_T_30s"]

    return result


def main() -> None:
    args = parse_args()
    predictions = run_inference(args)
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or str(args.output).endswith("/"):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "advanced_predictions.csv"
        else:
            # If output_path.parent exists as a file, remove it before creating the directory.
            if output_path.parent.is_file():
                output_path.parent.unlink()
            output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Wrote advanced predictions to {output_path}")
    else:
        print(predictions.head())


if __name__ == "__main__":
    main()
