#!/usr/bin/env python3
"""
Train and persist the StandardScaler + logistic regression baseline.

Usage:
    PYTHONPATH=. python src/scripts/train_logreg_baseline.py \
        --input src/notebooks/clean/inverter_labeled_1hz.csv \
        --output models/baseline/logreg_overheat.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
        default="models/baseline/logreg_overheat.joblib",
        help="Where to store the trained model artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of chronologically sorted rows to use for training (default: 0.8)",
    )
    return parser.parse_args()


def build_features(df: pd.DataFrame):
    exclude = {"inv_hot_spot_temp_mean", "inv_hot_spot_temp_future", "delta_T_30s"}
    feature_cols = [c for c in df.columns if c.endswith("_mean") and c not in exclude]
    return feature_cols


def main() -> None:
    args = parse_args()
    data_path = Path(args.input)
    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp")
    feature_cols = build_features(df)

    model_df = df.dropna(subset=feature_cols + ["overheat_label"])
    split_idx = int(len(model_df) * args.train_fraction)
    train_df = model_df.iloc[:split_idx]
    test_df = model_df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["overheat_label"]
    X_test = test_df[feature_cols]
    y_test = test_df["overheat_label"]

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    artifact = {
        "model": clf,
        "metadata": {
            "feature_columns": feature_cols,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "data_path": str(data_path),
            "train_fraction": args.train_fraction,
            "notes": "StandardScaler + LogisticRegression(class_weight='balanced')",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"Saved model artifact to {output_path}")


if __name__ == "__main__":
    main()
