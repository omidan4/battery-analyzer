#!/usr/bin/env python3
"""
train_advanced_models.py

Milestone 4: Advanced Modeling & Control

- Trains Gradient Boosting models for:
    * Overheat classification (overheat_label)
    * 30-second temperature rise regression (delta_T_30s)
- Adds simple temporal features:
    * inv_hot_spot_temp_mean_prev (lag-1)
    * delta_T_30s_prev (lag-1 of target)
    * power_mean, power_mean_roll_5s (rolling power proxy)
- Saves:
    * models/advanced/gb_overheat.joblib
    * models/advanced/gbr_deltaT.joblib
    * metrics/gb_overheat_metrics.json
    * metrics/gbr_deltaT_metrics.json
    * advanced_modeling_and_control.md (summary + heuristics)
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

DATA_PATH_DEFAULT = "src/notebooks/clean/inverter_labeled_1hz.csv"
MODELS_DIR = Path("models/advanced")
METRICS_DIR = Path("metrics")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple temporal / rolling features for Milestone 4."""
    df = df.sort_values("timestamp").copy()

    # lag features
    if "inv_hot_spot_temp_mean" in df.columns:
        df["inv_hot_spot_temp_mean_prev"] = df["inv_hot_spot_temp_mean"].shift(1)

    if "delta_T_30s" in df.columns:
        df["delta_T_30s_prev"] = df["delta_T_30s"].shift(1)

    # power proxy + rolling power
    if (
        "inv_dc_bus_voltage_mean" in df.columns
        and "inv_dc_bus_current_mean" in df.columns
    ):
        df["power_mean"] = (
            df["inv_dc_bus_voltage_mean"] * df["inv_dc_bus_current_mean"]
        )
        df["power_mean_roll_5s"] = df["power_mean"].rolling(
            window=5, min_periods=1
        ).mean()

    # Drop initial rows where lag features are NaN
    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    """Chronological train/test split (no shuffling)."""
    n = len(df)
    split_idx = int(n * train_fraction)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def train_advanced_classifier(df: pd.DataFrame):
    """Train Gradient Boosting classifier for overheat prediction."""
    target = "overheat_label"
    assert target in df.columns, f"{target} not found in dataframe"

    # features: all *_mean plus temporal/rolling features, avoiding leakage
    exclude = {
        "inv_hot_spot_temp_mean",  # avoid trivially using current temp alone
        "inv_hot_spot_temp_future",
        "delta_T_30s",
        "timestamp",
        target,
    }
    temporal_cols = [
        c
        for c in [
            "inv_hot_spot_temp_mean_prev",
            "delta_T_30s_prev",
            "power_mean",
            "power_mean_roll_5s",
        ]
        if c in df.columns
    ]

    feature_cols = [
        c
        for c in df.columns
        if (c.endswith("_mean") or c in temporal_cols) and c not in exclude
    ]

    train_df, test_df = chronological_split(df)
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # Feature importances
    importances = clf.feature_importances_
    fi = (
        pd.Series(importances, index=feature_cols)
        .sort_values(ascending=False)
        .to_frame(name="importance")
    )

    metrics = {
        "task": "overheat_classification",
        "model": "GradientBoostingClassifier",
        "auc": auc,
        "f1": f1,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_columns": feature_cols,
    }

    return clf, metrics, fi, report


def train_advanced_regressor(df: pd.DataFrame):
    """Train Gradient Boosting regressor for 30-second ΔT prediction."""
    target = "delta_T_30s"
    assert target in df.columns, f"{target} not found in dataframe"

    df_numeric = df.copy()
    # Keep only numeric columns for regression features
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()

    exclude = {target, "inv_hot_spot_temp_future"}
    feature_cols = [
        c
        for c in numeric_cols
        if c not in exclude and not c.endswith("_count")
    ]

    train_df, test_df = chronological_split(df_numeric)
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    reg = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    importances = reg.feature_importances_
    fi = (
        pd.Series(importances, index=feature_cols)
        .sort_values(ascending=False)
        .to_frame(name="importance")
    )

    metrics = {
        "task": "deltaT_regression",
        "model": "GradientBoostingRegressor",
        "mae": mae,
        "rmse": rmse,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_columns": feature_cols,
    }

    return reg, metrics, fi


def write_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_summary_markdown(
    clf_metrics,
    reg_metrics,
    clf_fi: pd.DataFrame,
    reg_fi: pd.DataFrame,
    clf_report: str,
    out_path: Path,
):
    """
    Create Milestone 4 summary markdown with:
    - metrics
    - top features
    - simple control heuristics
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract some nice numbers (rounded)
    auc = clf_metrics["auc"]
    f1 = clf_metrics["f1"]
    mae = reg_metrics["mae"]
    rmse = reg_metrics["rmse"]

    top_cls = clf_fi.head(10)
    top_reg = reg_fi.head(10)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Advanced Modeling & Control – Milestone 4\n\n")
        f.write("This document summarizes the advanced models (Gradient Boosting)\n")
        f.write("trained on top of the baseline logistic regression (overheat) and\n")
        f.write("ridge regression (ΔT) models.\n\n")

        f.write("## 1. Overheat Classification – Gradient Boosting\n\n")
        f.write(f"- Model: **{clf_metrics['model']}**\n")
        f.write(f"- Test AUC: **{auc:.4f}**\n")
        f.write(f"- Test F1 (overheat class): **{f1:.4f}**\n")
        f.write(f"- Train samples: {clf_metrics['n_train']}\n")
        f.write(f"- Test samples: {clf_metrics['n_test']}\n\n")

        f.write("**Classification report (test split):**\n\n")
        f.write("```text\n")
        f.write(clf_report)
        f.write("\n```\n\n")

        f.write("**Top 10 features by importance (classification):**\n\n")
        f.write("| Feature | Importance |\n|---|---|\n")
        for idx, row in top_cls.iterrows():
            f.write(f"| {idx} | {row['importance']:.4f} |\n")
        f.write("\n")

        f.write("## 2. ΔT Regression – Gradient Boosting\n\n")
        f.write(f"- Model: **{reg_metrics['model']}**\n")
        f.write(f"- Test MAE (°C): **{mae:.3f}**\n")
        f.write(f"- Test RMSE (°C): **{rmse:.3f}**\n")
        f.write(f"- Train samples: {reg_metrics['n_train']}\n")
        f.write(f"- Test samples: {reg_metrics['n_test']}\n\n")

        f.write("**Top 10 features by importance (regression):**\n\n")
        f.write("| Feature | Importance |\n|---|---|\n")
        for idx, row in top_reg.iterrows():
            f.write(f"| {idx} | {row['importance']:.4f} |\n")
        f.write("\n")

        f.write("## 3. Control Heuristics Derived from the Models\n\n")
        f.write(
            "Based on the advanced models, we propose simple torque derating\n"
            "heuristics that could be implemented in the vehicle controller:\n\n"
        )
        f.write(
            "- Let **p_overheat** be the model's predicted probability of overheating\n"
            "  over the next ~30 seconds, and **ΔT_pred** be the predicted temperature\n"
            "  rise over the next 30 seconds.\n"
        )
        f.write("- Let **T_hotspot** be the current inverter hot-spot temperature.\n\n")

        f.write("Suggested rules:\n\n")
        f.write("- If `p_overheat > 0.7` **and** `ΔT_pred > 8°C`:\n")
        f.write("  - Reduce torque request by ~20–30% for the next 30s.\n")
        f.write("  - Optionally limit DC current to a conservative cap.\n\n")
        f.write("- If `p_overheat > 0.9` **or** `T_hotspot > 65°C`:\n")
        f.write("  - Apply a stronger derate (e.g., 40–50%) and issue a driver warning.\n\n")
        f.write("- If `p_overheat < 0.3` **and** `ΔT_pred < 3°C`:\n")
        f.write("  - Full performance is allowed; no derating needed.\n\n")

        f.write(
            "These heuristics translate the advanced ML models into actionable\n"
            "safety/performance trade-offs for the FSAE accumulator and inverter.\n"
        )


def main(data_path: str = DATA_PATH_DEFAULT):
    print(f"[Milestone 4] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df = add_temporal_features(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Classification
    print("[Milestone 4] Training Gradient Boosting classifier...")
    clf, clf_metrics, clf_fi, clf_report = train_advanced_classifier(df)

    clf_model_path = MODELS_DIR / "gb_overheat.joblib"
    joblib.dump({"model": clf, "metadata": clf_metrics}, clf_model_path)
    print(f"[Milestone 4] Saved classifier to: {clf_model_path}")

    gb_cls_metrics_path = METRICS_DIR / "gb_overheat_metrics.json"
    write_metrics(gb_cls_metrics_path, clf_metrics)
    print(f"[Milestone 4] Wrote classifier metrics to: {gb_cls_metrics_path}")

    # Regression
    print("[Milestone 4] Training Gradient Boosting regressor...")
    reg, reg_metrics, reg_fi = train_advanced_regressor(df)

    reg_model_path = MODELS_DIR / "gbr_deltaT.joblib"
    joblib.dump({"model": reg, "metadata": reg_metrics}, reg_model_path)
    print(f"[Milestone 4] Saved regressor to: {reg_model_path}")

    gb_reg_metrics_path = METRICS_DIR / "gbr_deltaT_metrics.json"
    write_metrics(gb_reg_metrics_path, reg_metrics)
    print(f"[Milestone 4] Wrote regressor metrics to: {gb_reg_metrics_path}")

    # Summary markdown with control heuristics
    summary_path = Path("advanced_modeling_and_control.md")
    write_summary_markdown(
        clf_metrics, reg_metrics, clf_fi, reg_fi, clf_report, summary_path
    )
    print(f"[Milestone 4] Wrote summary + heuristics to: {summary_path}")


if __name__ == "__main__":
    main()
