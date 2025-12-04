## Baseline Model Metrics (Milestone 3)

Source data: `src/notebooks/clean/inverter_labeled_1hz.csv` (1 Hz aggregates from 2025‑06‑03 18:56–19:59). Train/test splits are chronological (80 % train, 20 % test).

### 1. Logistic Regression – Overheating Classification

| Metric | Value |
| --- | --- |
| ROC AUC | **0.994** |
| Accuracy | 0.869 |
| F1 (normal) | 0.922 |
| F1 (overheat) | 0.588 |
| Confusion Matrix (normal/overheat) | `[[516, 86], [1, 62]]` |

- Pipeline: StandardScaler → LogisticRegression (`class_weight='balanced'`, max_iter=1000).
- Features: `_mean` columns from control board temp, coolant temp, DC current/voltage, gate driver temp, module temps, and phase currents (see `metrics/logreg_metrics.json` for the full list).
- Interpretation:
  - High recall (0.984) on the overheat class ensures few missed events.
  - Precision on the rare class is ~0.42, so false positives remain; advanced models should target higher precision without sacrificing recall.

### 2. Ridge Regression – 30 s Temperature Delta (`delta_T_30s`)

| Metric | Value (°C) |
| --- | --- |
| MAE | **4.669** |
| RMSE | 9.270 |

- Pipeline: StandardScaler → Ridge (α = 1.0).
- Features: same `_mean` feature set as the classifier (temperatures, currents, voltages).
- Interpretation:
  - Average absolute error is <5 °C over a 30-second horizon, providing a baseline for short-term forecasting.
  - RMSE indicates larger errors occur during rapid transients; future models should exploit temporal context to reduce this variance.

### 3. Usage Notes

- Metrics are logged automatically to `metrics/logreg_metrics.json` and `metrics/ridge_metrics.json` every time the Makefile `train` target runs. Each entry includes feature lists, dataset paths, and split sizes for reproducibility.
- These baselines establish the minimum performance advanced models must exceed in Milestone 4.
