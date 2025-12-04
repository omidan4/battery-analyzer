# Advanced Modeling & Control – Milestone 4

This document summarizes the advanced models (Gradient Boosting)
trained on top of the baseline logistic regression (overheat) and
ridge regression (ΔT) models.

## 1. Overheat Classification – Gradient Boosting

- Model: **GradientBoostingClassifier**
- Test AUC: **0.9995**
- Test F1 (overheat class): **0.9688**
- Train samples: 2633
- Test samples: 659

**Classification report (test split):**

```text
              precision    recall  f1-score   support

           0      0.998     0.995     0.997       596
           1      0.954     0.984     0.969        63

    accuracy                          0.994       659
   macro avg      0.976     0.990     0.983       659
weighted avg      0.994     0.994     0.994       659

```

**Top 10 features by importance (classification):**

| Feature | Importance |
|---|---|
| inv_hot_spot_temp_mean_prev | 0.9184 |
| inv_coolant_temp_mean | 0.0304 |
| inv_module_c_temp_mean | 0.0169 |
| inv_module_a_temp_mean | 0.0109 |
| inv_module_b_temp_mean | 0.0079 |
| inv_dc_bus_current_mean | 0.0070 |
| inv_control_board_temp_mean | 0.0022 |
| inv_phase_b_current_mean | 0.0021 |
| inv_phase_c_current_mean | 0.0019 |
| delta_T_30s_prev | 0.0010 |

## 2. ΔT Regression – Gradient Boosting

- Model: **GradientBoostingRegressor**
- Test MAE (°C): **1.345**
- Test RMSE (°C): **3.884**
- Train samples: 2633
- Test samples: 659

**Top 10 features by importance (regression):**

| Feature | Importance |
|---|---|
| delta_T_30s_prev | 0.8607 |
| inv_control_board_temp_mean | 0.0207 |
| inv_hot_spot_temp_mean | 0.0178 |
| inv_gate_driver_board_temp_mean | 0.0153 |
| inv_hot_spot_temp_min | 0.0120 |
| inv_module_c_temp_max | 0.0107 |
| inv_module_c_temp_mean | 0.0084 |
| inv_coolant_temp_min | 0.0073 |
| inv_phase_c_current_max | 0.0072 |
| inv_dc_bus_voltage_mean | 0.0060 |

## 3. Control Heuristics Derived from the Models

Based on the advanced models, we propose simple torque derating
heuristics that could be implemented in the vehicle controller:

- Let **p_overheat** be the model's predicted probability of overheating
  over the next ~30 seconds, and **ΔT_pred** be the predicted temperature
  rise over the next 30 seconds.
- Let **T_hotspot** be the current inverter hot-spot temperature.

Suggested rules:

- If `p_overheat > 0.7` **and** `ΔT_pred > 8°C`:
  - Reduce torque request by ~20–30% for the next 30s.
  - Optionally limit DC current to a conservative cap.

- If `p_overheat > 0.9` **or** `T_hotspot > 65°C`:
  - Apply a stronger derate (e.g., 40–50%) and issue a driver warning.

- If `p_overheat < 0.3` **and** `ΔT_pred < 3°C`:
  - Full performance is allowed; no derating needed.

These heuristics translate the advanced ML models into actionable
safety/performance trade-offs for the FSAE accumulator and inverter.
