## DS3000 Battery Analyzer – Design Document

### 1. Project Overview
- **Purpose**: Detect and predict inverter overheating or power-delivery anomalies by correlating temperature readings with current draw, phase currents, and motor speed. Provide both historical analysis (offline model) and a lightweight control concept for live data.
- **Scope**: Use the provided inverter telemetry (temps, DC voltage/current, phase currents, torque & speed) stored under `data/`. Build a training/evaluation pipeline plus a prototype inference routine that could ingest a live stream.

### 2. Requirements & Success Criteria
- **Core model**: Learn a mapping from electrical load indicators (phase currents, DC bus current, speed) to temperature outcomes. Support two modes:
  1. Classification (safe vs. overheating risk) with adjustable thresholds.
  2. Regression predicting future temperature delta under a forecasted current draw.
- **Explainability**: Produce feature importance or SHAP-style rationale for operational insight.
- **Control heuristic**: Document (and optionally implement) a feedback rule that adjusts current draw/power request to keep predicted temperatures under a configurable ceiling.
- **Performance targets**: ≥90% recall for overheating-risk classification on validation splits and ≤2 °C MAE for short-horizon temperature forecasts (targets can be revised after baseline).
- **Operational deliverables**: Cleaned dataset artifacts, reusable training scripts/notebooks, evaluation report, and README instructions for running inference.

### 3. Data Inventory (from `data/`)
| File | Signals | Notes |
| --- | --- | --- |
| `Inverter Temps-data-as-joinbyfield-2025-10-03 17_39_12.csv` | Control board temp, coolant temp, gate driver temp, module temps (A/B/C), hotspot | Primary dependent variables; need unit verification and sensor alignment. |
| `Inverter Voltage and Current-data-as-joinbyfield-2025-10-03 17_38_50.csv` | DC bus voltage/current | Ensure units parsed (values include text like `Volts`, `Amps`). |
| `Inverter Phase Currents-data-as-joinbyfield-2025-10-03 17_38_30.csv` | Three phase currents | Likely in Amps; time stamps align with other logs. |
| `Torque & Motor Speed-data-2025-10-03 18_30_00.csv` | Motor speed (RPM) | Currently zeroed; verify additional torque columns or future logs. |

### Data-Cleaning Plan 
  - Parsing & Normalization
      - Load every CSV with UTF‑8‑SIG and drop the leading sep=, marker; centralize this logic in a load_csv() helper so
        every file is handled consistently.
      - Strip unit strings (e.g., Volts, Amps) from numeric columns during ingest; coerce everything to floats and log rows
        that fail conversion.
      - Rename verbose headers to canonical snake_case (INV_Motor_Speed, inv_dc_bus_voltage, etc.) to keep schema
        consistent across files.
      - Add schema validation (expected columns, ranges) before persisting cleaned tables.
  - Timestamp Alignment
      - Parse Time in UTC (or local if provided) and note that samples exist at sub-second resolution but are truncated
        to seconds.
      - Within each file, group rows by second, preserving ordering to compute aggregates:
          - For currents/voltages: mean, min, max, and count per second to retain sub-second variability info.
          - For temperatures: prefer median per second to reduce spike influence.
      - Emit a consolidated 1 Hz index covering the full 63‑minute window; outer-join per-signal aggregates onto this index
        to guarantee alignment.
      - Store optional “sample_count” columns so downstream modeling can weigh seconds with more observations.
  - Missing Values & Sensor Integrity
      - Identify structured gaps (e.g., temp dropouts). Use forward-fill capped at N seconds, then spline/linear
        interpolation only if neighboring values are within plausible deltas; otherwise leave NaN and mark for exclusion.
      - For phase currents/voltages, treat missing values as true gaps—no interpolation, just flag the second for removal
        or cautious imputation.
      - Temperature anomaly checks: ensure readings stay within physical bounds (e.g., 0–120 °C); quarantine any outliers
        for manual review.
      - Idle DC-current stretches: keep them but tag via a is_idle boolean to help with class balancing later.
  - Dataset Integration & Exclusions
      - Join the three June 3 logs on the unified 1 Hz timestamp to create the main modeling table. Record provenance
        metadata (source files, aggregation method).
      - Exclude the May 20 torque/speed file from the core dataset until matching telemetry exists; instead, stage it
        separately for future experiments.
      - After merging, drop seconds missing critical targets (e.g., hotspot temp) unless imputation produced high-
        confidence values.
      - Save cleaned artifacts (e.g., clean/inverter_thermal_1hz.parquet) and run automated QA:
          - Row counts vs. expected 3 324 timestamps.
          - No NaNs in required predictor/target columns unless explicitly allowed.
          - Summary stats to confirm ranges didn’t drift during cleaning.
      - Document every assumption (interpolation window, unit conversions) in README_cleaning.md to keep the pipeline
        reproducible.

### 4. System Architecture & Workflow
1. **Data ingestion layer**: Standardize CSV parsing (strip BOM, convert units, harmonize timestamps). Output parquet/clean CSV for downstream work.
2. **Data quality & exploratory analysis**: Visualize correlations, detect missing data, derive target labels (e.g., `temp_hotspot > threshold`).
3. **Feature engineering**:
   - Aggregate rolling statistics (mean, slope) over 1–5 sec windows.
   - Derive combined load indicators (e.g., RMS phase current, power estimates from voltage×current, mechanical power via torque×speed).
   - Encode ambient/contextual variables (coolant temp baseline, control board temp).
4. **Model training**:
   - Baseline algorithms: Linear regression/logistic regression and gradient boosted trees.
   - Hyperparameter search using cross-validation or time-based splits.
   - Track metrics via MLflow/W&B (optional) or simple CSV logs.
5. **Evaluation & validation**:
   - Hold-out validation covering different drive cycles.
   - Stress-test predictions under simulated current spikes.
   - Generate confusion matrices, ROC, MAE charts.
6. **Inference & control prototype**:
   - Batch inference script (`predict.py`) taking CSV input.
   - Optional streaming loop reading from stdin/socket to emulate live data.
   - Control heuristic outputs recommended current derate percentage when overheating risk exceeds threshold.
7. **Deployment/readiness artifacts**:
   - Documented reproducible pipeline commands.
   - Summary report describing findings and control recommendations.

### 5. Milestones & Collaborative Task Breakdown

| # | Name | Primary Owners | Expected Duration | Key Deliverables | Dependencies |
|---|------|----------------|-------------------|------------------|--------------|
| 1 | Data Audit & Infrastructure | Data Engineering | Week 1 | Standardized loaders, aggregation scripts, QA report | None |
| 2 | Exploratory Analysis & Target Definition | Analytics | Week 2 | EDA notebook, labeling spec, data dictionary | Milestone 1 |
| 3 | Baseline Modeling | Modeling Team A | Week 3 | Baseline models + metrics, preprocessing pipeline | Milestones 1–2 |
| 4 | Advanced Modeling & Control Logic | Modeling Team B | Weeks 4–5 | Tuned models, interpretability artifacts, control prototype | Milestones 1–3 |
| 5 | Integration, Validation, Reporting | Integration/QA | Week 6 | Unified CLI, test results, final report & deck | Milestones 1–4 |

#### Milestone 1 – Data Audit & Infrastructure (Owner: Data Engineering)
- **Tasks**
  - Finalize ingestion utilities (`src/data_loading.py`) and timestamp aggregation (`src/aggregation.py`) with unit tests.
  - Script the merge process for June 3 logs and produce canonical artifacts (`clean/inverter_merged_1hz.csv` + parquet copy).
  - Develop QA automation covering row counts, missing data, sensor range validation (scripts under `scripts/qa_*`).
  - Document data-cleaning assumptions in `README_cleaning.md` and ensure reproducible commands (e.g., `PYTHONPATH=. python scripts/merge_inverter_data.py`).
- **Acceptance Criteria**
  - QA script reports zero duplicate timestamps and <0.1% missing rate for required fields.
  - Clean dataset committed to repo (if allowed) or instructions to regenerate provided.
  - Folder structure established (`src/`, `scripts/`, `notebooks/`, `clean/`, `artifacts/`) with `.gitignore` updates.

#### Milestone 2 – Exploratory Analysis & Target Definition (Owner: Analytics)
- **Tasks**
  - Create a Jupyter notebook demonstrating descriptive stats, correlations, and anomaly detection on the merged dataset.
  - Define overheating labels (e.g., `inv_hot_spot_temp_mean >= 65 °C`) and derive auxiliary targets (temperature delta over next N seconds).
  - Quantify class balance, highlight idle vs. active periods, and flag problematic sensors (e.g., zeroed motor speed).
  - Produce a data dictionary describing every feature/target, units, and preprocessing notes.
- **Acceptance Criteria**
  - Notebook checked into `notebooks/` with clear narrative and conclusions.
  - Label-generation code lives in `src/labels.py` or similar, callable from CLI.
  - Decision log capturing threshold choices and rationale shared with modeling teams.

#### Milestone 3 – Baseline Modeling (Owner: Modeling Team A)
- **Tasks**
  - Build preprocessing pipeline (train/test split respecting time order, scaling, feature selection) using scikit-learn.
  - Train logistic regression for classification and linear regression for temperature forecasting; capture metrics (recall, precision, MAE).
  - Implement evaluation script (`scripts/train_baseline.py`) that saves metrics + model artifacts to `models/baseline/`.
  - Add unit tests for feature pipelines and ensure reproducibility seeds are fixed.
- **Acceptance Criteria**
  - Baseline metrics meet or exceed minimal targets (e.g., ≥70% recall, ≤5 °C MAE) or gap analysis documented.
  - Model artifacts saved with metadata (training window, features used).
  - README section explaining how to run baseline training/evaluation end-to-end.

#### Milestone 4 – Advanced Modeling & Control Logic (Owner: Modeling Team B)
- **Tasks**
  - Experiment with gradient boosted trees (XGBoost/LightGBM/CatBoost) and temporal models (simple RNN/Temporal CNN) if justified by data.
  - Apply hyperparameter tuning (Optuna/grid search) with proper validation splits; log experiments (e.g., MLflow).
  - Generate interpretability outputs (feature importance charts, SHAP values) to explain overheating predictions.
  - Translate predictions into a control heuristic that recommends current derate percentages or cooling alerts; prototype in `scripts/control_loop.py`.
- **Acceptance Criteria**
  - Advanced models surpass baseline targets (≥90% recall, ≤2 °C MAE) or produce clear mitigation plan if not.
  - Control heuristic documented with pseudo-code, parameter choices, and safety constraints.
  - Interpretability artifacts saved (plots, CSVs) and referenced in documentation.

#### Milestone 5 – Integration, Validation, and Reporting (Owner: Integration/QA)
- **Tasks**
  - Package CLI entry points (e.g., `python scripts/predict.py --input xyz`) and, if applicable, a simple UI or dashboard.
  - Run regression tests: training from scratch, inference on holdout data, QA on output statistics. For any JavaScript components, run `npm test` per team agreement.
  - Compile final technical report + presentation summarizing data pipeline, models, control logic, and recommendations.
  - Outline deployment or future work plan (data needed, monitoring strategy).
- **Acceptance Criteria**
  - All scripts runnable via documented commands; README updated with quick start.
  - Validation log demonstrating clean training/inference runs plus QA sign-off.
  - Final report/deck reviewed by stakeholders with action items captured.

### 6. Collaboration Guidelines
- **Version control**: Separate feature branches per milestone; open PRs with linked tasks.
- **Artifacts**: Store intermediate datasets in `artifacts/` (git-ignored) and commit configs/scripts only.
- **Meet cadence**: Weekly sync to unblock, ad-hoc async updates through issue tracker.
- **Handoffs**: Each milestone owner delivers documentation + runnable scripts/tests before the next milestone begins.

### 7. Risks & Mitigations
- **Sparse/zero motor speed data**: Plan to simulate wheel-speed scenarios or gather more logs; otherwise down-weight feature.
- **Data drift**: Keep ingestion modular to drop in newer CSVs; document schema.
- **Model generalization**: Use cross-session validation to avoid overfitting to a single timestamp batch.
- **Control safety**: Treat recommendations as advisory; include guardrails to prevent aggressive power cuts.
