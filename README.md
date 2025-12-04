## DS3000 Battery Analyzer

Early warning system for inverter overheating and power-delivery anomalies. We ingest inverter telemetry (temperatures, DC bus voltage/current, phase currents, and optional torque/speed) to train models that forecast thermal issues and recommend power-derating actions.

### Repository Layout
- `src/`
  - `data_loading.py` – CSV ingestion utilities (handles BOM, `sep=,`, unit stripping, canonical names).
  - `aggregation.py` – Timestamp parsing + 1 Hz aggregation helpers.
  - `scripts/data/`
    - `merge_inverter_data.py` – Builds unified per-second dataset from June 3 logs.
    - `qa_merged_data.py` – Quality checks (row counts, missing data, ranges).
    - `validate_data_loading.py` – Confirms loader/aggregation on current `data/` contents.
  - `scripts/training/`
    - `train_logreg_baseline.py` – Fits and persists the logistic overheating baseline.
    - `train_ridge_baseline.py` – Fits and persists the ridge delta-T baseline.
  - `scripts/inference/`
    - `predict_overheat.py` – Loads the persisted logistic baseline and emits timestamped probabilities/labels.
    - `predict_delta.py` – Runs inference for the ridge baseline.
  - `labels.py` – Generates `overheat_label` and 30s delta targets.
  - `notebooks/` – Analysis artifacts:
    - `01_eda_template.ipynb` – Full exploratory analysis, sensor QA, rolling features, charting.
    - `02_baseline_modeling.ipynb` – Logistic and ridge baselines with chronological splits and metrics.
  - `clean/` – Generated artifacts (`clean/inverter_merged_1hz.csv`, labeled variants).
- `models/baseline/` – Persisted artifacts (e.g., `logreg_overheat.joblib`).
- `predictions/` – Example inference outputs (created by `predict_overheat.py`).
- `data/` – Raw CSV exports from the inverter logger.
- `tests/` – `unittest` coverage for loader and aggregation utilities (`PYTHONPATH=. python -m unittest discover tests`).
- `DESIGN.md` – Project requirements, milestones, collaboration plan.
- `AGENTS.md` – Working agreements and scope reminders.
- `pyproject.toml` – Editable install definition (use `PYTHONPATH=.` if pip install isn’t available).

### Key Goals
1. **Data Pipeline (Milestone 1)** – Standardize ingestion, produce clean 1 Hz dataset, automate QA (done; document assumptions next).
2. **EDA + Target Definition (Milestone 2)** – Analyze thermal/electrical relationships, set overheating thresholds, label data.
3. **Baseline Modeling (Milestone 3)** – Train logistic/linear baselines with reproducible scripts, persist artifacts, and expose inference tooling.
4. **Advanced Modeling & Control (Milestone 4)** – Boosted trees/temporal models, interpretability, derive control heuristics.
5. **Integration & Reporting (Milestone 5)** – Package CLI/inference workflow, validate end-to-end, deliver final report.

Refer to `DESIGN.md` §5 for detailed tasks, owners, and acceptance criteria per milestone.

### Quick Start
1. Activate the venv and set project root on `PYTHONPATH`:
   ```bash
   source venv/bin/activate
   export PYTHONPATH=.
   ```
2. Regenerate merged dataset:
   ```bash
   python src/scripts/data/merge_inverter_data.py --output src/notebooks/clean/inverter_merged_1hz.csv
   ```
3. Label data + QA:
   ```bash
   python src/labels.py  # writes src/notebooks/clean/inverter_labeled_1hz.csv
   python src/scripts/data/qa_merged_data.py --input src/notebooks/clean/inverter_merged_1hz.csv
   ```
4. Execute tests:
   ```bash
   python -m unittest discover tests
   ```
5. Train/persist baselines:
   ```bash
   PYTHONPATH=. python src/scripts/training/train_logreg_baseline.py
   PYTHONPATH=. python src/scripts/training/train_ridge_baseline.py
   ```
6. Run inference with persisted artifacts:
   ```bash
   PYTHONPATH=. python src/scripts/inference/predict_overheat.py --output predictions
   PYTHONPATH=. python src/scripts/inference/predict_delta.py --output predictions
   ```

### Current Status & Next Steps
- ✅ Milestone 1: ingestion, aggregation, QA, initial scripts/tests.
- ✅ Milestone 2: EDA notebook, labels, class-balance report, data dictionary.
- ✅ Milestone 3 (baseline phase): logistic & ridge baselines documented in notebooks; both models have training CLIs and persisted artifacts (`models/baseline/logreg_overheat.joblib`, `models/baseline/ridge_deltaT.joblib`) plus inference tools.
- 🔜 Add ridge-regression persistence/inference and automated training script for reproducibility.
