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
    - `train_boosted_models.py` – Trains Gradient Boosting classifier/regressor and writes Milestone 4 summary.
  - `scripts/inference/`
    - `predict_overheat.py` – Loads the persisted logistic baseline and emits timestamped probabilities/labels.
    - `predict_delta.py` – Runs inference for the ridge baseline.
    - `predict_boosted.py` – Produces predictions for the advanced Gradient Boosting models.
  - `labels.py` – Generates `overheat_label` and 30s delta targets.
  - `notebooks/` – Analysis artifacts:
    - `01_eda_template.ipynb` – Full exploratory analysis, sensor QA, rolling features, charting.
    - `02_baseline_modeling.ipynb` – Logistic and ridge baselines with chronological splits and metrics.
  - `clean/` – Generated artifacts (`clean/inverter_merged_1hz.csv`, labeled variants).
- `models/baseline/` – Persisted baseline artifacts (e.g., `logreg_overheat.joblib`).
- `models/advanced/` – Gradient Boosting classifier/regressor artifacts.
- `predictions/` – Inference outputs (`overheat_predictions.csv`, `delta_predictions.csv`, and `advanced/advanced_predictions.csv`).
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
2. Rebuild the data + train baselines using the Makefile:
   ```bash
   make baseline
   ```
   (This runs data ingestion, labeling, and both baseline training scripts.)
3. Execute tests:
   ```bash
   make test
   ```
4. Run inference with persisted artifacts (baseline + advanced):
   ```bash
   make infer
   ```
5. Clean model/prediction artifacts:
   ```bash
   make clean
   ```

### Current Status & Next Steps
- ✅ Milestone 1: ingestion, aggregation, QA, initial scripts/tests.
- ✅ Milestone 2: EDA notebook, labels, class-balance report, data dictionary.
- ✅ Milestone 3 (baseline phase): logistic & ridge baselines documented in notebooks; both models have training CLIs and persisted artifacts (`models/baseline/logreg_overheat.joblib`, `models/baseline/ridge_deltaT.joblib`) plus inference tools.
- ⚙️ Milestone 4 (advanced phase): Gradient Boosting models train via `make advanced`, metrics land under `metrics/gb_overheat_metrics.json` / `metrics/gbr_deltaT_metrics.json`, and `docs/baseline_metrics.md` + `advanced_modeling_and_control.md` capture the results. Automation still needs:
  1. Integrating boosted-model inference outputs into reporting (plots/QA).
  2. Wiring boosted metrics into dashboards (or comparing against thresholds) for CI alerts.
  3. Finalizing control-loop prototype and validation scenarios.
- 🔜 Milestone 5: packaging (command docs/README updates), final report, and deployment guidance once advanced heuristics are validated.
