PYTHON ?= python
PYTHONPATH := PYTHONPATH=.

MERGED ?= src/notebooks/clean/inverter_merged_1hz.csv
LABELED ?= src/notebooks/clean/inverter_labeled_1hz.csv

baseline: data label train

advanced:
	$(PYTHONPATH) $(PYTHON) src/scripts/training/train_boosted_models.py --input $(LABELED)

test:
	$(PYTHONPATH) $(PYTHON) -m unittest discover tests

clean:
	rm -f models/baseline/*.joblib
	rm -rf predictions

data:
	$(PYTHONPATH) $(PYTHON) src/scripts/data/merge_inverter_data.py --output $(MERGED)
	$(PYTHONPATH) $(PYTHON) src/scripts/data/qa_merged_data.py --input $(MERGED)

label:
	$(PYTHONPATH) $(PYTHON) src/labels.py

train:
	$(PYTHONPATH) $(PYTHON) src/scripts/training/train_logreg_baseline.py --input $(LABELED)
	$(PYTHONPATH) $(PYTHON) src/scripts/training/train_ridge_baseline.py --input $(LABELED)

infer:
	$(PYTHONPATH) $(PYTHON) src/scripts/inference/predict_overheat.py --input $(LABELED) --output predictions
	$(PYTHONPATH) $(PYTHON) src/scripts/inference/predict_delta.py --input $(LABELED) --output predictions
	$(PYTHONPATH) $(PYTHON) src/scripts/inference/predict_boosted.py --input $(LABELED) --output predictions/advanced

.PHONY: baseline data label train infer test clean advanced
