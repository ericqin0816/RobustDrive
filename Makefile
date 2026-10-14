.PHONY: install test smoke train-baseline train-augmented figures lint
PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src scripts tests
smoke:
	$(PYTHON) scripts/smoke.py
train-baseline:
	$(PYTHON) scripts/train.py --config configs/baseline.yaml
train-augmented:
	$(PYTHON) scripts/train.py --config configs/augmented.yaml
figures:
	$(PYTHON) scripts/make_figures.py
