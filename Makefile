PYTHON ?= .venv/bin/python
DAY ?= 01

.PHONY: setup data features train evaluate serve test check lint

setup:  ## create .venv (Python 3.11) and install the project with dev tools
	uv venv --python 3.11 .venv
	uv pip install --python $(PYTHON) -e ".[dev]"
	$(PYTHON) -m ipykernel install --user --name credit-risk --display-name "credit-risk (Python 3.11)"

data:  ## download the Kaggle files into data/raw
	$(PYTHON) scripts/download_data.py

features:  ## build the modelling table
	$(PYTHON) -m credit_risk.features

train:  ## train the scorecard and the XGBoost challenger
	$(PYTHON) -m credit_risk.train

evaluate:  ## final evaluation on the test set (run once)
	$(PYTHON) -m credit_risk.evaluate

serve:  ## run the scoring API
	$(PYTHON) -m uvicorn credit_risk.api.main:app --reload

test:  ## run every test
	$(PYTHON) -m pytest

check:  ## run only one day's tests: make check DAY=03
	$(PYTHON) -m pytest tests/test_d$(DAY)_*.py -v

lint:
	$(PYTHON) -m ruff check src tests scripts
