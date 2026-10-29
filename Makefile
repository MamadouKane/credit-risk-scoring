PYTHON ?= .venv/bin/python

.PHONY: setup data features train evaluate serve test lint docker-build docker-run all

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

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

all: features train evaluate  ## full pipeline after `make data`

docker-build:  ## image of the scoring API (needs models/ from `make train`)
	docker build -t credit-risk-scoring .

docker-run:
	docker run --rm -p 8000:8000 credit-risk-scoring
