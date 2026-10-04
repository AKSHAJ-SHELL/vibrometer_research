.PHONY: setup reproduce handoff check test test-draft data experiment-cwru real figures-real features experiment figures all clean

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-draft:
	$(PYTHON) -m pytest tests_draft/ -v --tb=short

data:
	$(PYTHON) scripts/download_data.py --dataset all

real:
	$(PYTHON) scripts/run_real.py

# from scratch: re-extract every feature table, rerun everything, redraw, record provenance
reproduce:
	$(PYTHON) -u scripts/run_real.py --refresh
	$(PYTHON) scripts/make_real_figures.py
	$(PYTHON) scripts/write_provenance.py

handoff:
	$(PYTHON) scripts/make_handoff.py

figures-real:
	$(PYTHON) scripts/make_real_figures.py

experiment-cwru:
	$(PYTHON) scripts/run_experiment.py --dataset cwru

features:
	$(PYTHON) scripts/build_features.py --dataset synthetic --out results/features_synthetic.npz

experiment:
	$(PYTHON) scripts/run_experiment.py --config configs/default.yaml

figures:
	$(PYTHON) scripts/make_figures.py --out figures --synthetic

check:
	$(PYTHON) -m pytest tests/ tests_draft/ -v --tb=short

all: check features experiment figures

clean:
	rm -rf results/* figures/*.png figures/*.json __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
