PY ?= python

.PHONY: install test lint format notebook zip check clean

install:  ## editable install with dev + notebook extras
	$(PY) -m pip install -e ".[dev,notebook]"

test:
	$(PY) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

notebook:  ## regenerate notebooks/01_explore_graphs.ipynb
	$(PY) scripts/build_notebook.py

check: lint test

zip:  ## source-only archive (git-tracked files; never includes data/)
	git archive --format=zip --prefix=patchgraph-qa/ -o ../patchgraph-qa-day3.zip HEAD
	@echo "wrote ../patchgraph-qa-day3.zip (Colab also accepts any patchgraph-qa-day*.zip)"

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
