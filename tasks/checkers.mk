PYTHON ?= .venv/bin/python

.PHONY: check lint typecheck test

check: lint typecheck test

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest