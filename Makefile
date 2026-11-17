.PHONY: download test reproduce quick clean-results

PYTHON ?= python

download:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest -q

quick: download
	$(PYTHON) reproduce.py --quick

reproduce: download
	$(PYTHON) reproduce.py

clean-results:
	rm -rf results/* artifacts/models/*

