#!/usr/bin/env bash
# Run from this folder. Requires ../Bhuvan_Rajpoot_SSRN with data/raw/diabetic_data.csv (see README).
set -euo pipefail
PY=${PYTHON:-.venv/bin/python}
mkdir -p logs
$PY scripts/00_download_cdc.py
$PY scripts/01_fit_models.py      | tee logs/01_fit_models.log
$PY scripts/02_rerun_rq1_cdc.py   | tee logs/02_rerun_rq1_cdc.log
$PY scripts/03_metrics.py         | tee logs/03_metrics.log
$PY scripts/04_shap.py            | tee logs/04_shap.log
