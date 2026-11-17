# Salim & Ibrahim (Healthcare 2026) — Reproduction

This repository reproduces the method described in:

> S. S. Salim and A. A. Ibrahim, “A machine learning approach for predicting
> 30-day hospital readmission in patients with diabetes,” *Healthcare*, vol. 14,
> no. 9, Art. no. 1185, 2026. DOI: 10.3390/healthcare14091185.

Status: **complete full run** on all 101,766 encounters. The saved results use
the 54-predictor feature table specified by official Supplementary Table S1.

詳細的中文方法、論文逐項對照、數值比較與限制請見
[`REPRODUCTION_REPORT_zh-TW.md`](REPRODUCTION_REPORT_zh-TW.md)。

## What is implemented

- UCI source download with a pinned SHA-256 integrity check.
- Binary target: `<30` = 1; `>30` and `NO` = 0.
- Cleaning, missing-category handling, age midpoint conversion, top-10 medical
  specialty reduction, diagnosis flags, grouped administrative variables, and
  every engineered variable listed in Supplementary Table S1.
- Leakage-safe `Pipeline` / `ColumnTransformer`, z-score scaling and one-hot
  encoding.
- Cost-sensitive Logistic Regression, Random Forest, XGBoost and LightGBM.
- Logistic/XGBoost/LightGBM stacking with a Logistic Regression meta-learner.
- XGBoost 5-outer/3-inner nested cross-validation and grid search.
- Platt calibration, 1000 fixed-prediction bootstrap resamples, risk tertiles,
  threshold analysis, decision curve analysis and clinical impact curve.
- TreeSHAP global explanation and 100-bootstrap top-10 ranking stability.
- Machine-readable predictions/tables, figures, and serialized fitted models.

## Reproduce

The completed dedicated environment is at `.conda/` in the delivered folder.
To rerun there:

```bash
cd /media/amazon/E/zerone/DSP/salim-ibrahim-2026-readmission
conda run -p ./.conda python -m pytest -q
MPLCONFIGDIR=/tmp/salim-readmission-matplotlib \
  conda run --no-capture-output -p ./.conda python reproduce.py
```

To build a fresh environment:

```bash
conda env create -p ./.conda -f environment.yml
conda run -p ./.conda python scripts/download_data.py
conda run -p ./.conda python -m pytest -q
conda run --no-capture-output -p ./.conda python reproduce.py
```

The full run defaults to 5×3 nested CV, 1000 bootstrap resamples and 100 SHAP
stability refits. `python reproduce.py --quick` is only an end-to-end smoke test
and must not be reported as the paper reproduction.

## Main outputs

- `results/tables/model_comparison.csv`: outer-fold comparison of five models.
- `results/tables/out_of_fold_predictions.csv`: auditable OOF predictions.
- `results/tables/paper_vs_reproduction.csv`: paper/reproduction comparison.
- `results/tables/bootstrap_summary.csv`: 1000-bootstrap percentile intervals.
- `results/tables/threshold_metrics.csv`: 5%–30% operating points.
- `results/tables/shap_global_importance.csv`: mean absolute TreeSHAP ranking.
- `results/tables/shap_top10_stability.csv`: 100-refit top-10 frequency.
- `splits/outer_fold_assignments.csv`: exact five outer test-fold assignments.
- `splits/inner_fold_assignments.csv`: exact three inner validation assignments
  for every outer training set.
- `splits/split_summary.json`: seeds, class counts and source-data hash.
- `results/figures/`: ROC, PR, calibration, DCA, CIC, risk and SHAP figures.
- `artifacts/models/`: uncalibrated and Platt-calibrated XGBoost models.
- `docs/Supplementary_Table_S1.pdf`: official open-access feature table.

## Reproducibility boundary

The paper does not publish code, the candidate values in its XGBoost grid,
random seeds, exact ID-to-group lookup sets, or baseline hyperparameters. Those
choices are explicit in this repository and summarized in
[`ASSUMPTIONS_AND_PAPER_ISSUES.md`](ASSUMPTIONS_AND_PAPER_ISSUES.md). Results
should therefore be described as a close independent reproduction, not an exact
execution of the authors’ unavailable implementation.

