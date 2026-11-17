# Salim & Ibrahim (Healthcare 2026) — Reproduction

This repository reproduces the method described in:

> S. S. Salim and A. A. Ibrahim, “A machine learning approach for predicting
> 30-day hospital readmission in patients with diabetes,” *Healthcare*, vol. 14,
> no. 9, Art. no. 1185, 2026. DOI: 10.3390/healthcare14091185.

Status: **complete full-data independent reproduction** on all 101,766
encounters. The main findings closely reproduce the paper, but this is not an
exact rerun because the authors did not publish their code, random seeds,
complete hyperparameter grid, or administrative-code mappings.

詳細的中文方法、論文逐項對照、數值比較與限制請見
[`REPRODUCTION_REPORT_zh-TW.md`](REPRODUCTION_REPORT_zh-TW.md)。

## What is implemented

- UCI source download with a pinned SHA-256 integrity check.
- Binary target: `<30` = 1; `>30` and `NO` = 0.
- Cleaning, missing-category handling, age midpoint conversion, top-10 medical
  specialty reduction, diagnosis flags, grouped administrative variables, and
  every engineered variable listed in Supplementary Table S1.
- Fold-local preprocessing for the primary nested-CV XGBoost pipeline using
  `Pipeline` / `ColumnTransformer`, z-score scaling, and one-hot encoding.
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

## Main results

| Model | Paper ROC-AUC | Reproduction ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.657 | 0.6612 |
| Random Forest | 0.650 | 0.6504 |
| XGBoost | 0.664 | 0.6700 |
| LightGBM | 0.660 | 0.6669 |
| Stacking | 0.665 | 0.6703 |

The nested out-of-fold XGBoost AUC was 0.6698. The full-cohort calibrated
apparent AUC was 0.6961 with a Brier score of 0.0936. The latter is an apparent
estimate evaluated on the same cohort used for model development and should
not be interpreted as external-validation performance.

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

The published validation is encounter-level rather than patient-grouped.
Because some patients have multiple encounters, approximately 41%–42% of test
encounters in each outer fold belong to patients who also appear in the
corresponding training set. This follows the paper's described procedure but
does not evaluate generalization to entirely unseen patients.

The paper also contains internally inconsistent predictor counts and
10%-threshold metrics. These issues are documented in
`ASSUMPTIONS_AND_PAPER_ISSUES.md`.