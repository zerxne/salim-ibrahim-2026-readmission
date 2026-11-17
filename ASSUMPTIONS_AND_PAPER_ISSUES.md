# Assumptions and paper inconsistencies

## Explicit reconstruction choices

1. The random seed is 42. The paper reports no seed.
2. The XGBoost grid is `n_estimators=[100, 200]`, `max_depth=[3, 5]`, and
   `learning_rate=[0.03, 0.1]`. The paper names these three dimensions but does
   not publish their candidate values.
3. Five-fold stacking is used because this is the scikit-learn default concept
   described by the paper; the internal stacking fold count is not reported.
4. Administrative IDs are grouped from the UCI `IDS_mapping.csv` descriptions.
   Supplementary Table S1 gives the four output labels but not the exact member
   IDs for every label.
5. The ten most frequent `medical_specialty` values are calculated on the
   complete cleaned cohort, including `Unknown`. The paper does not list them.
6. Baseline models use class weighting as required by the paper and otherwise
   common/default settings. Their complete hyperparameter configurations are not
   reported in the paper.
7. SHAP stability refits the XGBoost classifier 100 times on bootstrap samples
   of the fixed encoded design matrix. This avoids changing one-hot columns
   across refits. The paper says the model was retrained but does not specify
   whether preprocessing was also refitted in every bootstrap.

## Internal inconsistencies in the paper

- Main-text Section 3.1.5/Table 2 says 55 predictors; official Supplementary
  Table S1 says 54. Applying every documented removal/addition gives 54, which
  this implementation uses.
- The abstract attributes AUC 0.664 to stacking. Table 3 reports stacking 0.665
  and XGBoost 0.664.
- The text calls 0.665 versus 0.664 a difference of 0.007; the arithmetic
  difference is 0.001.
- Table 4 and Table 5 report incompatible metrics at the same 10% threshold.
  For example, sensitivity is 0.723 in Table 4 but 0.881 in Table 5.
- The prose reports PR-AUC 0.215, while the Figure 4 legend shows 0.217.
- AUC 0.688 and Brier 0.094 are evaluated after full-data training/calibration
  on that same complete cohort. They are apparent estimates and cannot be
  compared directly with the nested out-of-fold AUC. The paper acknowledges the
  evaluation-strategy difference but still calls this “final model performance.”

## Interpretation boundary

This is an independent computational reproduction on a historical
administrative dataset. It is not external clinical validation and is not a
deployable clinical decision-support device. Planned versus unplanned
readmissions cannot be separated, and no contemporary or institution-held-out
cohort is available.

