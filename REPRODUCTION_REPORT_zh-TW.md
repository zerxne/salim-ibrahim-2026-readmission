# 30 天糖尿病住院再入院預測：論文復現報告

## 1. 結論

本專案已用完整 UCI Diabetes 130-US Hospitals 資料（101,766 encounters）完成
Salim 與 Ibrahim（2026）的獨立復現。資料數量、陽性率、模型排序、ROC-AUC、
Brier score、F1、風險分層與 DCA 有高度一致性；因此可以說**主要結果已成功近似
復現**。因作者未公開程式，且論文缺少 grid 候選值、seed 與部分 grouping lookup，
不應宣稱 bit-for-bit exact reproduction。

最強的重現證據：

- Random Forest ROC-AUC：0.6504（論文 0.650）。
- XGBoost F1：0.2716（論文 0.27）。
- calibrated Brier：0.09358（論文 0.094）。
- risk tertiles：4.47% / 9.70% / 19.31%（論文約 4–5% / 9–10% / 18–19%）。
- DCA 同時優於 Treat-All 與 Treat-None：1.586%–30% threshold（論文約
  1.6%–30%）。

## 2. 資料與 target

| 項目 | 論文 | 本實作 |
|---|---:|---:|
| encounters | 101,766 | 101,766 |
| `<30` positive | 11,357 (11.16%) | 11,357 (11.1599%) |
| negative (`>30` 或 `NO`) | 90,409 | 90,409 |
| identifiers | 移除 encounter_id、patient_nbr | 相同 |
| 高缺失欄位 | 移除 weight、payer_code | 相同 |
| final predictors | 正文 55；Table S1 54 | 54（採官方 Table S1） |

下載器會驗證 UCI ZIP SHA-256：
`f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a`。

## 3. 論文方法與程式對照

| 論文描述 | 本實作 | 位置 |
|---|---|---|
| `?` → missing，類別補 `Unknown` | 完整實作 | `src/features.py` |
| age interval → midpoint | `[60-70)` → 65 等 | `age_midpoint()` |
| diag_1–3 → clinical flags | diabetes/circulatory/respiratory/digestive/injury/cancer | `diagnosis_group()` |
| admin IDs 分組 | admission type、discharge、source 四分類 | grouping functions |
| top-10 medical specialty + Other | 含 Unknown 的最高頻十類，其餘 Other | `engineer_features()` |
| Table S1 composite features | total/acute visits、disease burden、polypharmacy、long stay、hospital pressure | `engineer_features()` |
| numeric z-score + categorical one-hot | `StandardScaler` + `OneHotEncoder(handle_unknown='ignore')` | `src/modeling.py` |
| cost-sensitive learning | XGBoost 每個 fold 動態算 `scale_pos_weight`；其他模型 balanced | `FoldBalancedXGBClassifier` |
| 5 outer / 3 inner nested CV | stratified、shuffle、seed 42 | `nested_xgboost()` |
| XGBoost grid search | trees × depth × learning rate，共 8 組 | `nested_xgboost()` |
| Logistic/RF/XGB/LGBM/stacking | 全部實作；stack meta-learner 為 Logistic Regression | `make_models()` |
| Platt scaling | `CalibratedClassifierCV(method='sigmoid', cv=5)` | `reproduce.py` |
| 1000 bootstrap | AUC、Brier、high/low RR percentile CI | `bootstrap_intervals()` |
| DCA / CIC | 1%–30%，輸出 CSV 與圖 | `decision_curve()` |
| TreeSHAP + stability | 5,000 筆 summary；100 次 bootstrap refit top-10 | `shap_analysis()` |

## 4. 主要實驗結果

### 4.1 Outer-fold model comparison

| Model | 論文 ROC-AUC | 復現 mean ROC-AUC | fold SD | 差值 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.657 | 0.6612 | 0.0076 | +0.0042 |
| Random Forest | 0.650 | 0.6504 | 0.0060 | +0.0004 |
| XGBoost | 0.664 | 0.6700 | 0.0064 | +0.0060 |
| LightGBM | 0.660 | 0.6669 | 0.0057 | +0.0069 |
| Stacking | 0.665 | 0.6703 | 0.0067 | +0.0053 |

與論文相同，stacking 只有非常小的領先；XGBoost 因可校準與 TreeSHAP 解釋而保留
為主模型。五個 XGBoost outer folds 是 0.6650、0.6791、0.6681、0.6739、
0.6637，五折都選到：`n_estimators=200, max_depth=5, learning_rate=0.03`。

將五折 OOF prediction 合併後，XGBoost ROC-AUC = 0.66978、PR-AUC =
0.21229、F1@0.5 = 0.27163。PR-AUC 與論文文字的 0.215 很接近；F1 幾乎一致。

### 4.2 Full-data calibration（apparent estimate）

| Metric | 論文 | 復現 | 差值 |
|---|---:|---:|---:|
| ROC-AUC | 0.688 | 0.69607 | +0.00807 |
| Brier score | 0.094 | 0.09358 | −0.00042 |
| PR-AUC | 0.215（Figure 4 為 0.217） | 0.24208 | +0.02708 |

這組數字和論文一樣，是在完整 cohort 上訓練/校準後，又對完整 cohort 做 prediction
的 apparent estimate；不能當成外部或 nested out-of-fold generalization。Platt calibration
不應改變同一批 score 的排序；AUC 變高主要來自評估資料重用，而不是 calibration 本身。

### 4.3 10% threshold

| Metric | 論文 Table 4 | 復現 |
|---|---:|---:|
| Sensitivity | 0.723 | 0.6939 |
| Specificity | 0.530 | 0.5787 |
| PPV | 0.162 | 0.1714 |
| NPV | 0.938 | 0.9377 |
| High risk / 1000 | 497.9 | 451.8 |
| True readmissions / 1000 | 80.6 | 77.4 |

NPV 與每千人 true readmissions 很接近；敏感度與 flagged population 有差異。論文
Table 5 對同一 10% threshold 又報 0.881 sensitivity、0.524 specificity、0.189 PPV、
0.972 NPV、521.0 high risk/1000，和 Table 4 本身無法同時成立，因此沒有單一實作
能同時重現兩組數字。

### 4.4 Risk stratification 與 bootstrap

| Tertile | Encounters | Observed readmission | Mean predicted risk |
|---|---:|---:|---:|
| Low | 33,922 | 4.469% | 5.669% |
| Medium | 33,922 | 9.705% | 9.572% |
| High | 33,922 | 19.306% | 18.630% |

1000 次 fixed-prediction bootstrap percentile intervals：

- AUC 0.69607，95% CI [0.69097, 0.70088]。
- Brier 0.09358，95% CI [0.09230, 0.09500]。
- high/low relative risk 4.320，95% CI [4.072, 4.555]。

論文說 high-risk 約是 low-risk 的 3.9–4.0 倍；復現值稍高，但分層梯度與各組實際
readmission rate 高度一致。

### 4.5 Clinical utility

在 100 個 1%–30% threshold points 中，從 1.586% 到 30% 的 model net benefit
同時大於 Treat-All 與 Treat-None，共 98 個點。這幾乎精確重現論文敘述的
1.6%–30%。10% 附近約每千人 flag 448 人，其中約 77 人是真正 30-day readmission。

### 4.6 SHAP 與穩定性

復現的主要特徵包括 discharge group、hospital pressure、acute utilization、prior
inpatient visits、number of diagnoses、total visits、age、length of stay、diabetes
medication 與 number of medications，和論文 Figure 3 的核心臨床訊號一致。

100 次 bootstrap refit 中，以下特徵進入 top-10 的頻率最高：

- discharge expired / transfer、hospital pressure：100%。
- acute utilization、number of diagnoses、number of inpatient visits：99%。
- age：87%；discharge home：77%；length of stay：62%。

排序並非完全相同；最可能原因是論文沒有公開 grid 候選值、seed、完整 admin ID
mapping 與最終參數。完整排名在 `results/tables/shap_global_importance.csv`。

## 5. 驗證狀態

- `pytest`: 3 passed。
- `runtime_versions.json`: `quick_mode=false`。
- Python 3.12.7、scikit-learn 1.8.0、XGBoost 3.1.2、LightGBM 4.6.0、
  SHAP 0.50.0，與論文列出的主要版本一致。
- 1000 bootstrap rows、100 SHAP stability refits、五個 outer folds 均已保存。
- 兩個 fitted model artifacts 已保存並可重新載入。
- `splits/` 已保存完整 5 outer × 3 inner row-level assignments；這篇論文採
  nested CV，因此不存在唯一的一組 train/test CSV。每個 outer fold 的 test rows
  由 `outer_test_fold` 指定，inner validation rows 由 `inner_validation_fold` 指定。

## 6. 可直接使用的英文摘要

> We independently reproduced the machine-learning framework of Salim and
> Ibrahim (2026) using all 101,766 encounters from the UCI Diabetes 130-US
> Hospitals dataset. Preprocessing and feature engineering followed official
> Supplementary Table S1, yielding 54 predictors. Under five-outer/three-inner
> nested cross-validation, XGBoost achieved a mean ROC-AUC of 0.670 (reported:
> 0.664), while stacking achieved 0.670 (reported: 0.665). The calibrated
> full-cohort apparent Brier score was 0.0936 (reported: 0.094). Observed
> readmission rates increased from 4.47% in the low-risk tertile to 19.31% in
> the high-risk tertile. Decision-curve analysis showed positive net benefit
> over Treat-All and Treat-None from approximately 1.6% to 30%, closely matching
> the paper. Differences are attributable to unreported random seeds, grid
> values, baseline hyperparameters, and administrative-code mappings; therefore,
> this work should be described as a close independent reproduction rather than
> an exact rerun of unavailable author code.

