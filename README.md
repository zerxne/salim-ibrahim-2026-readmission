# 30 天再入院預測：指標補齊與 SHAP 延伸分析

糖尿病 30 天再入院專題（第 6 組）的後續工作，包含兩部分：

1. **檢查期末簡報缺少哪些指標並補齊**：AUROC、Brier、校準、F1／F2、Specificity、NPV、95% CI、子群體表現。
2. **SHAP 延伸分析**：對全部測試病歷做精確 TreeSHAP，加上類別層級效果、漏抓案例分析與穩健性檢查。

完整結果與建議修改的投影片見 **[REPORT_zh-TW.md](REPORT_zh-TW.md)**。

## 重點結果

所有數字皆在 Rajpoot 復現的病人互斥 80/20 測試集上計算（19,942 筆，2,297 筆再入院）。

- **CatBoost vs HGB**：AP +0.014 [+0.006, +0.022]、Brier 也顯著較好，但 **AUROC +0.003 [−0.002, +0.007] 不顯著**。進步集中在高風險端。
- **類別不平衡**：balanced weights 與複製正例讓**平均預測風險從 11% 變成 45%**，Brier 翻倍。兩種重抽樣的 AP 也都顯著低於原始比例。
- **校準**：Platt 後校準斜率 1.098 → 1.037、ECE 0.0078 → 0.0059。
- **子群體**：70 歲以上 AUROC 最低（0.65）；性別與主要種族之間差異小。
- **SHAP 前兩名穩定**：過去住院次數（2.80 pp）、出院去向（1.50 pp），合計占 45%。第 3–7 名數值接近，384 筆抽樣排不穩，第 40 頁建議改用全測試集。
- **保險別**的重要性主要來自「缺值」，可能反映登錄習慣，不是臨床風險。
- **漏抓案例**：過去住院 0 次層中，被抓到的病人幾乎都靠出院去向（+13 pp）。漏抓的 995 例 SHAP 輪廓與真陰性幾乎相同，46 個欄位中找不到區分訊號。
- **CatBoost 類別組合歸因偏差**：chlorpropamide 因為 CTR 組合被分到不屬於它的 SHAP，已驗證並排除。關掉組合後 AP 從 0.251 降到 0.242，顯示類別組合是 CatBoost 勝過 HGB 的來源之一。

## 執行方式

本 repo 會讀取 [Rajpoot 復現 repo](https://github.com/MwYuREZ/Bhuvan_Rajpoot_SSRN)（含 `hgb_catboost/` 的版本，commit `98a0f9e` 之後）的程式、病人切分與 HGB 預測，兩者需放在同一層資料夾：

```
DSP/
├── Bhuvan_Rajpoot_SSRN/          # 團隊復現 repo（唯讀使用）
├── salim-ibrahim-2026-readmission/   # 選用：SHAP 跨研究比較
└── readmission-shap-metrics/     # 本 repo
```

```bash
# 1. 準備 UCI 資料（會驗證 SHA-256）
cd ../Bhuvan_Rajpoot_SSRN && python scripts/download_data.py && cd -

# 2. 建立環境（Python 3.12；版本與團隊 46 欄實驗相同）
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
#   或：python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3. 全部重跑（約 8 分鐘，4 執行緒 CPU）
./run_all.sh
```

路徑不同時可設定環境變數 `RAJPOOT_REPO`、`SALIM_REPO`。

| 步驟 | 內容 | 驗證 |
|---|---|---|
| `00_download_cdc.py` | 從 UCI（id 891）下載 CDC BRFSS 2015（253,680 筆） | SHA-256 |
| `01_fit_models.py` | 依 `selection_lock.json` 重訓 46 欄 CatBoost（診斷家族／原始 ICD-9）與 HGB | 測試 AP 需與團隊參考值相差 < 1e-9；HGB 需與已存預測一致 |
| `02_rerun_rq1_cdc.py` | 重跑第 31–34 頁的加權、複製、SMOTE-NC、CDC 對照組（三折 OOF＋測試） | 原始組需等於復現 HGB；第 18 頁 CDC 範例需完全重現 |
| `03_metrics.py` | 完整指標、病人 bootstrap CI、操作點、子群體、文獻比較 | — |
| `04_shap.py` | 全測試集 SHAP、類別效果、漏抓分析、穩健性檢查、圖表 | SHAP 加總需等於模型輸出（誤差 < 1e-8） |

## 檔案

| 路徑 | 內容 |
|---|---|
| [`REPORT_zh-TW.md`](REPORT_zh-TW.md) | 完整報告：逐頁指標缺口、補齊數字、SHAP 解讀、投影片修改建議 |
| `results/tables/metrics_46_fields.csv` | 46 欄模型的機率指標與 95% CI |
| `results/tables/operating_points_*.csv` | top-10%、開發集 F1／F2 門檻下的混淆矩陣指標 |
| `results/tables/paired_differences.csv` | 簡報中每組比較的配對差值與 CI（AP／AUROC／Brier） |
| `results/tables/metrics_rq1_cdc.csv` | 重跑的不平衡與 CDC 組（含簡報原數字對照） |
| `results/tables/subgroup_metrics.csv` | 性別、年齡、種族、過去住院次數的子群體表現 |
| `results/tables/calibration_deciles.csv`、`precision_at_recall.csv` | 第 36、39 頁的數值版 |
| `results/tables/literature_comparison.csv` | 與五篇文獻比較（含陽性 F1） |
| `results/tables/shap_*.csv` | SHAP 全域重要性、類別／數值效果、漏抓分析、穩健性檢查 |
| `results/figures/` | SHAP 與分數分布圖 |
| `results/predictions/test_*.csv.gz` | 測試集預測（46 欄模型與重跑組） |

模型、SHAP 原始矩陣、開發集 OOF 預測與原始資料不納入 Git，執行 `run_all.sh` 即可重建。

## 注意

第 31–34 頁的原始程式不在共享 repo 中，`02_rerun_rq1_cdc.py` 是依投影片描述重建的版本。原始比例與 balanced weights 的 AP 與簡報完全一致；duplication、SMOTE-NC 與 CDC 組因抽樣實作不同，數值略有差異，但結論方向相同。細節見報告〈注意事項〉。

資料來源：UCI Diabetes 130-US Hospitals（CC BY 4.0, doi:10.24432/C5230J）；CDC Diabetes Health Indicators（UCI id 891, CC BY 4.0）。
