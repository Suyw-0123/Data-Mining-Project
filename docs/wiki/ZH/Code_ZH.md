# 程式碼參考

[← 首頁](Home_ZH.md)

本文件描述 `src/neo_hazard/` 中每個模組及其公開 API。
所有模組遵循相同慣例：匯入時無副作用、盡可能使用純函式，
並從 `config` 匯入 `RANDOM_STATE` 以確保可重現性。

---

## `config.py` — 共用常數

除 `ensure_output_dirs` 外無其他函式。所有符號都是模組層級常數。

### 常數

| 名稱 | 型別 | 值 / 說明 |
|---|---|---|
| `PROJECT_ROOT` | `Path` | `config.py` 上兩層（repo 根目錄） |
| `DATA_PATH` | `Path` | `PROJECT_ROOT / "data" / "neo.csv"` |
| `REPORTS_DIR` | `Path` | `PROJECT_ROOT / "reports"` |
| `FIGURES_DIR` | `Path` | `REPORTS_DIR / "figures"` |
| `TABLES_DIR` | `Path` | `REPORTS_DIR / "tables"` |
| `MODELS_DIR` | `Path` | `PROJECT_ROOT / "models"` |
| `RANDOM_STATE` | `int` | `42` |
| `TARGET` | `str` | `"hazardous"` |
| `ID_COLUMNS` | `list[str]` | `["id", "name"]` |
| `CONSTANT_COLUMNS` | `list[str]` | `["orbiting_body", "sentry_object"]` |
| `BASE_NUMERIC_FEATURES` | `list[str]` | 五個原始數值特徵名稱 |
| `EXPECTED_COLUMNS` | `list[str]` | CSV 必須包含的所有欄位 |

### `ensure_output_dirs() → None`

若 `FIGURES_DIR`、`TABLES_DIR`、`MODELS_DIR` 不存在則建立（含父目錄）。
在每個入口 `main()` 開頭呼叫。

---

## `data.py` — 資料載入與驗證

### `load_neo_data(path: Path) → pd.DataFrame`

讀取 `neo.csv`，並驗證：
- 所有 `EXPECTED_COLUMNS` 均存在（缺少任何欄位會拋出 `ValueError`）。
- 所有 `BASE_NUMERIC_FEATURES` 可轉換為 float（遇錯即拋出例外）。
- `TARGET` 只包含 `True` / `False` 值（意外值拋出 `ValueError`）。

回傳型別正確的 DataFrame。

### `summarize_dataset(df) → dict[str, object]`

回傳包含 `rows`、`columns`、`hazardous_true`、`hazardous_false`、
`hazardous_rate`、`missing_values` 的 dict。

### `numeric_summary(df) → pd.DataFrame`

對 `BASE_NUMERIC_FEATURES` 執行 `describe()`，並將分位數欄重命名為 `q1`、`median`、`q3`。

### `correlation_table(df) → pd.DataFrame`

Pearson 相關矩陣，包含將布林目標編碼為 int（0/1）後的欄位。數值四捨五入至 6 位小數。

### `class_distribution(df) → pd.DataFrame`

回傳兩列 DataFrame，欄位為 `hazardous`、`count`、`ratio`。

### `missing_value_table(df) → pd.DataFrame`

各欄的 `missing_count` 與 `missing_ratio`。

### `constant_value_table(df) → pd.DataFrame`

回傳唯一值 ≤ 3 的欄位，包含唯一值數量與值列表。用於找出零變異量欄位。

### `safe_log1p(series: pd.Series) → pd.Series`

確認無負值後套用 `np.log1p`。若有負值，以特徵名稱拋出 `ValueError`。

---

## `eda.py` — EDA 入口

### `main() → None`

`neo-eda` 的 CLI 入口。呼叫 `load_neo_data`，執行所有摘要函式，
將表格儲存至 `TABLES_DIR`，將圖片儲存至 `FIGURES_DIR`。
最後印出簡短的完成訊息與輸出路徑。

---

## `features.py` — 特徵工程

### `build_feature_frame(df) → tuple[pd.DataFrame, pd.Series, pd.DataFrame]`

從原始 DataFrame 建構模型輸入。

回傳 `(X, y, metadata)`：

- `X`：10 個特徵的 DataFrame（5 個基礎 + 5 個工程化）。
- `y`：`TARGET` 欄位的布林 Series。
- `metadata`：`ID_COLUMNS + CONSTANT_COLUMNS` 的 DataFrame，保留供案例說明。

**計算的工程化特徵：**

```python
features["est_diameter_mean"]     = (min + max) / 2
features["est_diameter_range"]    = max - min
features["log_est_diameter_mean"] = safe_log1p(est_diameter_mean)
features["log_relative_velocity"] = safe_log1p(relative_velocity)
features["log_miss_distance"]     = safe_log1p(miss_distance)
```

---

## `evaluation.py` — 指標與閾值邏輯

### `probabilities_or_scores(model, X) → np.ndarray`

回傳正類機率的 1-D 陣列。
對沒有 `predict_proba` 的模型，退回使用 sigmoid 縮放的 `decision_function` 分數。
對兩者都不提供的模型拋出 `TypeError`。

### `metric_row(y_true, y_probability, *, model_name, split, threshold=0.5) → dict`

計算給定閾值下的一列指標：
`accuracy`、`precision`、`recall`、`f1`、`pr_auc`、`brier_score`、
`roc_auc`（只有一個類別時為 NaN）、`tn`、`fp`、`fn`、`tp`。

用於建立 validation 與 test 指標表格。

### `threshold_table(y_true, y_probability, thresholds=None) → pd.DataFrame`

對掃描陣列中的每個閾值呼叫 `metric_row`。
預設範圍：`np.arange(0.05, 0.951, 0.01)`（91 個閾值）。

### `choose_threshold(table: pd.DataFrame) → float`

選擇使 F1 最大化的閾值，以 recall → precision → 較低閾值為連續排名標準。

---

## `train.py` — 訓練入口

### `imbalance_ratio(y_train) → float`

回傳 `negative_count / positive_count`。用於設定 XGBoost/LightGBM 的 `scale_pos_weight`。

### `build_models(y_train) → dict[str, estimator]`

建立並回傳 7 個未訓練模型的 dict：
`majority_baseline`、`logistic_regression`、`logistic_regression_balanced`、
`random_forest_balanced`、`hist_gradient_boosting`、`xgboost`、`lightgbm`。

### `build_tuning_specs(y_train) → dict[str, (estimator, param_distributions)]`

回傳 4 個模型的調整規格：
`random_forest_tuned`、`hist_gradient_boosting_tuned`、`xgboost_tuned`、`lightgbm_tuned`。

### `tune_models(X_train, y_train, *, n_iter=4) → (dict[str, estimator], pd.DataFrame)`

對 `build_tuning_specs` 中的每個模型執行 `RandomizedSearchCV`。
回傳已訓練的最佳 estimator 與 CV PR-AUC 分數的 DataFrame。

### `split_data(X, y, metadata) → 9-tuple`

分層 70/15/15 切分。
回傳 `X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test`。

### `main() → None`

完整訓練流程：

1. 載入資料 → 建立特徵 → 切分。
2. 建立並調整所有模型。
3. 每個模型在 train 上 fit，在 validation 上評估。
4. 以 PR-AUC 選出最佳模型。
5. 在 validation 上調整閾值。
6. 用 `CalibratedClassifierCV(method="sigmoid", cv=3)` 校準。
7. 調整校準模型的閾值。
8. 在 test set 上評估原始與校準模型（3 列指標）。
9. 儲存 PR 曲線、ROC 曲線、校準曲線。
10. 將 artifact 寫入 `models/final_model.joblib`。
11. 寫入 `training_summary.json`。

---

## `plots.py` — 圖形輔助函式

所有函式先呼叫 `set_plot_style()`，再以 `dpi=160` 儲存至指定 `path`，
並以 `plt.close()` 關閉圖形。

### `set_plot_style() → None`

套用 `sns.set_theme(style="whitegrid", context="notebook")`。

### `save_target_distribution(df, target, path) → None`

使用 seaborn `countplot` 繪製目標欄位分布。

### `save_numeric_distributions(df, columns, path) → None`

2×3 網格的 `histplot`，每個特徵一個子圖。多餘的軸關閉。

### `save_correlation_heatmap(corr, path) → None`

帶標注的 seaborn `heatmap`，使用 `cmap="vlag"`，以 0 為中心。

### `save_precision_recall_curve(y_true, y_probability, path) → None`

使用 `sklearn.metrics.PrecisionRecallDisplay.from_predictions`。

### `save_roc_curve(y_true, y_probability, path) → None`

使用 `sklearn.metrics.RocCurveDisplay.from_predictions`。

### `save_calibration_curve(model, X, y, path) → None`

使用 `sklearn.calibration.CalibrationDisplay.from_estimator`，
`n_bins=10, strategy="quantile"`。

---

## `explain.py` — 可解釋性入口

### `save_permutation_importance(model, X_test, y_test) → pd.DataFrame`

以 `scoring="average_precision"`、`n_repeats=5` 執行 permutation importance。
儲存 `permutation_importance.csv` 與前 12 個特徵的水平長條圖。
回傳依 `importance_mean` 降序排列的 DataFrame。

### `selected_cases(meta_test, y_test, probability, threshold) → pd.DataFrame`

選取 1 TP + 1 FN + 1 FP（若無 FP 則選高分 TN）供局部解釋。
各群組依危險機率降序排列並取第一列。
儲存 `selected_explanation_cases.csv` 並回傳合併的 DataFrame。

### `run_shap_explanations(model, X_train, X_test, selected) → str`

需要 `shap` 套件（可選依賴；若未安裝則回傳提示訊息）。

步驟：
1. 以 `X_train` 的 80 筆背景樣本建立 `shap.Explainer`。
2. 對 `X_test` 的 120 筆樣本計算 SHAP 值。
3. 儲存全域 bar 圖與 beeswarm 圖。
4. 計算每個 `selected` 案例的局部 SHAP（每案例前 5 個特徵）。
5. 儲存 `shap_local_case_contributions.csv`。
6. 成功時回傳 `"SHAP outputs created."`。

### `main() → None`

載入 `models/final_model.joblib`，呼叫上述三個解釋函式，
寫入 `explainability_summary.json`，並印出完成訊息。
若 artifact 不存在則拋出 `FileNotFoundError`。

---

## 慣例

| 慣例 | 說明 |
|---|---|
| 型別標注 | 每個模組開頭均有 `from __future__ import annotations` |
| 隨機種子 | 一律使用 `config` 的 `RANDOM_STATE`；不使用魔術數字 |
| 輸出路徑 | 一律使用 `FIGURES_DIR / "name.png"` 或 `TABLES_DIR / "name.csv"` |
| 無全域副作用 | 入口 `main()` 開頭呼叫 `ensure_output_dirs()`；模組匯入無副作用 |
| Ruff | 行長限制 100，規則 E/F/W/I；以 `uv run ruff check src/` 執行檢查 |

[← 首頁](Home_ZH.md)