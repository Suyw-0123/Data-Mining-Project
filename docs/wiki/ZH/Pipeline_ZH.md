# 流程架構

[← 首頁](Home_ZH.md)

## 1. 概覽

本專案組織為三個依序執行的 CLI 指令，每個指令封裝一個獨立的流程階段。
所有指令都定義在 `pyproject.toml` 的 entry points 中，透過 `uv run` 執行。

```
neo-eda   →  neo-train  →  neo-explain
```

每個階段從前一階段的輸出讀取資料，並將自己的輸出寫入
`reports/tables/`、`reports/figures/` 或 `models/`。

## 2. 階段流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│  neo-eda  (eda.py)                                              │
│                                                                 │
│  data/neo.csv                                                   │
│       │                                                         │
│       ▼                                                         │
│  load_neo_data()   ← 驗證 schema、型別、目標欄位值              │
│       │                                                         │
│       ├──► summarize_dataset()    → dataset_summary.json       │
│       ├──► missing_value_table()  → missing_values.csv         │
│       ├──► constant_value_table() → constant_values.csv        │
│       ├──► class_distribution()   → class_distribution.csv     │
│       ├──► numeric_summary()      → numeric_summary.csv        │
│       ├──► correlation_table()    → correlation_matrix.csv     │
│       ├──► save_target_distribution()  → *.png                 │
│       ├──► save_numeric_distributions()→ *.png                 │
│       └──► save_correlation_heatmap()  → *.png                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  neo-train  (train.py)                                          │
│                                                                 │
│  data/neo.csv                                                   │
│       │                                                         │
│       ▼                                                         │
│  load_neo_data()                                                │
│       │                                                         │
│       ▼                                                         │
│  build_feature_frame()   ← 工程化 10 個特徵，分離 X / y        │
│       │                                                         │
│       ▼                                                         │
│  split_data()   ← 分層切分 70% / 15% / 15%                    │
│       │                                                         │
│       ├──► build_models()   ← 7 個 baseline + 候選模型         │
│       ├──► tune_models()    ← RandomizedSearchCV × 4 個模型   │
│       │                                                         │
│       ├── 每個模型：                                            │
│       │     model.fit(X_train)                                  │
│       │     metric_row(y_val, proba_val) → validation_rows     │
│       │                                                         │
│       ├──► validation_metrics.csv（所有模型排名）               │
│       │                                                         │
│       ├──► best_model = validation PR-AUC 最高的模型            │
│       ├──► threshold_table(y_val, best_val_proba)              │
│       ├──► choose_threshold()  → best_threshold                │
│       │                                                         │
│       ├──► CalibratedClassifierCV(best_model, method="sigmoid")│
│       ├──► 校準模型的閾值選擇                                   │
│       │                                                         │
│       ├──► metric_row(y_test, ...) × 3 種設定 → test_metrics  │
│       ├──► 儲存 PR curve / ROC curve / calibration curve       │
│       │                                                         │
│       └──► joblib.dump(artifact) → models/final_model.joblib   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  neo-explain  (explain.py)                                      │
│                                                                 │
│  models/final_model.joblib                                      │
│       │                                                         │
│       ▼                                                         │
│  joblib.load(artifact)                                          │
│       │                                                         │
│       ├──► save_permutation_importance()                        │
│       │      permutation_importance.csv + *.png                │
│       │                                                         │
│       ├──► selected_cases()                                     │
│       │      從 test set 挑選 1 TP + 1 FN + 1 FP               │
│       │      selected_explanation_cases.csv                     │
│       │                                                         │
│       └──► run_shap_explanations()                              │
│              shap_global_bar.png                                │
│              shap_summary_beeswarm.png                          │
│              shap_local_case_contributions.csv                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 模組職責

| 模組 | 入口指令 | 職責 |
|---|---|---|
| `config.py` | — | 共用常數：路徑、特徵列表、隨機種子 |
| `data.py` | — | 載入、驗證、摘要原始資料 |
| `eda.py` | `neo-eda` | 執行 EDA，匯出表格與圖片 |
| `features.py` | — | 建立 10 個特徵的模型輸入矩陣 |
| `evaluation.py` | — | 計算指標、閾值表、閾值選擇 |
| `train.py` | `neo-train` | 完整訓練流程、模型 artifact、曲線圖 |
| `plots.py` | — | EDA 與 train 共用的圖形輔助函式 |
| `explain.py` | `neo-explain` | Permutation importance + SHAP 解釋 |

## 4. Artifact 格式

`models/final_model.joblib` 是一個 Python dict，包含以下 key：

| Key | 型別 | 說明 |
|---|---|---|
| `best_model_name` | str | 最佳模型的名稱 |
| `best_model` | sklearn estimator | 已訓練的未校準模型 |
| `calibrated_model` | sklearn estimator | 已訓練的 `CalibratedClassifierCV` |
| `raw_threshold` | float | 未校準模型的最佳閾值 |
| `calibrated_threshold` | float | 校準模型的最佳閾值 |
| `feature_columns` | list[str] | 輸入特徵名稱（有序） |
| `X_train` | DataFrame | 訓練特徵 |
| `X_val` | DataFrame | 驗證特徵 |
| `X_test` | DataFrame | 測試特徵 |
| `y_train` | Series | 訓練標籤 |
| `y_val` | Series | 驗證標籤 |
| `y_test` | Series | 測試標籤 |
| `meta_train` | DataFrame | 訓練集 metadata（id, name 等） |
| `meta_val` | DataFrame | 驗證集 metadata |
| `meta_test` | DataFrame | 測試集 metadata |

Artifact 以 level 3 壓縮（`joblib.dump(..., compress=3)`），
因為體積較大且可重新生成，已加入 `.gitignore`。

## 5. 可重現性

- 所有隨機操作使用 `config.py` 中的 `RANDOM_STATE = 42`。
- 依賴套件版本釘定於 `uv.lock`。
- 執行 `uv sync` 安裝釘定版本，接著依序執行三個指令即可完整重現。

## 6. 輸出目錄結構

```
reports/
├── figures/
│   ├── target_distribution.png
│   ├── numeric_distributions.png
│   ├── correlation_heatmap.png
│   ├── final_precision_recall_curve.png
│   ├── final_roc_curve.png
│   ├── final_calibration_curve.png
│   ├── permutation_importance.png
│   ├── shap_global_bar.png
│   └── shap_summary_beeswarm.png
└── tables/
    ├── dataset_summary.json
    ├── numeric_summary.csv
    ├── class_distribution.csv
    ├── correlation_matrix.csv
    ├── missing_values.csv
    ├── constant_values.csv
    ├── model_metrics_validation.csv
    ├── hyperparameter_tuning_results.csv
    ├── threshold_tuning_validation.csv
    ├── threshold_tuning_validation_calibrated.csv
    ├── final_test_metrics.csv
    ├── final_test_predictions.csv
    ├── training_summary.json
    ├── permutation_importance.csv
    ├── selected_explanation_cases.csv
    ├── shap_local_case_contributions.csv
    └── explainability_summary.json
```

[← 首頁](Home_ZH.md)