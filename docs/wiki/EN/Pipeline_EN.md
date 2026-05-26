# Pipeline Architecture

[← Home](Home_EN.md)

## 1. Overview

The project is organized as three sequential CLI commands, each wrapping a distinct pipeline stage.
All commands are defined as entry points in `pyproject.toml` and executed via `uv run`.

```
neo-eda   →  neo-train  →  neo-explain
```

Each stage reads from the previous stage's outputs and writes its own outputs to
`reports/tables/`, `reports/figures/`, or `models/`.

## 2. Stage Map

```
┌─────────────────────────────────────────────────────────────────┐
│  neo-eda  (eda.py)                                              │
│                                                                 │
│  data/neo.csv                                                   │
│       │                                                         │
│       ▼                                                         │
│  load_neo_data()   ← validates schema, dtypes, target values    │
│       │                                                         │
│       ├──► summarize_dataset()   → dataset_summary.json        │
│       ├──► missing_value_table() → missing_values.csv          │
│       ├──► constant_value_table()→ constant_values.csv         │
│       ├──► class_distribution()  → class_distribution.csv      │
│       ├──► numeric_summary()     → numeric_summary.csv         │
│       ├──► correlation_table()   → correlation_matrix.csv      │
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
│  build_feature_frame()   ← engineers 10 features, splits X/y  │
│       │                                                         │
│       ▼                                                         │
│  split_data()   ← stratified 70% / 15% / 15%                  │
│       │                                                         │
│       ├──► build_models()     ← 7 baseline + candidate models  │
│       ├──► tune_models()      ← RandomizedSearchCV × 4 models  │
│       │                                                         │
│       ├── for each model:                                       │
│       │     model.fit(X_train)                                  │
│       │     metric_row(y_val, proba_val) → validation_rows     │
│       │                                                         │
│       ├──► validation_metrics.csv  (all models ranked)         │
│       │                                                         │
│       ├──► best_model = top validation PR-AUC model            │
│       ├──► threshold_table(y_val, best_val_proba)              │
│       ├──► choose_threshold()  → best_threshold                │
│       │                                                         │
│       ├──► CalibratedClassifierCV(best_model, method="sigmoid")│
│       ├──► calibrated threshold selection                       │
│       │                                                         │
│       ├──► metric_row(y_test, ...)  × 3 settings → test_metrics│
│       ├──► save PR curve / ROC curve / calibration curve       │
│       │                                                         │
│       └──► joblib.dump(artifact)  → models/final_model.joblib  │
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
│       │      picks 1 TP + 1 FN + 1 FP from test set            │
│       │      selected_explanation_cases.csv                     │
│       │                                                         │
│       └──► run_shap_explanations()                              │
│              shap_global_bar.png                                │
│              shap_summary_beeswarm.png                          │
│              shap_local_case_contributions.csv                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Module Responsibilities

| Module | Entry Point | Responsibilities |
|---|---|---|
| `config.py` | — | Shared constants: paths, feature lists, random seed |
| `data.py` | — | Load, validate, and summarize raw data |
| `eda.py` | `neo-eda` | Run EDA and export tables + figures |
| `features.py` | — | Build the 10-feature model input matrix |
| `evaluation.py` | — | Compute metrics, threshold tables, threshold selection |
| `train.py` | `neo-train` | Full training pipeline, model artifact, curves |
| `plots.py` | — | Reusable figure helpers used by eda and train |
| `explain.py` | `neo-explain` | Permutation importance + SHAP explanations |

## 4. Artifact Format

`models/final_model.joblib` is a Python dict with the following keys:

| Key | Type | Description |
|---|---|---|
| `best_model_name` | str | Name of the winning model |
| `best_model` | sklearn estimator | Fitted uncalibrated model |
| `calibrated_model` | sklearn estimator | Fitted `CalibratedClassifierCV` |
| `raw_threshold` | float | Best threshold for uncalibrated model |
| `calibrated_threshold` | float | Best threshold for calibrated model |
| `feature_columns` | list[str] | Ordered list of input feature names |
| `X_train` | DataFrame | Training features |
| `X_val` | DataFrame | Validation features |
| `X_test` | DataFrame | Test features |
| `y_train` | Series | Training labels |
| `y_val` | Series | Validation labels |
| `y_test` | Series | Test labels |
| `meta_train` | DataFrame | Training metadata (id, name, …) |
| `meta_val` | DataFrame | Validation metadata |
| `meta_test` | DataFrame | Test metadata |

The artifact is compressed at level 3 (`joblib.dump(..., compress=3)`).
It is ignored by git because it is generated and can be several hundred MB.

## 5. Reproducibility

- All random operations use `RANDOM_STATE = 42` from `config.py`.
- Dependencies are pinned in `uv.lock`.
- Run `uv sync` to install the exact pinned versions, then the three commands in order.

## 6. Output Directory Layout

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

[← Home](Home_EN.md)