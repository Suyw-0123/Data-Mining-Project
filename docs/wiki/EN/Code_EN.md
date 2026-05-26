# Code Reference

[← Home](Home_EN.md)

This document describes every module in `src/neo_hazard/` and its public API.
All modules follow the same conventions: no side effects on import,
pure functions where possible, and `RANDOM_STATE` imported from `config` for reproducibility.

---

## `config.py` — Shared constants

No functions except `ensure_output_dirs`. All symbols are module-level constants.

### Constants

| Name | Type | Value / Description |
|---|---|---|
| `PROJECT_ROOT` | `Path` | Two levels above `config.py` (the repo root) |
| `DATA_PATH` | `Path` | `PROJECT_ROOT / "data" / "neo.csv"` |
| `REPORTS_DIR` | `Path` | `PROJECT_ROOT / "reports"` |
| `FIGURES_DIR` | `Path` | `REPORTS_DIR / "figures"` |
| `TABLES_DIR` | `Path` | `REPORTS_DIR / "tables"` |
| `MODELS_DIR` | `Path` | `PROJECT_ROOT / "models"` |
| `RANDOM_STATE` | `int` | `42` |
| `TARGET` | `str` | `"hazardous"` |
| `ID_COLUMNS` | `list[str]` | `["id", "name"]` |
| `CONSTANT_COLUMNS` | `list[str]` | `["orbiting_body", "sentry_object"]` |
| `BASE_NUMERIC_FEATURES` | `list[str]` | The five raw numeric feature names |
| `EXPECTED_COLUMNS` | `list[str]` | All columns the CSV must contain |

### `ensure_output_dirs() → None`

Creates `FIGURES_DIR`, `TABLES_DIR`, and `MODELS_DIR` (with parents) if they do not exist.
Called at the start of every entry-point `main()`.

---

## `data.py` — Data loading and validation

### `load_neo_data(path: Path) → pd.DataFrame`

Reads `neo.csv`, validates:
- All `EXPECTED_COLUMNS` are present (raises `ValueError` if any are missing).
- All `BASE_NUMERIC_FEATURES` can be coerced to float (raises on failure).
- `TARGET` contains only `True` / `False` values (raises `ValueError` on unexpected values).

Returns the validated DataFrame with corrected dtypes.

### `summarize_dataset(df) → dict[str, object]`

Returns a dict with keys: `rows`, `columns`, `hazardous_true`, `hazardous_false`,
`hazardous_rate`, `missing_values`.

### `numeric_summary(df) → pd.DataFrame`

`describe()` for `BASE_NUMERIC_FEATURES` with renamed quantile columns (`q1`, `median`, `q3`).

### `correlation_table(df) → pd.DataFrame`

Pearson correlation matrix including the boolean target encoded as `int` (0/1).
Values rounded to 6 decimal places.

### `class_distribution(df) → pd.DataFrame`

Returns a two-row DataFrame with columns `hazardous`, `count`, `ratio`.

### `missing_value_table(df) → pd.DataFrame`

Per-column `missing_count` and `missing_ratio`.

### `constant_value_table(df) → pd.DataFrame`

Returns rows for columns with ≤ 3 unique values, with their unique counts and value lists.
Useful for finding zero-variance or near-constant columns.

### `safe_log1p(series: pd.Series) → pd.Series`

Applies `np.log1p` after asserting no negative values exist.
Raises `ValueError` with the feature name if any negatives are found.

---

## `eda.py` — EDA entry point

### `main() → None`

CLI entry point for `neo-eda`. Calls `load_neo_data`, runs all summary functions,
saves tables to `TABLES_DIR`, and saves figures to `FIGURES_DIR`.
Prints a brief completion message with output paths.

---

## `features.py` — Feature engineering

### `build_feature_frame(df) → tuple[pd.DataFrame, pd.Series, pd.DataFrame]`

Constructs the model input from the raw DataFrame.

Returns `(X, y, metadata)`:

- `X`: DataFrame of 10 features (5 base + 5 engineered).
- `y`: boolean Series for the `TARGET` column.
- `metadata`: DataFrame of `ID_COLUMNS + CONSTANT_COLUMNS`, kept for case explanations.

**Engineered features computed:**

```python
features["est_diameter_mean"]     = (min + max) / 2
features["est_diameter_range"]    = max - min
features["log_est_diameter_mean"] = safe_log1p(est_diameter_mean)
features["log_relative_velocity"] = safe_log1p(relative_velocity)
features["log_miss_distance"]     = safe_log1p(miss_distance)
```

---

## `evaluation.py` — Metrics and threshold logic

### `probabilities_or_scores(model, X) → np.ndarray`

Returns 1-D array of positive-class probabilities.
Falls back to sigmoid-scaled `decision_function` scores for models without `predict_proba`.
Raises `TypeError` for models that expose neither.

### `metric_row(y_true, y_probability, *, model_name, split, threshold=0.5) → dict`

Computes one row of metrics for a given threshold:
`accuracy`, `precision`, `recall`, `f1`, `pr_auc`, `brier_score`,
`roc_auc` (NaN if only one class present), `tn`, `fp`, `fn`, `tp`.

Used to build both validation and test metric tables.

### `threshold_table(y_true, y_probability, thresholds=None) → pd.DataFrame`

Calls `metric_row` for each threshold in a sweep array.
Default range: `np.arange(0.05, 0.951, 0.01)` (91 thresholds).

### `choose_threshold(table: pd.DataFrame) → float`

Selects the threshold that maximizes F1, breaking ties by recall then precision then lower threshold.

---

## `train.py` — Training entry point

### `imbalance_ratio(y_train) → float`

Returns `negative_count / positive_count`. Used to set `scale_pos_weight` for XGBoost/LightGBM.

### `build_models(y_train) → dict[str, estimator]`

Constructs and returns a dict of 7 unfitted models:
`majority_baseline`, `logistic_regression`, `logistic_regression_balanced`,
`random_forest_balanced`, `hist_gradient_boosting`, `xgboost`, `lightgbm`.

### `build_tuning_specs(y_train) → dict[str, (estimator, param_distributions)]`

Returns tuning specifications for 4 models:
`random_forest_tuned`, `hist_gradient_boosting_tuned`, `xgboost_tuned`, `lightgbm_tuned`.

### `tune_models(X_train, y_train, *, n_iter=4) → (dict[str, estimator], pd.DataFrame)`

Runs `RandomizedSearchCV` for each model in `build_tuning_specs`.
Returns fitted best estimators and a DataFrame of CV PR-AUC scores.

### `split_data(X, y, metadata) → 9-tuple`

Stratified 70/15/15 split.
Returns `X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test`.

### `main() → None`

Full training pipeline:

1. Load data → build features → split.
2. Build and tune all models.
3. Fit each model on train, evaluate on validation.
4. Select best model by PR-AUC.
5. Tune threshold on validation.
6. Calibrate with `CalibratedClassifierCV(method="sigmoid", cv=3)`.
7. Tune calibrated threshold.
8. Evaluate raw and calibrated models on test set (3 metric rows).
9. Save PR curve, ROC curve, calibration curve.
10. Dump artifact to `models/final_model.joblib`.
11. Write `training_summary.json`.

---

## `plots.py` — Figure helpers

All functions call `set_plot_style()` first, then save to a given `path` at `dpi=160`,
and close the figure with `plt.close()`.

### `set_plot_style() → None`

Applies `sns.set_theme(style="whitegrid", context="notebook")`.

### `save_target_distribution(df, target, path) → None`

Seaborn `countplot` for the target column.

### `save_numeric_distributions(df, columns, path) → None`

2×3 grid of `histplot` for each feature in `columns`. Unused axes are turned off.

### `save_correlation_heatmap(corr, path) → None`

Annotated seaborn `heatmap` with `cmap="vlag"`, centered at 0.

### `save_precision_recall_curve(y_true, y_probability, path) → None`

Uses `sklearn.metrics.PrecisionRecallDisplay.from_predictions`.

### `save_roc_curve(y_true, y_probability, path) → None`

Uses `sklearn.metrics.RocCurveDisplay.from_predictions`.

### `save_calibration_curve(model, X, y, path) → None`

Uses `sklearn.calibration.CalibrationDisplay.from_estimator` with `n_bins=10, strategy="quantile"`.

---

## `explain.py` — Explainability entry point

### `save_permutation_importance(model, X_test, y_test) → pd.DataFrame`

Runs `sklearn.inspection.permutation_importance` with `scoring="average_precision"`, `n_repeats=5`.
Saves `permutation_importance.csv` and a horizontal bar chart of the top 12 features.
Returns the importance DataFrame sorted by `importance_mean` descending.

### `selected_cases(meta_test, y_test, probability, threshold) → pd.DataFrame`

Selects 1 TP + 1 FN + 1 FP (or high-scoring TN if no FP) for local explanation.
Each group is sorted by descending hazard probability and the top row is taken.
Saves `selected_explanation_cases.csv` and returns the combined DataFrame.

### `run_shap_explanations(model, X_train, X_test, selected) → str`

Requires the `shap` package (optional dependency; returns a message if missing).

Steps:
1. Build `shap.Explainer` with 80-row background from `X_train`.
2. Compute SHAP values on 120-row sample from `X_test`.
3. Save global bar and beeswarm plots.
4. Compute local SHAP for each `selected` case (top 5 features per case).
5. Save `shap_local_case_contributions.csv`.
6. Returns `"SHAP outputs created."` on success.

### `main() → None`

Loads `models/final_model.joblib`, calls the three explanation functions above,
writes `explainability_summary.json`, and prints a completion message.
Raises `FileNotFoundError` if the artifact does not exist.

---

## Conventions

| Convention | Detail |
|---|---|
| Type annotations | `from __future__ import annotations` in every module |
| Random seed | Always `RANDOM_STATE` from `config`; never a magic number |
| Output paths | Always `FIGURES_DIR / "filename.png"` or `TABLES_DIR / "filename.csv"` |
| No global state | Entry points call `ensure_output_dirs()` at the start; modules have no side effects |
| Ruff | Line length 100, rules E/F/W/I enforced; run `uv run ruff check src/` |

[← Home](Home_EN.md)