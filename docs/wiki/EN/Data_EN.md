# Dataset & EDA

[← Home](Home_EN.md)

## 1. Data Source

| Item | Detail |
|---|---|
| Name | NASA Nearest Earth Objects |
| Provider | Kaggle (dataset by Sameep Vani) |
| Upstream | NASA Open API + JPL CNEOS close-approach data |
| License | CC0 Public Domain |
| Local path | `data/neo.csv` |

The dataset represents close-approach records for NEOs observed by NASA/JPL.
Each row is one close-approach event for one object.

## 2. Schema

| Column | Type | Role | Notes |
|---|---|---|---|
| `id` | integer | identifier | Not used as a model feature; kept for case tracking |
| `name` | string | identifier | Not used as a model feature; kept for case studies |
| `est_diameter_min` | float | base feature | Estimated minimum diameter (km) |
| `est_diameter_max` | float | base feature | Estimated maximum diameter (km) |
| `relative_velocity` | float | base feature | Relative velocity at close approach (km/h) |
| `miss_distance` | float | base feature | Closest approach distance (km) |
| `orbiting_body` | string | dropped | Always `Earth` in this snapshot — zero variance |
| `sentry_object` | boolean | dropped | Always `False` in this snapshot — zero variance |
| `absolute_magnitude` | float | base feature | Absolute magnitude (H); lower = larger / brighter |
| `hazardous` | boolean | target | `True` = potentially hazardous asteroid (PHA) |

**Dataset size:** 90,836 rows × 10 columns. No missing values observed.

## 3. Target Distribution

| Class | Count | Ratio |
|---|---:|---:|
| `hazardous = False` | 81,996 | 90.27% |
| `hazardous = True` | 8,840 | 9.73% |

Imbalance ratio ≈ 1 : 9.3 (positive : negative).

This imbalance has two important consequences:

1. Accuracy is a misleading metric. A model that always predicts `False` achieves 90.27% accuracy but zero recall.
2. Model selection and threshold tuning must use recall, F1, and PR-AUC as primary metrics.

## 4. Numeric Feature Summary

| Feature | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `est_diameter_min` | 0.000609 | 0.019 | 0.048 | 0.143 | 37.89 | 0.127 |
| `est_diameter_max` | 0.001362 | 0.043 | 0.108 | 0.321 | 84.73 | 0.285 |
| `relative_velocity` | 203 | 28,619 | 44,190 | 62,924 | 236,990 | 48,067 |
| `miss_distance` | 6,746 | 17,210,820 | 37,846,579 | 56,548,996 | 74,798,651 | 37,066,546 |
| `absolute_magnitude` | 9.23 | 21.34 | 23.70 | 25.70 | 33.20 | 23.53 |

All numeric features have right-skewed distributions; `relative_velocity` and `miss_distance`
span several orders of magnitude. This motivates the `log1p` transformations added during
feature engineering.

## 5. Pearson Correlation with Target

| Feature Pair | Pearson r | Interpretation |
|---|---:|---|
| `hazardous` vs `absolute_magnitude` | −0.365 | Moderate negative; lower magnitude (larger/brighter) → more likely hazardous |
| `hazardous` vs `relative_velocity` | +0.191 | Weak-to-moderate positive; faster → slightly more likely hazardous |
| `hazardous` vs `miss_distance` | +0.042 | Very weak; distance alone has limited linear predictive power |
| `est_diameter_min` vs `est_diameter_max` | ≈ 1.000 | Near-perfect collinearity; redundant pair |
| `absolute_magnitude` vs `est_diameter_min` | −0.560 | Consistent with the magnitude–size physical relationship |

Key insight: although `miss_distance` has weak linear correlation with the target,
the nonlinear Random Forest model ranks it as the most important feature.
Linear correlation understates its predictive value.

## 6. Data Quality Checks

The pipeline in `data.py` performs the following checks on load:

- All columns in `EXPECTED_COLUMNS` must be present; missing columns raise `ValueError`.
- All base numeric features are coerced with `pd.to_numeric(errors="raise")`.
- The `hazardous` column must contain only `True` / `False` values; any unexpected value raises `ValueError`.

## 7. EDA Outputs

Running `uv run neo-eda` produces:

| File | Description |
|---|---|
| `reports/tables/dataset_summary.json` | Row count, column count, class counts, missing values |
| `reports/tables/numeric_summary.csv` | Descriptive statistics per base numeric feature |
| `reports/tables/class_distribution.csv` | Class counts and ratios |
| `reports/tables/correlation_matrix.csv` | Pearson correlation matrix (numeric + encoded target) |
| `reports/tables/missing_values.csv` | Per-column missing count and ratio |
| `reports/tables/constant_values.csv` | Columns with ≤ 3 unique values |
| `reports/figures/target_distribution.png` | Bar chart of class imbalance |
| `reports/figures/numeric_distributions.png` | Histograms of the five base numeric features |
| `reports/figures/correlation_heatmap.png` | Annotated Pearson heatmap |

[← Home](Home_EN.md)