# Feature Engineering

[← Home](Home_EN.md)

## 1. Dropped Columns

The following columns are excluded from model input before any feature is built:

| Column | Reason |
|---|---|
| `id` | Identifier; no predictive signal |
| `name` | Identifier; no predictive signal |
| `orbiting_body` | Constant (`Earth`) in this dataset snapshot |
| `sentry_object` | Constant (`False`) in this dataset snapshot |

`id` and `name` are retained in a separate `metadata` frame so they can be used for case-level
explanations in the explainability stage.

## 2. Base Numeric Features

These five columns are used directly as model input after dtype coercion:

| Feature | Unit | Description |
|---|---|---|
| `est_diameter_min` | km | Estimated minimum diameter of the object |
| `est_diameter_max` | km | Estimated maximum diameter of the object |
| `relative_velocity` | km/h | Velocity relative to Earth at closest approach |
| `miss_distance` | km | Distance at closest approach |
| `absolute_magnitude` | — | Absolute magnitude H; lower = brighter / tends to be larger |

## 3. Engineered Features

Five additional features are computed in `features.py:build_feature_frame()`:

| Feature | Definition | Purpose |
|---|---|---|
| `est_diameter_mean` | `(est_diameter_min + est_diameter_max) / 2` | Reduce collinearity; single representative size estimate |
| `est_diameter_range` | `est_diameter_max − est_diameter_min` | Capture uncertainty in the diameter estimate |
| `log_est_diameter_mean` | `log1p(est_diameter_mean)` | Reduce right-skew in size values |
| `log_relative_velocity` | `log1p(relative_velocity)` | Reduce right-skew in velocity values |
| `log_miss_distance` | `log1p(miss_distance)` | Reduce right-skew in distance values |

`log1p` is used instead of `log` to safely handle any near-zero values.
The `safe_log1p()` helper in `data.py` first verifies that no negative values exist before applying
the transform; it raises `ValueError` if any are found.

## 4. Final Feature Set

The model input matrix has **10 features**:

```
est_diameter_min
est_diameter_max
relative_velocity
miss_distance
absolute_magnitude
est_diameter_mean       ← engineered
est_diameter_range      ← engineered
log_est_diameter_mean   ← engineered
log_relative_velocity   ← engineered
log_miss_distance       ← engineered
```

Note that `est_diameter_min` and `est_diameter_max` are kept alongside `est_diameter_mean`
and `est_diameter_range`. Tree-based models handle the collinearity between them without
any harm, and retaining the originals allows SHAP to assign attribution to individual
diameter bounds.

## 5. Scaling

Tree-based models (Random Forest, HistGradientBoosting, XGBoost, LightGBM) do not require
feature scaling.

Logistic Regression models use a `StandardScaler` applied inside a `sklearn.pipeline.Pipeline`,
so the scaler is always fitted on training data only and correctly applied to validation / test.

## 6. Feature Importance Preview

After training, the top features by permutation importance (PR-AUC drop) are:

| Rank | Feature | Importance Mean |
|---:|---|---:|
| 1 | `miss_distance` | 0.0915 |
| 2 | `log_miss_distance` | 0.0893 |
| 3 | `absolute_magnitude` | 0.0451 |
| 4 | `log_est_diameter_mean` | 0.0443 |
| 5 | `est_diameter_min` | 0.0424 |

Despite `miss_distance` having very weak linear correlation with the target (r = 0.042),
it is the most important feature in the nonlinear Random Forest.
This confirms that the `log1p`-transformed version adds complementary signal and
both are worth retaining.

[← Home](Home_EN.md)