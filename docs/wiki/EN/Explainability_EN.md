# Explainability

[← Home](Home_EN.md)

## 1. Why Explainability Matters Here

The `hazardous` label is a safety-relevant classification. An expert who receives a model
prediction needs to know not just "is it hazardous?" but "why does the model think so?"
without that they cannot audit, trust, or correct the model.

This project uses two complementary explanation methods:

| Method | Scope | What it answers |
|---|---|---|
| Permutation Importance | Global | Which features matter most across all predictions? |
| SHAP | Global + Local | How does each feature push each prediction up or down? |

## 2. Permutation Importance

### How it works

Each feature is independently shuffled (breaking its correlation with the target) and the
drop in PR-AUC is measured over 5 random repeats. A larger drop means the feature is more
important.

```python
permutation_importance(
    model, X_test, y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)
```

### Results (top 10)

| Rank | Feature | Importance Mean | Importance Std |
|---:|---|---:|---:|
| 1 | `miss_distance` | 0.0915 | — |
| 2 | `log_miss_distance` | 0.0893 | — |
| 3 | `absolute_magnitude` | 0.0451 | — |
| 4 | `log_est_diameter_mean` | 0.0443 | — |
| 5 | `est_diameter_min` | 0.0424 | — |
| 6 | `est_diameter_mean` | — | — |
| 7 | `est_diameter_max` | — | — |
| 8 | `est_diameter_range` | — | — |
| 9 | `log_relative_velocity` | — | — |
| 10 | `relative_velocity` | — | — |

### Key insight

`miss_distance` and `log_miss_distance` dominate, despite `miss_distance` having only
r = 0.042 linear correlation with the target. This confirms that the Random Forest exploits
nonlinear interactions with distance — objects that are close but also large/bright are labeled
hazardous more frequently, and the model has learned that joint condition.

### Output files

- `reports/tables/permutation_importance.csv` — feature × importance_mean × importance_std
- `reports/figures/permutation_importance.png` — horizontal bar chart of top 12 features

## 3. SHAP

### Setup

A `shap.Explainer` is built using the calibrated model's prediction function.
Because the model does not have a native SHAP tree explainer (it's a calibrated wrapper),
a background sample of 80 training rows is used as the reference distribution:

```python
background = X_train.sample(80, random_state=42)
explainer = shap.Explainer(predict_positive, background)
shap_values = explainer(explain_sample)  # explain_sample: 120 test rows
```

`predict_positive` returns `model.predict_proba(X)[:, 1]`, the positive-class probability.

### Global SHAP plots

**Bar plot** — mean absolute SHAP value per feature, showing overall global importance.
Output: `reports/figures/shap_global_bar.png`

**Beeswarm plot** — each dot is one sample; the horizontal position shows SHAP value;
the color shows the feature value (red = high, blue = low).
This reveals the direction of each feature's effect.
Output: `reports/figures/shap_summary_beeswarm.png`

### Interpretation of global SHAP

From the beeswarm:

- **`miss_distance` / `log_miss_distance`:** low values (objects that pass close) push
  the hazardous probability **up**; high values push it down.
- **`absolute_magnitude`:** low values (brighter/larger objects) push the probability **up**.
- **`log_est_diameter_mean` / `est_diameter_min`:** high values (larger objects) push
  the probability **up**.
- **`relative_velocity`:** moderate positive effect.

These patterns are physically intuitive: larger, brighter objects that pass close to Earth are
more likely to be labeled potentially hazardous.

## 4. Local Case Studies

Three test cases are selected by `selected_cases()`:

| Case Type | Selection Rule |
|---|---|
| True Positive | Highest hazard probability among correctly predicted hazardous objects |
| False Negative | Highest hazard probability among missed hazardous objects |
| False Positive | Highest hazard probability among incorrectly flagged non-hazardous objects |

If no false positive exists, a high-scoring true negative is used instead.

### Case 1: True Positive

| Attribute | Value |
|---|---|
| `id` | 3774091 |
| `name` | (2017 HP3) |
| True label | hazardous |
| Predicted probability | 0.8957 |
| Threshold | 0.19 |
| Predicted label | hazardous ✓ |

SHAP shows that `log_miss_distance`, `miss_distance`, `est_diameter_min`,
`absolute_magnitude`, and `log_est_diameter_mean` all push the probability **upward**.
All five contributing features agree on the direction: this object passes close to Earth,
has a relatively small miss distance, and is physically large/bright.

### Case 2: False Negative

| Attribute | Value |
|---|---|
| `id` | 3713941 |
| `name` | (2015 EO61) |
| True label | hazardous |
| Predicted probability | 0.1897 |
| Threshold | 0.19 |
| Predicted label | non-hazardous ✗ |

The probability (0.1897) is just below the threshold (0.19). This is a boundary case.
SHAP shows that size-related features push the probability up, while `log_miss_distance`
pushes it down — the model interprets this object as relatively far away.
A small threshold change or a future miss distance update could flip the prediction.

### Case 3: False Positive

| Attribute | Value |
|---|---|
| `id` | 3566975 |
| `name` | (2011 KO17) |
| True label | non-hazardous |
| Predicted probability | 0.8954 |
| Threshold | 0.19 |
| Predicted label | hazardous ✗ |

Despite being non-hazardous, the model assigns a high risk score.
SHAP shows that distance, magnitude, and size features strongly push the prediction up.
From a screening perspective, this false positive increases expert review workload but is
less costly than a false negative (missing a truly dangerous object).

### Local SHAP output

The top 5 SHAP contributions for each selected case are saved to
`reports/tables/shap_local_case_contributions.csv` with columns:

| Column | Description |
|---|---|
| `source_index` | Row index in the test DataFrame |
| `rank` | Feature rank by absolute SHAP value (1 = most influential) |
| `feature` | Feature name |
| `feature_value` | Actual feature value for this object |
| `shap_value` | SHAP contribution (positive = pushes toward hazardous) |
| `direction` | `pushes_probability_up` or `pushes_probability_down` |

## 5. Limitations

1. SHAP explains model behavior, not physical causality. If the model has learned dataset
   artifacts, SHAP will also reflect those artifacts.
2. The background sample size (80 rows) is a trade-off between explanation quality and
   compute time. A larger background would give more stable SHAP values.
3. The hazard probability is the model's estimate of the dataset label, not a true
   physical probability of Earth impact. Users should not interpret SHAP contributions as
   "factors that make this object physically dangerous."

[← Home](Home_EN.md)