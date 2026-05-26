# Models & Experiments

[← Home](Home_EN.md)

## 1. Design Philosophy

The model comparison follows three principles:

1. **Include a dumb baseline** to reveal the cost of class imbalance on accuracy.
2. **Compare linear and nonlinear models** to show whether the decision boundary is linear.
3. **Apply lightweight tuning** to give each tree model a fair chance without prohibitive cost.

Model selection uses **validation PR-AUC** as the primary criterion, with F1 and ROC-AUC
as secondary tie-breakers. The test set is only used once at the very end for final reporting.

## 2. Class Imbalance Strategy

The positive class (`hazardous = True`) is only 9.73% of the data.
Three strategies are used:

| Strategy | Where Applied |
|---|---|
| `class_weight="balanced"` | Logistic Regression (balanced variant) |
| `class_weight="balanced_subsample"` | Random Forest |
| `scale_pos_weight=neg/pos` | XGBoost, LightGBM |
| Threshold tuning on validation set | All models |

Undersampling and SMOTE were considered but not applied: the class ratio is manageable
with weight-based methods, and synthetic sampling may produce physically unrealistic NEO records.

## 3. Model Catalog

### 3.1 Majority Baseline

```python
DummyClassifier(strategy="most_frequent")
```

Always predicts the majority class (`False`). Used to demonstrate that 90.27% accuracy
means nothing in this task.

### 3.2 Logistic Regression (default)

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000, random_state=42)),
])
```

Linear decision boundary. Provides a calibrated probability baseline.
`max_iter=3000` is set to ensure convergence on this dataset size.

### 3.3 Balanced Logistic Regression

Same as above but with `class_weight="balanced"`.
Increases the weight of the minority class proportionally to its inverse frequency.

### 3.4 Balanced Random Forest

```python
RandomForestClassifier(
    n_estimators=120,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
```

`balanced_subsample` re-weights each bootstrap sample independently,
which is more stable than global `balanced` for ensemble methods.

### 3.5 HistGradientBoostingClassifier

```python
HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=250,
    l2_regularization=0.01,
    random_state=42,
)
```

scikit-learn's histogram-based gradient boosting. Does not natively support `scale_pos_weight`,
so class imbalance is handled primarily through threshold tuning.

### 3.6 XGBoost

```python
XGBClassifier(
    n_estimators=180, max_depth=4, learning_rate=0.06,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
    reg_lambda=1.0, objective="binary:logistic",
    eval_metric="logloss", tree_method="hist",
    scale_pos_weight=neg/pos,
    random_state=42, n_jobs=-1,
)
```

`scale_pos_weight` is set to the negative/positive class ratio computed from the training set.

### 3.7 LightGBM

```python
LGBMClassifier(
    n_estimators=180, max_depth=-1, num_leaves=31,
    learning_rate=0.06, subsample=0.9, colsample_bytree=0.9,
    min_child_samples=30, reg_lambda=1.0,
    scale_pos_weight=neg/pos,
    random_state=42, n_jobs=-1, verbose=-1,
)
```

Leaf-wise growth strategy. `verbose=-1` suppresses LightGBM's training log.

## 4. Hyperparameter Tuning

Tuning runs after the default models are trained.
It uses `RandomizedSearchCV` with stratified 2-fold CV, scoring on `average_precision` (PR-AUC).

**Setting:** `n_iter=4` per model (lightweight, reproducible, computationally manageable).

Four models are tuned:

### Random Forest (tuned)

```python
param_distributions = {
    "n_estimators": randint(80, 220),
    "max_depth": [None, 6, 10, 14],
    "min_samples_leaf": randint(1, 8),
    "max_features": ["sqrt", "log2", None],
}
```

### HistGradientBoosting (tuned)

```python
param_distributions = {
    "learning_rate": uniform(0.03, 0.12),
    "max_iter": randint(120, 360),
    "max_leaf_nodes": randint(15, 64),
    "l2_regularization": uniform(0.0, 0.15),
    "min_samples_leaf": randint(10, 60),
}
```

### XGBoost (tuned)

```python
param_distributions = {
    "n_estimators": randint(120, 320),
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.03, 0.12),
    "subsample": uniform(0.75, 0.25),
    "colsample_bytree": uniform(0.75, 0.25),
    "min_child_weight": randint(1, 8),
    "reg_lambda": uniform(0.5, 2.0),
}
```

### LightGBM (tuned)

```python
param_distributions = {
    "n_estimators": randint(120, 320),
    "num_leaves": randint(15, 80),
    "learning_rate": uniform(0.03, 0.12),
    "subsample": uniform(0.75, 0.25),
    "colsample_bytree": uniform(0.75, 0.25),
    "min_child_samples": randint(10, 80),
    "reg_lambda": uniform(0.5, 2.0),
}
```

**Tuning CV PR-AUC results:**

| Model | Best CV PR-AUC |
|---|---:|
| LightGBM Tuned | 0.5284 |
| XGBoost Tuned | 0.5276 |
| Random Forest Tuned | 0.5233 |
| HistGradientBoosting Tuned | 0.5208 |

## 5. Validation Comparison (threshold = 0.5)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority Baseline | 0.9027 | 0.0000 | 0.0000 | 0.0000 | 0.0973 | 0.5000 |
| Logistic Regression | 0.9078 | 0.6606 | 0.1086 | 0.1865 | 0.4170 | 0.8883 |
| Balanced Logistic Regression | 0.7869 | 0.3073 | 0.9487 | 0.4642 | 0.3951 | 0.8908 |
| HistGradientBoosting | 0.9163 | 0.7218 | 0.2270 | 0.3454 | 0.5591 | 0.9255 |
| HistGradientBoosting Tuned | 0.9155 | 0.7384 | 0.2044 | 0.3201 | 0.5591 | 0.9253 |
| XGBoost | 0.7918 | 0.3169 | 0.9864 | 0.4797 | 0.5411 | 0.9212 |
| XGBoost Tuned | 0.8080 | 0.3312 | 0.9540 | 0.4916 | 0.5536 | 0.9248 |
| LightGBM | 0.8037 | 0.3288 | 0.9766 | 0.4920 | 0.5647 | 0.9281 |
| LightGBM Tuned | 0.8027 | 0.3275 | 0.9751 | 0.4903 | 0.5687 | 0.9292 |
| Random Forest Tuned | 0.8094 | 0.3351 | 0.9744 | 0.4987 | 0.5548 | 0.9259 |
| **Balanced Random Forest** | **0.8979** | **0.4808** | **0.6154** | **0.5399** | **0.5880** | **0.9347** |

**Winner: `random_forest_balanced`** — highest PR-AUC (0.5880), F1 (0.5399), and ROC-AUC (0.9347).

XGBoost and LightGBM achieve very high recall (> 0.97) at the default threshold but at the cost
of very low precision (~ 0.33), producing many false positives. Balanced Random Forest strikes
a better balance across all metrics.

## 6. Model Selection Decision

`random_forest_balanced` is selected as the base model for calibration and final testing.
The decision is based on validation PR-AUC and confirmed by F1 and ROC-AUC rankings.

Tuned variants of LightGBM and XGBoost have higher cross-validation PR-AUC scores during tuning,
but they do not surpass the untuned Balanced Random Forest on the actual validation set.
This suggests the tuning search was not wide enough to close the gap on this dataset.

[← Home](Home_EN.md)