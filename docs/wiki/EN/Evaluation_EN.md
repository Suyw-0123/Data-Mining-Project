# Evaluation & Results

[← Home](Home_EN.md)

## 1. Why Accuracy is Misleading

The dataset is 90.27% non-hazardous. A model that always predicts `False` achieves 90.27% accuracy
but misses every hazardous object (recall = 0). Reporting accuracy as the headline metric would
make the worst possible model look acceptable.

This project uses the following as primary metrics:

| Metric | What It Measures |
|---|---|
| **Recall** | Fraction of true hazardous objects that the model correctly flags |
| **Precision** | Fraction of model-flagged objects that are actually hazardous |
| **F1** | Harmonic mean of recall and precision |
| **PR-AUC** | Area under the precision-recall curve; threshold-independent; ideal for imbalanced data |
| **ROC-AUC** | Ranking quality; less sensitive to class imbalance than PR-AUC |
| **Brier Score** | Mean squared error of predicted probabilities; lower is better calibrated |
| **Confusion Matrix** | Absolute TP/FP/FN/TN counts for chosen threshold |

Accuracy is still reported but is never used to select or compare models.

## 2. Data Splits

| Split | Rows | Purpose |
|---|---:|---|
| Train | 63,585 | Model fitting |
| Validation | 13,625 | Model selection, threshold tuning, calibration fitting |
| Test | 13,626 | Final one-time evaluation |

Splits are stratified on `hazardous` to preserve the 9.73% positive rate in every subset.
The test set is held out until the final evaluation; it plays no role in model selection.

## 3. Threshold Tuning

### Why the default 0.5 threshold is wrong here

At threshold = 0.5, the Balanced Random Forest has recall = 0.6154.
That means 38.5% of hazardous objects are missed. For a planetary defense screening task,
this is too high a miss rate.

### How threshold tuning works

After the best model is selected on validation PR-AUC, a fine threshold sweep is performed:

```python
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)  # 91 candidates
```

For each threshold, `metric_row()` computes F1, recall, precision, PR-AUC, ROC-AUC, Brier score,
and the confusion matrix on the validation set.

### Selection criterion

```python
candidates.sort_values(
    ["f1", "recall", "precision", "threshold"],
    ascending=[False, False, False, True],
)
```

Primary: F1. Tie-breaker 1: recall. Tie-breaker 2: precision. Tie-breaker 3: lower threshold.
This is conservative — when F1 is tied, the more recall-oriented threshold wins.

### Result

The selected threshold for the calibrated Balanced Random Forest is **0.19**.

## 4. Probability Calibration

### Purpose

An uncalibrated model may output probabilities that are overconfident or underconfident.
The calibrated probability is more suitable as a risk score for expert triage.

### Method

```python
CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
```

Sigmoid (Platt) scaling fits a logistic transformation on top of the raw model scores using
3-fold CV on the training data.

### Effect on metrics

| Setting | Brier Score | PR-AUC |
|---|---:|---:|
| Raw Random Forest | 0.0673 | 0.5634 |
| Calibrated Random Forest | 0.0600 | 0.5595 |

Calibration lowers the Brier score (better probability quality) at the small cost of a
slightly lower PR-AUC. This trade-off is acceptable because the primary goal is risk ranking
quality, not just AUC.

## 5. Final Test Results

Three configurations of the winning model are evaluated on the test set:

| Configuration | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | Brier | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Random Forest | 0.34 | 0.8544 | 0.3838 | 0.8198 | 0.5228 | 0.5634 | 0.0673 | 0.9267 |
| **Calibrated Random Forest** | **0.19** | **0.8727** | **0.4145** | **0.7474** | **0.5332** | **0.5595** | **0.0600** | **0.9269** |
| Calibrated RF, default threshold | 0.50 | 0.9119 | 0.5946 | 0.2986 | 0.3976 | 0.5595 | 0.0600 | 0.9269 |

The calibrated model at threshold = 0.19 is the recommended deployment configuration.

## 6. Confusion Matrix (Calibrated RF, threshold = 0.19)

|  | Predicted False | Predicted True |
|---|---:|---:|
| **True False** | 10,900 | 1,400 |
| **True True** | 335 | 991 |

- **True Positives (991):** hazardous objects correctly flagged for review.
- **False Negatives (335):** hazardous objects missed — the most costly error in this domain.
- **False Positives (1,400):** non-hazardous objects flagged for review — increases workload.
- **True Negatives (10,900):** correctly cleared as non-hazardous.

Compared to the default threshold of 0.5, this configuration recovers 991 − 396 = **595 more
hazardous objects** at the cost of 1,400 − 270 = **1,130 additional false positives**.

## 7. Threshold Effect on Precision-Recall Trade-off

The table below shows how threshold choice affects screening behavior:

| Threshold | Recall | Precision | F1 | Expert review workload |
|---:|---:|---:|---:|---|
| 0.10 | ~0.89 | ~0.28 | ~0.43 | Very high — most objects flagged |
| 0.19 | 0.7474 | 0.4145 | 0.5332 | Moderate — chosen configuration |
| 0.34 | 0.8198 | 0.3838 | 0.5228 | Moderate (raw model) |
| 0.50 | 0.2986 | 0.5946 | 0.3976 | Low — many hazardous objects missed |

## 8. Output Files

| File | Description |
|---|---|
| `model_metrics_validation.csv` | All models at threshold = 0.5 on validation set |
| `hyperparameter_tuning_results.csv` | CV PR-AUC and best params for tuned models |
| `threshold_tuning_validation.csv` | Threshold sweep for uncalibrated model |
| `threshold_tuning_validation_calibrated.csv` | Threshold sweep for calibrated model |
| `final_test_metrics.csv` | Three configurations on the test set |
| `final_test_predictions.csv` | Per-row: probability, threshold, predicted label |
| `training_summary.json` | Summary of key decisions and file paths |
| `final_precision_recall_curve.png` | PR curve on test set |
| `final_roc_curve.png` | ROC curve on test set |
| `final_calibration_curve.png` | Reliability diagram (predicted vs actual probability) |

[← Home](Home_EN.md)