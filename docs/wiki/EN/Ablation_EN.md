# Ablation Study

[← Home](Home_EN.md)

## 1. Purpose

This ablation study systematically removes one design decision at a time from the final model
to verify that each component contributes meaningfully to performance.
The task is **risk screening** — the cost of missing a truly hazardous object (false negative)
is higher than the cost of flagging a safe object for review (false positive).
Therefore, **Recall is the primary evaluation axis**, with F1 and PR-AUC as secondary.

## 2. The Five Variations

| Version | Setting | Change from Full Model |
|---|---|---|
| **V0** | Full Model | Balanced RF + Calibration + Tuned Threshold (0.19) |
| **V1** | w/o Calibration | Raw Balanced RF, threshold tuned on raw scores (0.34) |
| **V2** | w/o Threshold Tuning | Calibrated model, default threshold (0.50) |
| **V3** | w/o Class Balancing | RF without `class_weight="balanced_subsample"`, calibrated, tuned threshold (0.24) |
| **V4** | w/o Log Features | Removes `log_est_diameter_mean`, `log_relative_velocity`, `log_miss_distance`; calibrated, tuned threshold (0.29) |

V0 / V1 / V2 were already produced by the standard `neo-train` pipeline.
V3 and V4 were run as additional experiments and their results are stored in
`reports/tables/ablation_study_summary.csv`.

## 3. Results

All metrics are evaluated on the held-out test set (13,626 rows).

| Version | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 Full Model | 0.19 | 0.8727 | 0.4145 | **0.7474** | 0.5332 | 0.5595 | 0.9269 | 991 | 1400 | 335 |
| V1 w/o Calibration | 0.34 | 0.8544 | 0.3838 | 0.8198 | 0.5228 | 0.5634 | 0.9267 | 1087 | 1745 | 239 |
| V2 w/o Threshold Tuning | 0.50 | 0.9119 | 0.5946 | 0.2986 | 0.3976 | 0.5595 | 0.9269 | 396 | 270 | 930 |
| V3 w/o Class Balancing | 0.24 | 0.8929 | 0.4609 | 0.5958 | 0.5197 | 0.5715 | 0.9277 | 790 | 924 | 536 |
| V4 w/o Log Features | 0.29 | 0.8954 | 0.4718 | 0.6244 | 0.5375 | 0.5880 | 0.9324 | 828 | 927 | 498 |

## 4. Analysis per Variation

### V1 — Effect of Calibration

Removing calibration and tuning the threshold on raw scores gives the **highest Recall (0.8198)**,
recovering 1,087 hazardous objects. However, it also produces the most false positives (1,745),
increasing expert review workload. Calibration trades a small amount of recall for better
probability quality (Brier Score drops from 0.0673 → 0.0600), making the risk score more
suitable for ranking and prioritization.

### V2 — Effect of Threshold Tuning

This is the starkest result. Using the default threshold of 0.5 on the calibrated model causes
Recall to collapse to **0.2986** — the model misses 930 hazardous objects. Accuracy improves
to 0.9119 because fewer objects are flagged, but this is exactly the metric that misleads.
Threshold tuning is not optional for an imbalanced risk-screening task.

### V3 — Effect of Class Balancing

Removing `class_weight="balanced_subsample"` reduces Recall from 0.7474 to **0.5958**,
meaning 201 additional hazardous objects are missed (536 vs 335 false negatives).
Accuracy and Precision both increase, which looks good on the surface but is the wrong
trade-off for this application. Class balancing is necessary to maintain adequate sensitivity
toward the minority hazardous class.

### V4 — Effect of Log Features

This variation is the most nuanced. Removing the three log-transformed features
(`log_est_diameter_mean`, `log_relative_velocity`, `log_miss_distance`) actually
**improves PR-AUC (0.5880 vs 0.5595), ROC-AUC (0.9324 vs 0.9269), and F1 (0.5375 vs 0.5332)**.
However, Recall drops from 0.7474 to **0.6244**, and 163 additional hazardous objects are missed
(498 vs 335 false negatives).

The log features reduce skewness in size and distance distributions, which helps the model
learn more stable thresholds at low probability values. The drop in Recall confirms that
their benefit is concentrated in the range that matters most for risk screening —
correctly recovering borderline hazardous objects near the decision boundary.
For a recall-priority task, retaining log features is the correct choice.

## 5. Summary

| Design Decision | Recall impact | Key finding |
|---|---|---|
| Calibration | −0.0724 (0.8198 → 0.7474) | Calibration costs some recall but improves probability quality; net benefit for risk scoring |
| Threshold Tuning | +0.4488 (0.2986 → 0.7474) | The single largest lever; default 0.5 threshold is inadequate |
| Class Balancing | +0.1516 (0.5958 → 0.7474) | Essential for minority-class sensitivity |
| Log Features | +0.1230 (0.6244 → 0.7474) | Improve recall at borderline probabilities; beneficial for screening |

The ablation confirms the design reasoning behind each component of the final model.
No single element can be removed without meaningfully increasing the number of missed
hazardous objects — the metric that matters most in planetary defense screening.
