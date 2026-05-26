# Predicting Near-Earth Object Hazard Status with Explainable Risk Scoring

## Abstract

Near-Earth Objects (NEOs) regularly approach Earth, and a small subset may be labeled as potentially hazardous due to their size, velocity, and close-approach characteristics. Manually reviewing every observed object can be costly, so a data-driven screening model can help experts prioritize objects that deserve earlier inspection. This study uses the NASA Nearest Earth Objects dataset to build an explainable binary classification pipeline for predicting whether a NEO is labeled as `hazardous`. The task remains binary classification, but the model also outputs the estimated probability of `hazardous=True`, which is treated as a risk score for prioritization. We compare Majority Baseline, Logistic Regression, class-weighted Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, and LightGBM models, and we apply lightweight hyperparameter tuning to the main tree-based models. The best validation model is `random_forest_balanced`. After sigmoid calibration and validation-based threshold tuning, the final calibrated model achieves F1 = 0.5332, recall = 0.7474, precision = 0.4145, PR-AUC = 0.5595, and ROC-AUC = 0.9269 on the test set using threshold = 0.19. Explainability analysis using permutation importance and SHAP shows that `miss_distance`, `absolute_magnitude`, and diameter-related features are the most influential factors. The final system provides both global and local explanations, supporting expert review rather than replacing domain judgment.

**Keywords:** Near-Earth Objects, binary classification, class imbalance, risk scoring, probability calibration, SHAP, explainable AI

## 1. Introduction

Near-Earth Objects are important in planetary defense and astronomical monitoring. Most NEOs do not pose immediate danger, but some objects require closer review when their size, velocity, or close-approach conditions suggest higher risk. In practice, experts often need to screen many observed objects before performing deeper orbital analysis. A data mining model that ranks objects by risk and explains its predictions can reduce review burden and support early prioritization.

This study focuses on predicting whether a NEO is labeled as `hazardous` from observed attributes. The problem is naturally a supervised binary classification task. However, a class label alone is not sufficient for risk triage, because experts also need to know which objects should be checked first. Therefore, this project uses a two-layer prediction design: first, the model outputs the estimated probability of `hazardous=True`; second, a decision threshold converts this probability into the final class label.

The hazardous probability in this study should not be interpreted as the true physical probability of Earth impact. It is the model-estimated probability that an object is labeled as hazardous under this dataset's label definition and available feature set. For that reason, this study includes probability calibration and model explainability to make the output more transparent and less likely to be overinterpreted.

The contributions of this work are:

1. A reproducible binary classification pipeline for NEO hazardous-label prediction.
2. A comparison of multiple models and class-weight strategies under class imbalance.
3. Validation-based threshold tuning for a risk-screening decision setting.
4. Calibration and explainability analysis using calibration curves, Brier score, permutation importance, and SHAP.
5. Global feature importance and local case explanations for report and presentation use.

## 2. Dataset and Problem Definition

### 2.1 Data Source

This study uses the NASA Nearest Earth Objects dataset from Kaggle. The dataset author notes that the data comes from NASA Open API and JPL CNEOS close-approach data. The license is CC0 Public Domain. The local experiment uses `data/neo.csv`.

The dataset contains 90,836 rows and 10 columns:

`id`, `name`, `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `orbiting_body`, `sentry_object`, `absolute_magnitude`, `hazardous`

The target column is `hazardous`. The remaining columns are used either as model features or as identifiers for case studies.

### 2.2 Data Quality and Target Distribution

No missing values were observed in the local dataset snapshot. The column `orbiting_body` is always `Earth`, and `sentry_object` is always `False`; therefore, both are constant columns and are not useful as model features. The columns `id` and `name` are identifiers, so they are excluded from model input but kept for local case interpretation.

The target distribution is:

| Class | Count | Ratio |
|---|---:|---:|
| `hazardous = False` | 81,996 | 90.27% |
| `hazardous = True` | 8,840 | 9.73% |

This distribution shows clear class imbalance. A model that always predicts `False` can still achieve about 90.27% accuracy, but it has zero recall for hazardous objects. Therefore, this study does not use accuracy as the main model selection metric. Instead, it focuses on recall, precision, F1, PR-AUC, and ROC-AUC.

![Target distribution](../reports/figures/target_distribution.png)

### 2.3 Numeric Feature Observations

The main numeric feature summary is:

| Feature | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `est_diameter_min` | 0.000609 | 0.019256 | 0.048368 | 0.143402 | 37.892650 | 0.127432 |
| `est_diameter_max` | 0.001362 | 0.043057 | 0.108153 | 0.320656 | 84.730541 | 0.284947 |
| `relative_velocity` | 203.35 | 28619.02 | 44190.12 | 62923.60 | 236990.13 | 48066.92 |
| `miss_distance` | 6745.53 | 17210820.24 | 37846579.26 | 56548996.45 | 74798651.45 | 37066546.03 |
| `absolute_magnitude` | 9.23 | 21.34 | 23.70 | 25.70 | 33.20 | 23.53 |

Pearson correlation shows that `absolute_magnitude` has a correlation of -0.3653 with `hazardous`, indicating that lower absolute magnitude is associated with a higher probability of being labeled hazardous. `relative_velocity` has a positive correlation of 0.1912 with the target. `miss_distance` has only weak linear correlation with the target, at 0.0423. However, later model importance results suggest that `miss_distance` can still be important in a nonlinear model.

![Correlation heatmap](../reports/figures/correlation_heatmap.png)

### 2.4 Task Definition

The prediction task is binary classification:

```text
hazardous ∈ {True, False}
```

The model output is:

```text
P(hazardous = True | observed features)
```

The probability is then converted to a class label using a decision threshold:

```text
if P(hazardous=True) >= threshold:
    predict hazardous
else:
    predict non-hazardous
```

This design supports both classification and risk-based ranking.

## 3. Methodology

### 3.1 Data Splitting

The data is split into train, validation, and test sets using stratified splitting, with an approximate ratio of 70%, 15%, and 15%. Stratification is important because the positive class accounts for only 9.73% of the dataset. Without stratification, the validation or test distribution may become unstable.

The final split sizes are:

| Split | Rows |
|---|---:|
| Train | 63,585 |
| Validation | 13,625 |
| Test | 13,626 |

The validation set is used for model selection and threshold tuning. The test set is used only for final evaluation.

### 3.2 Feature Processing and Feature Engineering

The columns `id`, `name`, `orbiting_body`, and `sentry_object` are removed from model input. The first two are identifiers, while the latter two are constant in the current data snapshot.

The base numeric features are:

- `est_diameter_min`
- `est_diameter_max`
- `relative_velocity`
- `miss_distance`
- `absolute_magnitude`

The following engineered features are added:

| Feature | Definition | Purpose |
|---|---|---|
| `est_diameter_mean` | `(est_diameter_min + est_diameter_max) / 2` | Reduce diameter redundancy with a single size estimate |
| `est_diameter_range` | `est_diameter_max - est_diameter_min` | Represent diameter estimate range |
| `log_est_diameter_mean` | `log1p(est_diameter_mean)` | Reduce skewness in size |
| `log_relative_velocity` | `log1p(relative_velocity)` | Reduce skewness in velocity |
| `log_miss_distance` | `log1p(miss_distance)` | Reduce skewness in distance |

The final model input contains 10 features.

### 3.3 Models

This study compares:

1. **Majority Baseline**: always predicts `hazardous=False`.
2. **Logistic Regression**: a linear model with StandardScaler.
3. **Balanced Logistic Regression**: Logistic Regression with `class_weight="balanced"`.
4. **Balanced Random Forest**: Random Forest with `class_weight="balanced_subsample"`.
5. **HistGradientBoostingClassifier**: a gradient boosting model for nonlinear tabular patterns.
6. **XGBoost**: a histogram-based boosted tree model using `scale_pos_weight` for imbalance.
7. **LightGBM**: an efficient leaf-wise boosting model using `scale_pos_weight`.

The study also applies lightweight hyperparameter tuning to Random Forest, HistGradientBoosting, XGBoost, and LightGBM. Tuning uses `RandomizedSearchCV`, stratified 2-fold cross-validation, four candidate settings per model, and `average_precision` (PR-AUC) as the scoring metric. This includes tuning evidence while keeping the experiment reproducible and computationally manageable.

### 3.4 Threshold Tuning and Probability Calibration

Each model first outputs the probability of `hazardous=True`. A threshold is then selected on the validation set. The search range is 0.05 to 0.95 with a step size of 0.01. The primary selection criterion is F1, with recall and precision used as secondary considerations.

The best model is also calibrated using sigmoid calibration. Calibration is not intended to maximize all classification metrics. Its purpose is to make the probability output more suitable as a risk score. The probability quality is evaluated using calibration curve and Brier score.

![Calibration curve](../reports/figures/final_calibration_curve.png)

### 3.5 Explainability Methods

This study uses two explanation methods:

1. **Permutation Importance**: measures how much PR-AUC decreases when a feature is randomly shuffled.
2. **SHAP**: provides both global feature contribution patterns and local explanations for individual predictions.

Permutation importance summarizes global model behavior. SHAP provides per-sample attribution and helps explain why a specific object receives a high or low hazardous probability.

## 4. Experimental Results

### 4.1 Validation Model Comparison

The validation results at threshold = 0.5 are:

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
| Balanced Random Forest | 0.8979 | 0.4808 | 0.6154 | 0.5399 | 0.5880 | 0.9347 |

The Majority Baseline has high accuracy because the dataset is imbalanced, but its recall and F1 are both zero. This confirms that accuracy is misleading for this task. XGBoost and LightGBM obtain very high recall, but their precision is low, producing many false positives. Tuning improves LightGBM's PR-AUC to 0.5687, but it still does not exceed the Balanced Random Forest PR-AUC of 0.5880. Balanced Random Forest performs best in F1, PR-AUC, and ROC-AUC, so it is selected for calibration and final testing.

### 4.1.1 Hyperparameter Tuning Results

The lightweight tuning cross-validation PR-AUC results are:

| Tuned Model | Best CV PR-AUC |
|---|---:|
| LightGBM Tuned | 0.5284 |
| XGBoost Tuned | 0.5276 |
| Random Forest Tuned | 0.5233 |
| HistGradientBoosting Tuned | 0.5208 |

These results show that XGBoost and LightGBM are competitive during tuning, but their validation-set performance does not surpass the untuned balanced Random Forest. Based on validation PR-AUC, F1, and ROC-AUC together, this study keeps `random_forest_balanced` as the final model.

### 4.2 Final Test Results

The final test results compare three settings:

| Setting | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | Brier | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Random Forest | 0.34 | 0.8544 | 0.3838 | 0.8198 | 0.5228 | 0.5634 | 0.0673 | 0.9267 |
| Calibrated Random Forest | 0.19 | 0.8727 | 0.4145 | 0.7474 | 0.5332 | 0.5595 | 0.0600 | 0.9269 |
| Calibrated Random Forest, default threshold | 0.50 | 0.9119 | 0.5946 | 0.2986 | 0.3976 | 0.5595 | 0.0600 | 0.9269 |

The calibrated model with threshold = 0.19 achieves the best F1 score, 0.5332, with recall = 0.7474. The default threshold of 0.5 has higher accuracy and precision, but its recall drops to 0.2986. This supports the main design choice of the project: threshold tuning is necessary when the goal is risk screening under class imbalance.

![Precision-recall curve](../reports/figures/final_precision_recall_curve.png)

![ROC curve](../reports/figures/final_roc_curve.png)

### 4.3 Confusion Matrix Interpretation

For the calibrated Random Forest at threshold = 0.19, the test confusion matrix is:

|  | Predicted False | Predicted True |
|---|---:|---:|
| True False | 10,900 | 1,400 |
| True True | 335 | 991 |

This setting recovers 991 hazardous objects and misses 335 hazardous objects. Compared with threshold = 0.5, the number of true positives increases from 396 to 991, but false positives also increase from 270 to 1,400. This is a typical trade-off in screening systems: reducing missed hazardous objects requires accepting more candidates for expert review.

## 5. Explainability Analysis

### 5.1 Global Feature Importance

The top five permutation importance results are:

| Rank | Feature | Importance Mean |
|---:|---|---:|
| 1 | `miss_distance` | 0.0915 |
| 2 | `log_miss_distance` | 0.0893 |
| 3 | `absolute_magnitude` | 0.0451 |
| 4 | `log_est_diameter_mean` | 0.0443 |
| 5 | `est_diameter_min` | 0.0424 |

The model relies strongly on close-approach distance, magnitude, and diameter-related features. Although `miss_distance` has weak linear correlation with the target, it becomes highly important in the nonlinear Random Forest model. This suggests that simple linear correlation is not sufficient to describe all useful predictive signals in this dataset.

![Permutation importance](../reports/figures/permutation_importance.png)

The SHAP global plots show a similar pattern: distance, magnitude, and size-related features dominate the model's hazardous probability output.

![SHAP global bar](../reports/figures/shap_global_bar.png)

![SHAP summary beeswarm](../reports/figures/shap_summary_beeswarm.png)

### 5.2 Local Case Studies

This study selects three test cases: one true positive, one false negative, and one false positive.

#### Case 1: True Positive

- `id`: 3774091
- `name`: (2017 HP3)
- True label: hazardous
- Predicted probability: 0.8957
- Threshold: 0.19
- Predicted label: hazardous

SHAP shows that `log_miss_distance`, `miss_distance`, `est_diameter_min`, `absolute_magnitude`, and `log_est_diameter_mean` all push the hazardous probability upward. This case shows that the model can correctly identify a high-risk object based on distance, size, and magnitude signals.

#### Case 2: False Negative

- `id`: 3713941
- `name`: (2015 EO61)
- True label: hazardous
- Predicted probability: 0.1897
- Threshold: 0.19
- Predicted label: non-hazardous

This case is very close to the decision threshold. SHAP shows that size-related features push the probability upward, while `log_miss_distance` pushes it downward. This type of boundary case is important in practice because a small threshold change can alter the final decision.

#### Case 3: False Positive

- `id`: 3566975
- `name`: (2011 KO17)
- True label: non-hazardous
- Predicted probability: 0.8954
- Threshold: 0.19
- Predicted label: hazardous

Although the true label is non-hazardous, the model assigns a high risk score. SHAP indicates that distance, magnitude, and size-related features strongly push the prediction upward. From a screening perspective, this false positive increases expert review workload, but it is less severe than missing a truly hazardous object.

## 6. Discussion

### 6.1 Effect of Threshold Choice

The results show that threshold selection has a major impact on model behavior. The default threshold of 0.5 improves precision and accuracy but sharply reduces recall. In a planetary defense screening scenario, false negatives are usually more costly than false positives, so a lower threshold is reasonable. However, the threshold cannot be lowered arbitrarily because too many false positives would overload expert review. This study uses the validation-F1 threshold of 0.19 as a compromise between recall and precision.

### 6.2 Correct Interpretation of Probability Scores

The calibrated Random Forest obtains a Brier score of 0.0600, better than the raw model's 0.0673. This suggests that calibration improves probability quality. However, the probability still represents the model's estimate of the dataset label, not the physical probability of impact. A physically stronger hazard model would require more complete orbital features and domain-specific modeling.

### 6.3 Feature Explanation and Domain Intuition

The importance of `absolute_magnitude` and diameter-related features is consistent with domain intuition, because magnitude and object size are related. Larger or brighter objects are more likely to be labeled potentially hazardous. `relative_velocity` is positively correlated with the target, but permutation importance ranks it below distance and size features. `miss_distance` has high model importance despite weak linear correlation, suggesting nonlinear effects.

## 7. Limitations

This study has several limitations:

1. The `hazardous` label is a dataset label, not the true probability of Earth impact.
2. The dataset has limited features and does not include full orbital parameters such as MOID, orbital inclination, semi-major axis, or eccentricity.
3. XGBoost, LightGBM, and hyperparameter tuning are included, but the tuning uses a lightweight search; broader search spaces and more CV folds may improve performance further.
4. SHAP explains model behavior but does not prove causality.
5. The selected threshold is based on validation F1. If the real review cost function changes, the optimal threshold may also change.

## 8. Conclusion

This study builds a reproducible and explainable binary classification pipeline for NEO hazardous-label prediction. The results show that accuracy is not suitable as the main metric under strong class imbalance. PR-AUC, F1, recall, and confusion matrix analysis provide a better view of whether the model is useful for risk screening. Balanced Random Forest performs best on the validation set. After sigmoid calibration and threshold tuning, the final model achieves F1 = 0.5332, recall = 0.7474, and precision = 0.4145 on the test set. Explainability analysis shows that close-approach distance, magnitude, and diameter-related features are the main drivers of the model's predictions.

Overall, the best positioning of this project is an explainable probability-based binary classification system. It estimates the probability that a NEO is labeled as hazardous and uses that risk score to support expert prioritization. Future work should add richer orbital data and a formal review-cost function to make the model closer to real planetary defense decision support.

## References

1. Sameep Vani. NASA Nearest Earth Objects. Kaggle Dataset. <https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects>
2. NASA Open APIs. <https://api.nasa.gov/>
3. NASA Jet Propulsion Laboratory, Center for Near Earth Object Studies. Close-Approach Data. <https://cneos.jpl.nasa.gov/ca/>
4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
