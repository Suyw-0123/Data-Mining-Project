# Data Mining Term Project Implementation Plan

## 1. Project Title

**Predicting Near-Earth Object Hazard Status with Explainable Risk Scoring**

This project uses the NASA Near-Earth Objects (NEO) dataset to build a data mining pipeline for predicting whether a NEO is labeled as `hazardous`. The core task remains a **supervised binary classification** problem. In addition to the final class label, the model will output the estimated probability of `hazardous=True`, which will be used as a risk score for prioritizing expert review.

---

## 2. Background and Scenario

Near-Earth Objects regularly pass close to Earth. Most are harmless, but some objects may be labeled as potentially hazardous due to their size, velocity, and close-approach conditions. Reviewing all observed objects manually can be costly, so this project frames the task as a planetary defense screening problem: build a data-driven model that helps experts prioritize NEOs that deserve earlier inspection.

The system should not only output "hazardous" or "non-hazardous". It should also explain why a prediction was made. For example, it should identify which features increased or decreased the estimated hazardous probability for a specific object. This makes the model more auditable, interpretable, and useful in a safety-related setting.

---

## 3. Problem Definition

### 3.1 Task Type

- Learning type: supervised learning
- Task type: binary classification
- Target column: `hazardous`
- Positive class: `hazardous = True`
- Negative class: `hazardous = False`

### 3.2 Main Prediction Objective

Given observed NEO attributes, predict whether the object is labeled as hazardous:

```text
hazardous ∈ {True, False}
```

### 3.3 Probability Output

This project does not convert the task into regression. Instead, it uses **probability-based binary classification**:

```text
P(hazardous = True | observed features)
```

This probability means "the model-estimated probability that the object is labeled as hazardous under this dataset's labeling definition and feature set." It can be used as a risk score and ranking signal, but it must not be interpreted as the true physical probability of Earth impact.

### 3.4 Decision Threshold

The model will first output a hazardous probability, then convert it into a final class using a decision threshold:

```text
if P(hazardous=True) >= threshold:
    predict hazardous
else:
    predict non-hazardous
```

Because the dataset is imbalanced and false negatives are more costly in the project scenario, the implementation should not rely only on the default `0.5` threshold. Instead, threshold tuning will be performed on the validation set using F1, recall, precision, PR-AUC, and the practical review workload.

---

## 4. Dataset Overview

### 4.1 Data Source

- Dataset: NASA Nearest Earth Objects
- Kaggle: <https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects>
- Source noted by dataset author: NASA Open API and JPL CNEOS close-approach data
- License: CC0 Public Domain

### 4.2 Local Dataset Snapshot

The local dataset is stored at:

```text
data/neo.csv
```

Dataset specification:

- Rows: 90,836
- Columns: 10
- Target column: `hazardous`
- Missing values: no missing values were observed in the current snapshot

### 4.3 Column List

| Column | Description | Planned Use |
|---|---|---|
| `id` | Object identifier | Not used as a model feature; kept for case tracking |
| `name` | Object name | Not used as a model feature; kept for case studies |
| `est_diameter_min` | Estimated minimum diameter | Candidate numeric feature |
| `est_diameter_max` | Estimated maximum diameter | Candidate numeric feature |
| `relative_velocity` | Relative velocity | Candidate numeric feature |
| `miss_distance` | Close-approach miss distance | Candidate numeric feature |
| `orbiting_body` | Orbiting body | Constant column in this snapshot; planned to drop |
| `sentry_object` | Whether the object is a sentry object | Constant column in this snapshot; planned to drop |
| `absolute_magnitude` | Absolute magnitude | Candidate numeric feature |
| `hazardous` | Hazard label | Target column |

### 4.4 Target Distribution

| Class | Count | Ratio |
|---|---:|---:|
| `hazardous = False` | 81,996 | 90.27% |
| `hazardous = True` | 8,840 | 9.73% |

The dataset is clearly imbalanced. Accuracy alone is not suitable because a model that mostly predicts the majority class can look accurate while missing many hazardous objects.

---

## 5. Project Goals

The project has four levels of goals:

1. Build a reliable binary classifier for the `hazardous` label.
2. Output a calibrated or at least calibration-checked hazardous probability as a risk score.
3. Use evaluation metrics and decision thresholds that are appropriate for imbalanced data.
4. Build an explainability workflow that provides both global and local explanations for expert review.

The final project should answer:

- Which features most strongly affect hazardous prediction?
- Which model provides the best trade-off among recall, precision, F1, and PR-AUC?
- If NEOs are ranked by risk score, which objects should be reviewed first?
- For an individual NEO, why does the model assign a high or low hazardous probability?

---

## 6. Methodology

### 6.1 Overall Pipeline

```text
Load data
→ Data quality checks
→ Exploratory data analysis
→ Feature preprocessing and feature engineering
→ Train / validation / test stratified split
→ Build baseline models
→ Handle class imbalance
→ Compare models and tune thresholds
→ Check probability calibration
→ Select final model
→ Generate SHAP / feature importance explanations
→ Export figures, tables, and case studies
```

### 6.2 Data Splitting Strategy

To avoid distribution shift caused by class imbalance, the implementation will use stratified splitting.

Recommended split:

- Train: 70%
- Validation: 15%
- Test: 15%

Purpose:

- Train: model fitting
- Validation: model selection, threshold tuning, calibration selection
- Test: final one-time evaluation, not used for tuning

If time allows, stratified k-fold cross-validation can be applied on the training set to compare model stability.

### 6.3 Feature Processing

Planned processing steps:

1. Drop identifier columns:
   - `id`
   - `name`

2. Drop low-information constant columns:
   - `orbiting_body`
   - `sentry_object`

3. Handle highly collinear features:
   - `est_diameter_min` and `est_diameter_max` are almost perfectly collinear.
   - The implementation will compare two designs:
     - Keep both and let tree-based models handle them.
     - Create `est_diameter_mean` or `est_diameter_range` to reduce redundancy.

4. Test transformations for skewed numeric features:
   - `est_diameter_min`
   - `est_diameter_max`
   - `relative_velocity`
   - `miss_distance`
   - `log1p` transformations will be considered.

5. Apply scaling for linear models:
   - Logistic Regression should use StandardScaler or RobustScaler.
   - Random Forest and Gradient Boosting models usually do not require scaling.

### 6.4 Candidate Feature Engineering

Planned engineered features:

| Engineered Feature | Definition | Motivation |
|---|---|---|
| `est_diameter_mean` | `(est_diameter_min + est_diameter_max) / 2` | Reduce collinearity using a single size estimate |
| `est_diameter_range` | `est_diameter_max - est_diameter_min` | Represent uncertainty in size estimation |
| `log_relative_velocity` | `log1p(relative_velocity)` | Reduce skewness in velocity |
| `log_miss_distance` | `log1p(miss_distance)` | Reduce skewness in distance |
| `log_est_diameter_mean` | `log1p(est_diameter_mean)` | Reduce skewness in size |

Feature engineering should remain controlled. Each new feature should be justified by validation performance and interpretability.

---

## 7. Model Design

### 7.1 Baseline Models

Baselines are used to understand task difficulty and verify whether later models provide meaningful improvement.

1. Majority Class Baseline
   - Always predicts `hazardous = False`
   - Demonstrates the limitation of accuracy

2. Logistic Regression
   - Highly interpretable
   - Provides coefficient directions
   - Serves as a linear baseline

3. Logistic Regression with Class Weight
   - Uses `class_weight="balanced"`
   - Provides a simple imbalance-aware baseline

### 7.2 Main Candidate Models

1. Random Forest
   - Captures nonlinear relationships
   - Less sensitive to feature scaling
   - Provides feature importance

2. Gradient Boosting / HistGradientBoosting
   - Often performs well on tabular data
   - Can be paired with SHAP explanations

3. XGBoost or LightGBM
   - Included in the main model comparison
   - Uses `scale_pos_weight` to address class imbalance
   - Serves as a strong tabular-data baseline against Random Forest and HistGradientBoosting

### 7.3 Hyperparameter Tuning

The implementation performs lightweight `RandomizedSearchCV` tuning for the main tree-based models. This avoids the high computational cost of exhaustive grid search.

Tuning setup:

- Cross-validation: stratified 2-fold
- Search method: RandomizedSearchCV
- Scoring: `average_precision` (PR-AUC)
- Candidate settings per model: 4
- Tuned models:
  - Random Forest
  - HistGradientBoosting
  - XGBoost
  - LightGBM

Tuning results are exported to:

```text
reports/tables/hyperparameter_tuning_results.csv
```

### 7.4 Imbalance Handling

Planned comparison:

| Method | Strength | Risk |
|---|---|---|
| `class_weight="balanced"` | Simple, stable, does not change data distribution | Not equally effective for all models |
| Random Undersampling | Reduces majority-class dominance | May discard useful non-hazardous examples |
| SMOTE | Increases minority examples | May create synthetic samples that are not physically meaningful |
| Threshold tuning | Does not modify training data; adjusts decision rule directly | Must use validation data carefully to avoid test leakage |

Recommended priority:

1. Start with class weight and threshold tuning.
2. Add undersampling if time allows.
3. Treat SMOTE as an extension experiment and discuss its limitation for astronomy-related data.

---

## 8. Evaluation Design

### 8.1 Main Metrics

Because hazardous objects are the minority class, evaluation should focus on positive-class performance.

| Metric | Purpose |
|---|---|
| Recall | Measures how many hazardous objects are recovered |
| Precision | Measures how many predicted hazardous objects are truly hazardous |
| F1-score | Balances precision and recall |
| PR-AUC | Suitable for imbalanced binary classification |
| ROC-AUC | Supplementary ranking metric |
| Confusion Matrix | Shows actual FP and FN counts |
| Brier Score | Evaluates probability calibration quality |

Accuracy will be reported but will not be used as the main model selection metric.

### 8.2 Threshold Tuning

After the model outputs probabilities, different thresholds will be tested on the validation set:

- `0.1`
- `0.2`
- `0.3`
- `0.4`
- `0.5`
- or a finer grid search

Threshold selection should consider:

- If the goal is to find more hazardous objects, choose a recall-oriented threshold.
- If the goal is to reduce expert review workload, choose a higher-precision threshold.
- If a balanced decision is needed, use F1 or F-beta score.

Recommended report items:

```text
Best threshold by validation F1
Recall-oriented threshold
Default 0.5 threshold
```

This will clearly show how threshold choice affects practical outcomes.

### 8.3 Probability Calibration

If model probabilities are used as risk scores, calibration must be checked.

Planned methods:

- Calibration curve
- Brier score
- CalibratedClassifierCV
  - sigmoid / Platt scaling
  - isotonic regression

If time is limited, the project should at least output a calibration curve and Brier score, then discuss whether the model probability is suitable as a risk score.

---

## 9. Explainability Design

### 9.1 Global Explanation

Global explanation answers: "Which features influence the model most overall?"

Planned outputs:

- Logistic Regression coefficient plot
- Tree-based feature importance
- SHAP summary plot
- SHAP mean absolute value ranking

Key checks:

- Whether `absolute_magnitude` is a high-impact feature.
- Whether `relative_velocity` increases hazardous probability.
- Whether `miss_distance` has weaker influence.
- Whether size-related features and magnitude behave consistently with domain intuition.

### 9.2 Local Explanation

Local explanation answers: "Why did the model assign this NEO a high or low hazardous probability?"

Planned case studies:

1. True Positive: correctly predicted hazardous object.
2. False Negative: missed hazardous object, used for risk analysis.
3. False Positive or high-risk non-hazardous object: used to analyze over-warning behavior.

Each case should include:

- `id` / `name`
- True label
- Predicted probability
- Decision threshold
- Predicted label
- Top 3 to 5 features that increased or decreased the probability
- Human-readable explanation

### 9.3 Threshold Explanation

The report should explain not only the predicted probability, but also why the object is classified as hazardous under a specific threshold.

Example format:

```text
This object has a hazardous probability of 0.42.
Under the recall-oriented threshold of 0.30, it is prioritized for expert review.
Under the default threshold of 0.50, it would not be classified as hazardous.
```

This separates the model score from the final decision rule.

---

## 10. Experiment Plan

### 10.1 Experiment A: Baseline and Imbalance Effect

Purpose:

- Show why accuracy is insufficient.
- Build Majority Class and Logistic Regression baselines.

Outputs:

- Accuracy, precision, recall, F1, PR-AUC
- Confusion matrix
- Default-threshold performance

### 10.2 Experiment B: Feature Processing Comparison

Purpose:

- Compare raw features, log transformations, and size-based engineered features.

Comparison groups:

1. Raw numeric features.
2. Compact feature set after dropping constant columns and ID/name.
3. Feature set with log transformations.
4. Feature set with `est_diameter_mean` and related size features.

### 10.3 Experiment C: Model Comparison

Purpose:

- Compare linear and tree-based models on imbalanced data.

Candidate models:

- Logistic Regression
- Logistic Regression with class weight
- Random Forest
- Random Forest with class weight
- HistGradientBoosting / Gradient Boosting
- XGBoost
- LightGBM
- Tuned Random Forest / HistGradientBoosting / XGBoost / LightGBM

### 10.3.1 Experiment C-2: Hyperparameter Tuning

Purpose:

- Compare default model settings with lightweight tuned variants.
- Use PR-AUC as the tuning objective so the search is aligned with imbalanced classification.

Outputs:

- `hyperparameter_tuning_results.csv`
- validation metrics for tuned models
- decision evidence for whether a tuned model replaces the final model

### 10.4 Experiment D: Threshold Tuning

Purpose:

- Find a decision threshold more suitable than `0.5` for this scenario.

Outputs:

- Threshold vs precision / recall / F1 table
- Precision-recall curve
- Explanation of the selected threshold

### 10.5 Experiment E: Probability Calibration

Purpose:

- Check whether model probabilities can be used as risk scores.

Outputs:

- Calibration curve
- Brier score
- Before-and-after calibration comparison

### 10.6 Experiment F: Explainability Analysis

Purpose:

- Provide global and local explanations for the final report.

Outputs:

- Feature importance
- SHAP summary plot
- SHAP bar plot
- 2 to 3 case explanations

---

## 11. Expected Outcomes

### 11.1 Technical Outcomes

- Complete EDA results.
- At least one baseline model.
- Comparison of at least two main models.
- Imbalance handling and threshold tuning results.
- Hazardous probability risk score.
- Calibration check results.
- Global and local explanation figures.

### 11.2 Report Outcomes

The final report should include:

- Problem background and dataset introduction.
- Data quality and target distribution analysis.
- Methodology pipeline diagram.
- Model comparison table.
- Threshold tuning analysis.
- Calibration analysis.
- SHAP / feature importance explanations.
- Representative case studies.
- Limitations and future improvements.

### 11.3 Presentation Outcomes

The presentation should focus on:

1. The task is binary classification, but the model outputs a risk probability.
2. Class imbalance makes accuracy unreliable.
3. Threshold tuning makes the model more aligned with planetary defense screening.
4. SHAP explains why an object is assigned a high or low risk score.
5. The model's limitation: dataset hazard labels are not the same as real impact probabilities.

---

## 12. Project File Plan

Recommended structure for later implementation:

```text
Data-Mining-Projects/
├── data/
│   └── neo.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── project-proposal/
├── plan/
│   ├── plan_ZH.md
│   └── plan_EN.md
├── README.md
├── pyproject.toml
└── uv.lock
```

The project is still in the planning stage. This structure is only a recommendation for the implementation phase and will not be created automatically at this stage.

### 12.1 Python Environment Management

The implementation phase will use `uv` to manage the Python environment and dependencies. This avoids mixing the system Python, manually created virtual environments, and ad hoc `pip` installation states.

Planned principles:

- Use `pyproject.toml` to declare project dependencies.
- Use `uv.lock` to pin reproducible dependency versions.
- Use `uv add <package>` to add dependencies.
- Use `uv run <command>` to run Python scripts, notebook-kernel commands, and test commands.
- If the execution environment restricts the default cache location, temporarily set `UV_CACHE_DIR` to a writable directory.

Expected core dependencies include:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `shap`
- `jupyter`
- `imbalanced-learn` if undersampling or SMOTE experiments are included
- `xgboost`
- `lightgbm`

---

## 13. Schedule

| Stage | Work Items | Expected Output |
|---|---|---|
| Stage 1 | Data verification, EDA, column checks | EDA figures and data quality summary |
| Stage 2 | Baseline model, data split, initial evaluation | Baseline metric table |
| Stage 3 | Feature engineering and model comparison | Model comparison table and best candidate model |
| Stage 4 | Imbalance handling and threshold tuning | Threshold analysis and PR curve |
| Stage 5 | Calibration and risk score checking | Calibration curve and Brier score |
| Stage 6 | SHAP and case explanations | Global and local explanation figures |
| Stage 7 | Report and presentation preparation | Final report, slides, figures |

---

## 14. Risks and Limitations

### 14.1 Label Limitation

`hazardous` is a dataset label. It is not the same as actual Earth impact probability. The model learns the relationship between the dataset's hazardous label and the available observed features.

### 14.2 Limited Features

The dataset contains only a limited set of attributes and does not include complete orbital parameters. A more physically realistic hazard model may require features such as semi-major axis, eccentricity, orbital inclination, MOID, and other orbital descriptors.

### 14.3 Class Imbalance

Hazardous samples account for only about 9.73% of the dataset. The model may be biased toward the non-hazardous class, so recall, PR-AUC, and false negatives must be closely monitored.

### 14.4 Probability Interpretation Risk

Uncalibrated model probabilities may be overconfident or underconfident. The report must distinguish among:

- classification probability
- calibrated probability
- real-world physical hazard probability

This project focuses on the first two and does not claim to predict true impact probability.

### 14.5 Explainability Limitation

SHAP explains how the model uses features, but it does not prove causality. If the model learns dataset bias, SHAP will also reflect that bias.

---

## 15. Success Criteria

The implementation can be considered successful if it achieves the following:

1. Builds a reproducible data processing and model evaluation workflow.
2. Compares at least a baseline, a linear model, and a tree-based model.
3. Uses metrics suitable for imbalanced data instead of relying only on accuracy.
4. Performs threshold tuning and explains the chosen threshold.
5. Outputs hazardous probability and checks calibration for the final model.
6. Provides global feature importance and 2 to 3 local case explanations.
7. Uses `uv` to manage the Python environment and dependencies, keeping `pyproject.toml` and `uv.lock` for reproducibility.
8. Clearly states model capabilities and limitations, especially the correct interpretation of hazardous probability.

---

## 16. Final Positioning

The most appropriate positioning for this project is:

> An explainable probability-based binary classification system that estimates the probability of a NEO being labeled as hazardous from observed features and uses the risk score to support expert prioritization.

This framing preserves the rigor of binary classification while making threshold tuning, probability calibration, and SHAP-based explanation more suitable for the planetary defense screening scenario.
