# Data Mining Term Project Proposal (English)


## Selected Dataset
- Kaggle: https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects
- Source noted by dataset author: NASA Open API + JPL CNEOS close-approach data
- License: CC0 (Public Domain)

---

## 1) Scenario (10%)

Near-Earth Objects (NEOs) pass close to Earth every day. Most are harmless, but a small subset may become hazardous if they are large enough and approach Earth with risky orbital conditions.  
Assume we are part of a planetary defense analytics team. Our objective is to build a data-driven screening model that helps prioritize which newly observed NEOs should be checked first by experts. This reduces manual workload and improves early warning response.  
In addition, the team must provide **human-readable explanations** for each prediction (why a NEO is predicted as hazardous/non-hazardous), so domain experts can trust and audit the model’s decisions.

---

## 2) Problem Definition (15%)

### Prediction Task
Given observed NEO attributes, **predict whether the object is hazardous (`hazardous` = True/False)**.

### Explainability Task (XAI)
For each prediction, generate both:
- **Global explanation**: overall feature influence across the dataset
- **Local explanation**: per-object reason for a specific prediction

### Learning Type
- Supervised learning
- Binary classification

### Input Features (planned)
- `est_diameter_min`
- `est_diameter_max`
- `relative_velocity`
- `miss_distance`
- `absolute_magnitude`
- (plus optional engineered features)

### Target Variable
- `hazardous`

### Why this is a valid data mining problem
- It is a clear predictive task
- It has measurable outcomes (precision/recall/F1/ROC-AUC)
- It has practical value for risk triage in astronomy/planetary defense
- It includes interpretable decision support via Explainable AI (XAI), not only black-box classification

---

## 3) Dataset Observation (40%)

### 3.1 Dataset Requirement Check
- Our selected dataset contains **90,836 instances** and **10 columns** 

Columns:
`id`, `name`, `est_diameter_min`, `est_diameter_max`, `relative_velocity`, `miss_distance`, `orbiting_body`, `sentry_object`, `absolute_magnitude`, `hazardous`

### 3.2 Basic Statistics

#### Data Quality
- Missing values: **0 in all 10 columns**
- `orbiting_body`: all values are `Earth` (single-value column)
- `sentry_object`: all values are `False` (single-value column)

#### Class Distribution (Target)
- `hazardous = True`: **8,840** (9.73%)
- `hazardous = False`: **81,996** (90.27%)
- Imbalance ratio ≈ **1 : 9.27**

#### Numeric Feature Summary (from local inspection)

Method note: all statistics were computed from the same `neo.csv` snapshot. Reported correlations use Pearson correlation; for correlation with `hazardous`, labels were encoded as True=1 and False=0.

| Feature | Min | Q1 | Median | Q3 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|
| est_diameter_min | 0.000609 | 0.019256 | 0.048368 | 0.143402 | 37.892650 | 0.127432 |
| est_diameter_max | 0.001362 | 0.043057 | 0.108153 | 0.320656 | 84.730541 | 0.284947 |
| relative_velocity | 203.35 | 28617.55 | 44190.11 | 62923.54 | 236990.13 | 48066.92 |
| miss_distance | 6745.53 | 17210647.11 | 37845843.44 | 56548383.98 | 74798651.45 | 37066546.03 |
| absolute_magnitude | 9.23 | 21.34 | 23.70 | 25.70 | 33.20 | 23.53 |

#### Pearson Correlation Summary (supports Section 3.3 findings)

| Pair | Pearson r | Interpretation |
|---|---:|---|
| hazardous vs absolute_magnitude | -0.365 | Moderate negative relation; lower magnitude (brighter/larger tendency) is associated with higher hazard probability |
| hazardous vs relative_velocity | 0.191 | Weak-to-moderate positive relation; faster objects are more likely to be hazardous |
| hazardous vs miss_distance | 0.042 | Very weak positive relation; miss distance alone has limited predictive power |
| est_diameter_min vs est_diameter_max | 1.000 | Near-perfect collinearity (strong redundancy) |
| absolute_magnitude vs est_diameter_min | -0.560 | Moderate negative relation; consistent with known magnitude–size behavior |
| absolute_magnitude vs est_diameter_max | -0.560 | Moderate negative relation; consistent with known magnitude–size behavior |

### 3.3 Properties, Patterns, and Interesting Findings

1. **Strong redundancy**: `est_diameter_min` and `est_diameter_max` are almost perfectly collinear (corr ≈ 1.0).
2. **Magnitude–diameter relation**: `absolute_magnitude` is moderately negatively correlated with diameter (corr ≈ -0.56), consistent with physical intuition.
3. **Class imbalance is significant**: only about 9.7% hazardous objects, so plain accuracy can be misleading.
4. **Non-informative columns exist**: `orbiting_body` and `sentry_object` have no variance in this dataset snapshot.
5. **Hazard signal is multi-factor**: correlation with hazard is stronger for `absolute_magnitude` (~ -0.365) and `relative_velocity` (~ 0.191) than for `miss_distance` (~ 0.042), implying distance alone is insufficient.
6. **Interpretability need is explicit**: because hazard decisions are safety-related, predictions should be accompanied by feature-level explanations so users can validate model behavior.

---

## 4) Challenges (25%)

### Challenge 1: Class imbalance (hazardous is minority)
**Why difficult:** a naive classifier can achieve high accuracy by predicting “non-hazardous” most of the time, but miss important positives.

### Challenge 2: Redundant / low-information features
**Why difficult:** highly correlated or constant columns can reduce model robustness, increase overfitting risk, and weaken interpretability.

### Challenge 3: Wide-range and skewed numeric distributions
**Why difficult:** features such as velocity/distance span very large ranges, which may destabilize certain models and hurt threshold quality.

### Challenge 4: Explainability and trustworthiness of predictions
**Why difficult:** a high score alone is not enough in safety-oriented scenarios; we need consistent, understandable reasons for each prediction, and explanations must align with domain intuition (e.g., size/velocity effects).

---

## 5) To-dos (10%)

### To-dos for Challenge 1 (imbalance)
1. Use **stratified split + stratified k-fold CV**.
2. Compare **class-weighted learning** vs. **sampling methods** (e.g., RandomUnderSampler / SMOTE).
3. Optimize decision threshold using **F1 / PR-AUC / recall-oriented criteria**.

### To-dos for Challenge 2 (feature redundancy)
1. Drop or transform low-information columns (`orbiting_body`, `sentry_object`).
2. Perform correlation screening and test compact feature sets.
3. Add explainability checks (feature importance / SHAP for final model).

### To-dos for Challenge 3 (scale and distribution)
1. Test `log1p` transforms for skewed continuous variables.
2. Compare scaling strategies (StandardScaler vs RobustScaler where applicable).
3. Evaluate both linear and tree-based baselines (e.g., Logistic Regression, Random Forest, XGBoost/LightGBM).

### To-dos for Challenge 4 (Explainable AI)
1. Build an explainability pipeline using **SHAP** (global importance + per-instance local explanation).
2. Add interpretable baseline analysis (e.g., Logistic Regression coefficients) and compare with tree-model SHAP attributions.
3. Design report outputs for presentation: top global factors, plus 2–3 case studies showing why model predicts hazardous/non-hazardous.


---

## References

- Kaggle dataset page: https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects
- NASA Open API: https://api.nasa.gov/
- JPL CNEOS close-approach portal: https://cneos.jpl.nasa.gov/ca/
