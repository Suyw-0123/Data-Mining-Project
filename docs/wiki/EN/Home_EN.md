# NEO Hazard Risk — Project Wiki (English)

This wiki covers the complete technical documentation for the NEO Hazard Risk classification project.
Use the table below to navigate to the topic you need.

## Navigation

| Document | Description |
|---|---|
| [Dataset & EDA](Data_EN.md) | Data source, schema, quality checks, EDA findings |
| [Pipeline Architecture](Pipeline_EN.md) | End-to-end pipeline design, module roles, data flow |
| [Feature Engineering](Features_EN.md) | Base features, engineered features, transformations |
| [Models & Experiments](Models_EN.md) | All models, hyperparameter tuning, comparison |
| [Evaluation & Results](Evaluation_EN.md) | Metrics, threshold tuning, calibration, final test results |
| [Explainability](Explainability_EN.md) | Permutation importance, SHAP global and local analysis |
| [Code Reference](Code_EN.md) | Module-by-module source code guide |

## Chinese Version

中文 wiki 首頁：[點擊此處](../ZH/Home_ZH.md)

## Project in One Paragraph

This project builds an explainable binary classification system for predicting whether a
NASA Near-Earth Object (NEO) is labeled as hazardous.
The pipeline covers data loading and validation, EDA, feature engineering, model training and
hyperparameter tuning, threshold selection, probability calibration, and SHAP-based explainability.
The final model is a calibrated Balanced Random Forest that achieves F1 = 0.5332 and
Recall = 0.7474 on the test set at threshold = 0.19.
