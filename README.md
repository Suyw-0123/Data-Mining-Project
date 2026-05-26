# Data-Mining-Projects

Data Mining Term Project

## Data set

[Kaggle](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)

## Proposal

[English](project-proposal/Proposal_Report_EN.md)

[Chinese](project-proposal/Proposal_Report_ZH.md)

## Plan

[English](plan/plan_EN.md)

[Chinese](plan/plan_ZH.md)

## Implementation

This project uses `uv` for Python environment and dependency management.

Run the full pipeline:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv sync
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run neo-eda
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run neo-train
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv run neo-explain
```

Main outputs:

- `reports/tables/model_metrics_validation.csv`
- `reports/tables/hyperparameter_tuning_results.csv`
- `reports/tables/final_test_metrics.csv`
- `reports/tables/threshold_tuning_validation_calibrated.csv`
- `reports/tables/permutation_importance.csv`
- `reports/tables/shap_local_case_contributions.csv`
- `reports/figures/final_precision_recall_curve.png`
- `reports/figures/final_calibration_curve.png`
- `reports/figures/shap_global_bar.png`
- `reports/figures/shap_summary_beeswarm.png`

The saved model artifact is written to `models/final_model.joblib` and is ignored by git because it is generated and relatively large.
