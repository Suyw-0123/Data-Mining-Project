# NEO Hazard Risk — Data Mining Term Project

Binary classification of NASA Near-Earth Objects (NEOs) as hazardous or non-hazardous,
with probability calibration, threshold tuning, and SHAP-based explainability.

## Documentation

| Document | Link |
|---|---|
| **Wiki (EN)** | [Home\_EN.md](docs/wiki/EN/Home_EN.md) |
| **Wiki (ZH)** | [Home\_ZH.md](docs/wiki/ZH/Home_ZH.md) |
| Paper (EN) | [Paper\_EN.md](docs/paper/Paper_EN.md) |
| Paper (ZH) | [Paper\_ZH.md](docs/paper/Paper_ZH.md) |
| Proposal (EN) | [Proposal\_Report\_EN.md](docs/project-proposal/Proposal_Report_EN.md) |
| Proposal (ZH) | [Proposal\_Report\_ZH.md](docs/project-proposal/Proposal_Report_ZH.md) |
| Plan (EN) | [plan\_EN.md](docs/plan/plan_EN.md) |
| Plan (ZH) | [plan\_ZH.md](docs/plan/plan_ZH.md) |
| Dataset (Kaggle) | [NASA Nearest Earth Objects](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects) |

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.12 | [python.org](https://www.python.org/downloads/) |
| uv | any recent | `pip install uv` or [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |

---

## Quickstart

### 1. Clone the repository

```bash
git clone <repo-url>
cd Data-Mining-Projects
```

### 2. Place the dataset

Download `neo.csv` from Kaggle and put it at:

```
data/neo.csv
```

### 3. Install dependencies

```bash
uv sync
```

This creates a virtual environment and installs all packages listed in `pyproject.toml`.

> **Note (shared / restricted environments)**
> If the default cache locations are not writable, prefix every `uv` command with:
> ```bash
> UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache uv ...
> ```

### 4. Run the pipeline

Execute the three stages **in order**:

```bash
# Stage 1 — Exploratory Data Analysis
uv run neo-eda

# Stage 2 — Model training, tuning, and evaluation
uv run neo-train

# Stage 3 — Permutation importance and SHAP explanations
#            (requires neo-train to have run first)
uv run neo-explain
```

Each stage prints a short summary and the paths of files it wrote.

---

## Project structure

```
.
├── data/
│   └── neo.csv                  ← dataset (not tracked by git)
├── docs/
│   ├── wiki/                    ← technical wiki (EN + ZH)
│   ├── paper/                   ← final paper (EN + ZH)
│   ├── plan/                    ← implementation plan (EN + ZH)
│   └── project-proposal/        ← project proposal (EN + ZH)
├── models/
│   └── final_model.joblib       ← saved artifact (generated, not tracked by git)
├── reports/
│   ├── figures/                 ← PNG plots
│   └── tables/                  ← CSV / JSON outputs
├── src/neo_hazard/
│   ├── config.py                ← paths and shared constants
│   ├── data.py                  ← data loading and validation
│   ├── eda.py                   ← neo-eda entry point
│   ├── evaluation.py            ← metrics and threshold selection
│   ├── explain.py               ← neo-explain entry point
│   ├── features.py              ← feature engineering
│   ├── plots.py                 ← figure helpers
│   └── train.py                 ← neo-train entry point
├── pyproject.toml
└── uv.lock
```

---

## Key outputs

After running all three stages the following files are produced:

| File | Description |
|---|---|
| `reports/tables/dataset_summary.json` | Row/column counts, class balance |
| `reports/tables/numeric_summary.csv` | Descriptive statistics per feature |
| `reports/tables/class_distribution.csv` | Class counts and ratios |
| `reports/tables/correlation_matrix.csv` | Pearson correlation matrix |
| `reports/tables/model_metrics_validation.csv` | Validation metrics for all models |
| `reports/tables/hyperparameter_tuning_results.csv` | Best CV score per tuned model |
| `reports/tables/threshold_tuning_validation_calibrated.csv` | F1/recall/precision at each threshold |
| `reports/tables/final_test_metrics.csv` | Test-set metrics for the chosen model |
| `reports/tables/final_test_predictions.csv` | Per-row predictions on the test set |
| `reports/tables/permutation_importance.csv` | Feature importance by PR-AUC drop |
| `reports/tables/shap_local_case_contributions.csv` | Top SHAP features for selected cases |
| `reports/figures/target_distribution.png` | Class imbalance bar chart |
| `reports/figures/numeric_distributions.png` | Histograms of raw features |
| `reports/figures/correlation_heatmap.png` | Pearson heatmap |
| `reports/figures/final_precision_recall_curve.png` | PR curve on test set |
| `reports/figures/final_roc_curve.png` | ROC curve on test set |
| `reports/figures/final_calibration_curve.png` | Probability calibration curve |
| `reports/figures/permutation_importance.png` | Bar chart of permutation importance |
| `reports/figures/shap_global_bar.png` | SHAP global feature importance |
| `reports/figures/shap_summary_beeswarm.png` | SHAP beeswarm summary plot |
| `models/final_model.joblib` | Serialised model artifact |
