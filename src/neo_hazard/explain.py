from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from neo_hazard.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TABLES_DIR, ensure_output_dirs
from neo_hazard.evaluation import probabilities_or_scores
from neo_hazard.plots import set_plot_style


def save_permutation_importance(model, X_test, y_test) -> pd.DataFrame:
    """
    Compute PR-AUC permutation importance
    and save both table and bar chart outputs.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    table.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    top = table.head(12).sort_values("importance_mean", ascending=True)
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C78A8")
    ax.set_title("Permutation Importance by PR-AUC")
    ax.set_xlabel("Mean importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)
    return table


def selected_cases(meta_test: pd.DataFrame, y_test: pd.Series, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Pick representative TP/FN/FP cases
    for local model explanation in the report.
    """
    cases = meta_test.reset_index(drop=False).rename(columns={"index": "source_index"}).copy()
    cases["true_hazardous"] = y_test.reset_index(drop=True)
    cases["hazard_probability"] = probability
    cases["predicted_hazardous"] = cases["hazard_probability"] >= threshold

    selections = []
    tp = cases[(cases["true_hazardous"]) & (cases["predicted_hazardous"])]
    fn = cases[(cases["true_hazardous"]) & (~cases["predicted_hazardous"])]
    fp = cases[(~cases["true_hazardous"]) & (cases["predicted_hazardous"])]
    tn = cases[(~cases["true_hazardous"]) & (~cases["predicted_hazardous"])]

    if not tp.empty:
        selections.append(tp.sort_values("hazard_probability", ascending=False).head(1).assign(case_type="true_positive"))
    if not fn.empty:
        selections.append(fn.sort_values("hazard_probability", ascending=False).head(1).assign(case_type="false_negative"))
    if not fp.empty:
        selections.append(fp.sort_values("hazard_probability", ascending=False).head(1).assign(case_type="false_positive"))
    elif not tn.empty:
        selections.append(tn.sort_values("hazard_probability", ascending=False).head(1).assign(case_type="high_score_true_negative"))

    selected = pd.concat(selections, ignore_index=True)
    selected["threshold"] = threshold
    selected.to_csv(TABLES_DIR / "selected_explanation_cases.csv", index=False)
    return selected


def run_shap_explanations(model, X_train: pd.DataFrame, X_test: pd.DataFrame, selected: pd.DataFrame) -> str:
    """
    Generate global SHAP plots and
    top local SHAP contributions for selected cases.
    """
    try:
        import shap
    except ImportError:
        return "SHAP is not installed; skipped SHAP outputs."

    background = X_train.sample(min(80, len(X_train)), random_state=RANDOM_STATE)
    explain_sample = X_test.sample(min(120, len(X_test)), random_state=RANDOM_STATE)

    def predict_positive(data):
        """Adapt SHAP array inputs back into DataFrames
        before model prediction."""
        frame = pd.DataFrame(data, columns=X_train.columns)
        return probabilities_or_scores(model, frame)

    explainer = shap.Explainer(predict_positive, background)
    shap_values = explainer(explain_sample)

    plt.figure()
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    local_rows = []
    selected_indices = selected["source_index"].tolist()
    local_X = X_test.loc[selected_indices]
    local_values = explainer(local_X)
    for row_position, source_index in enumerate(selected_indices):
        values = np.asarray(local_values.values[row_position])
        order = np.argsort(np.abs(values))[::-1][:5]
        for rank, feature_index in enumerate(order, start=1):
            feature = X_test.columns[feature_index]
            local_rows.append(
                {
                    "source_index": source_index,
                    "rank": rank,
                    "feature": feature,
                    "feature_value": float(local_X.iloc[row_position, feature_index]),
                    "shap_value": float(values[feature_index]),
                    "direction": "pushes_probability_up" if values[feature_index] > 0 else "pushes_probability_down",
                }
            )

    pd.DataFrame(local_rows).to_csv(TABLES_DIR / "shap_local_case_contributions.csv", index=False)
    return "SHAP outputs created."


def main() -> None:
    """
    Load the trained artifact and export
    permutation-importance and SHAP explanations.
    """
    ensure_output_dirs()
    artifact_path = MODELS_DIR / "final_model.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError("Run `uv run neo-train` before `uv run neo-explain`.")

    artifact = joblib.load(artifact_path)
    model = artifact["calibrated_model"]
    threshold = float(artifact["calibrated_threshold"])
    X_train = artifact["X_train"]
    X_test = artifact["X_test"]
    y_test = artifact["y_test"]
    meta_test = artifact["meta_test"]

    probabilities = probabilities_or_scores(model, X_test)
    importance = save_permutation_importance(model, X_test, y_test)
    selected = selected_cases(meta_test, y_test, probabilities, threshold)
    shap_status = run_shap_explanations(model, X_train, X_test, selected)

    summary = {
        "threshold": threshold,
        "top_permutation_features": importance.head(5)["feature"].tolist(),
        "selected_case_count": int(len(selected)),
        "shap_status": shap_status,
    }
    (TABLES_DIR / "explainability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Explainability complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
