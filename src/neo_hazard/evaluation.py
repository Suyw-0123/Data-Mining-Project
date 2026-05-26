from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def probabilities_or_scores(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return positive-class probabilities,
    falling back to sigmoid-scaled scores if needed.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probabilities or scores.")


def metric_row(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    *,
    model_name: str,
    split: str,
    threshold: float = 0.5,
) -> dict[str, float | int | str]:
    """
    Calculate classification, ranking, calibration,
    and confusion-matrix metrics.
    """
    y_pred = y_probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    row: dict[str, float | int | str] = {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_probability),
        "brier_score": brier_score_loss(y_true, y_probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, y_probability)
    else:
        row["roc_auc"] = np.nan
    return row


def threshold_table(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate many probability thresholds on the same validation predictions.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    rows = []
    for threshold in thresholds:
        row = metric_row(
            y_true,
            y_probability,
            model_name="threshold_candidate",
            split="validation",
            threshold=float(threshold),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame) -> float:
    """
    Select the threshold with best F1,
    using recall and precision as tie-breakers.
    """
    candidates = table.copy()
    candidates = candidates.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    )
    return float(candidates.iloc[0]["threshold"])
