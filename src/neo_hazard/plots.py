from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def set_plot_style() -> None:
    """
    Apply one consistent seaborn style for every generated figure.
    """
    sns.set_theme(style="whitegrid", context="notebook")


def save_target_distribution(df: pd.DataFrame, target: str, path) -> None:
    """
    Save a bar chart showing the class imbalance of the target label.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df, x=target, hue=target, palette=["#4C78A8", "#F58518"], ax=ax, legend=False
    )
    ax.set_title("Target Distribution")
    ax.set_xlabel("hazardous")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_numeric_distributions(df: pd.DataFrame, columns: list[str], path) -> None:
    """
    Save histograms for the main numeric variables used during EDA.
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, column in zip(axes, columns):
        sns.histplot(df[column], bins=50, ax=ax, color="#4C78A8")
        ax.set_title(column)
    for ax in axes[len(columns) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame, path) -> None:
    """
    Save a heatmap for the prepared correlation matrix.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Pearson Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_precision_recall_curve(y_true, y_probability, path) -> None:
    """
    Save the precision-recall curve
    used to assess imbalanced classification.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_probability, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_roc_curve(y_true, y_probability, path) -> None:
    """
    Save the ROC curve as a supplementary ranking-performance figure.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_probability, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_calibration_curve(model, X, y, path) -> None:
    """
    Save a calibration curve to check
    whether predicted probabilities are reliable.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_estimator(model, X, y, n_bins=10, strategy="quantile", ax=ax)
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
