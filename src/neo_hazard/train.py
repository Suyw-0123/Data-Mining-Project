from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neo_hazard.config import DATA_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TABLES_DIR, ensure_output_dirs
from neo_hazard.data import load_neo_data
from neo_hazard.evaluation import choose_threshold, metric_row, probabilities_or_scores, threshold_table
from neo_hazard.features import build_feature_frame
from neo_hazard.plots import save_calibration_curve, save_precision_recall_curve, save_roc_curve


def build_models() -> dict[str, object]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    }


def split_data(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame):
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X,
        y,
        metadata,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp,
        y_temp,
        meta_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test


def main() -> None:
    ensure_output_dirs()
    df = load_neo_data(DATA_PATH)
    X, y, metadata = build_feature_frame(df)
    splits = split_data(X, y, metadata)
    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = splits

    validation_rows = []
    fitted_models: dict[str, object] = {}
    validation_probabilities: dict[str, np.ndarray] = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_val_probability = probabilities_or_scores(model, X_val)
        validation_rows.append(
            metric_row(
                y_val,
                y_val_probability,
                model_name=name,
                split="validation",
                threshold=0.5,
            )
        )
        fitted_models[name] = model
        validation_probabilities[name] = y_val_probability

    validation_metrics = pd.DataFrame(validation_rows).sort_values(
        ["pr_auc", "f1", "recall"],
        ascending=[False, False, False],
    )
    validation_metrics.to_csv(TABLES_DIR / "model_metrics_validation.csv", index=False)

    best_model_name = str(validation_metrics.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_val_probability = validation_probabilities[best_model_name]

    thresholds = threshold_table(y_val, best_val_probability)
    best_threshold = choose_threshold(thresholds)
    thresholds.to_csv(TABLES_DIR / "threshold_tuning_validation.csv", index=False)

    calibrated_model = CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    calibrated_val_probability = probabilities_or_scores(calibrated_model, X_val)
    calibrated_thresholds = threshold_table(y_val, calibrated_val_probability)
    calibrated_threshold = choose_threshold(calibrated_thresholds)
    calibrated_thresholds.to_csv(TABLES_DIR / "threshold_tuning_validation_calibrated.csv", index=False)

    y_test_probability_raw = probabilities_or_scores(best_model, X_test)
    y_test_probability_calibrated = probabilities_or_scores(calibrated_model, X_test)

    test_rows = [
        metric_row(
            y_test,
            y_test_probability_raw,
            model_name=best_model_name,
            split="test",
            threshold=best_threshold,
        ),
        metric_row(
            y_test,
            y_test_probability_calibrated,
            model_name=f"{best_model_name}_calibrated",
            split="test",
            threshold=calibrated_threshold,
        ),
        metric_row(
            y_test,
            y_test_probability_calibrated,
            model_name=f"{best_model_name}_calibrated_default_threshold",
            split="test",
            threshold=0.5,
        ),
    ]
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(TABLES_DIR / "final_test_metrics.csv", index=False)

    save_precision_recall_curve(
        y_test,
        y_test_probability_calibrated,
        FIGURES_DIR / "final_precision_recall_curve.png",
    )
    save_roc_curve(
        y_test,
        y_test_probability_calibrated,
        FIGURES_DIR / "final_roc_curve.png",
    )
    save_calibration_curve(
        calibrated_model,
        X_test,
        y_test,
        FIGURES_DIR / "final_calibration_curve.png",
    )

    test_predictions = meta_test.reset_index(drop=False).rename(columns={"index": "source_index"})
    test_predictions["true_hazardous"] = y_test.reset_index(drop=True)
    test_predictions["hazard_probability"] = y_test_probability_calibrated
    test_predictions["threshold"] = calibrated_threshold
    test_predictions["predicted_hazardous"] = (
        test_predictions["hazard_probability"] >= calibrated_threshold
    )
    test_predictions.to_csv(TABLES_DIR / "final_test_predictions.csv", index=False)

    artifact = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "calibrated_model": calibrated_model,
        "raw_threshold": best_threshold,
        "calibrated_threshold": calibrated_threshold,
        "feature_columns": list(X.columns),
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_val": meta_val,
        "meta_test": meta_test,
    }
    joblib.dump(artifact, MODELS_DIR / "final_model.joblib", compress=3)

    summary = {
        "best_model_name": best_model_name,
        "raw_validation_best_threshold": best_threshold,
        "calibrated_validation_best_threshold": calibrated_threshold,
        "feature_count": len(X.columns),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "validation_metrics_file": str(TABLES_DIR / "model_metrics_validation.csv"),
        "test_metrics_file": str(TABLES_DIR / "final_test_metrics.csv"),
    }
    (TABLES_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
