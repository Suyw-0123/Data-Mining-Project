from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd

from neo_hazard.config import DATA_PATH, TABLES_DIR, ensure_output_dirs
from neo_hazard.data import load_neo_data
from neo_hazard.evaluation import probabilities_or_scores
from neo_hazard.features import build_feature_frame
from neo_hazard.train import build_models, split_data


def _median_seconds(durations: list[float]) -> float:
    """Return the median of a list of elapsed-time samples."""
    return float(np.median(durations))


def measure_fit_time(model: object, X_train: pd.DataFrame, y_train: pd.Series, *, repeats: int) -> float:
    """
    Fit a fresh clone of the model `repeats` times and
    return the median wall-clock fit time in seconds.
    """
    from sklearn.base import clone

    durations: list[float] = []
    for _ in range(repeats):
        candidate = clone(model)
        start = time.perf_counter()
        candidate.fit(X_train, y_train)
        durations.append(time.perf_counter() - start)
    return _median_seconds(durations)


def measure_predict_time(
    fitted_model: object,
    X_test: pd.DataFrame,
    *,
    repeats: int,
    single_repeats: int,
) -> tuple[float, float]:
    """
    Measure batch and single-row scoring latency for a fitted model.

    Returns (batch_seconds, single_row_seconds) as medians over repeats,
    using the same probability/score call the pipeline relies on.
    """
    batch_durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        probabilities_or_scores(fitted_model, X_test)
        batch_durations.append(time.perf_counter() - start)

    single_row = X_test.iloc[[0]]
    single_durations: list[float] = []
    for _ in range(single_repeats):
        start = time.perf_counter()
        probabilities_or_scores(fitted_model, single_row)
        single_durations.append(time.perf_counter() - start)

    return _median_seconds(batch_durations), _median_seconds(single_durations)


def benchmark_models(
    *,
    fit_repeats: int = 3,
    predict_repeats: int = 10,
    single_repeats: int = 200,
) -> pd.DataFrame:
    """
    Benchmark training and prediction efficiency for every base model
    on the same features and train/test split used by `neo-train`.
    """
    df = load_neo_data(DATA_PATH)
    X, y, metadata = build_feature_frame(df)
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data(X, y, metadata)

    rows = []
    for name, model in build_models(y_train).items():
        fit_seconds = measure_fit_time(model, X_train, y_train, repeats=fit_repeats)

        # Use the fitted instance from the final fit for prediction timing.
        model.fit(X_train, y_train)
        batch_seconds, single_seconds = measure_predict_time(
            model,
            X_test,
            repeats=predict_repeats,
            single_repeats=single_repeats,
        )

        rows.append(
            {
                "model": name,
                "fit_time_s": round(fit_seconds, 4),
                "predict_batch_ms": round(batch_seconds * 1e3, 3),
                "throughput_rows_per_s": round(len(X_test) / batch_seconds, 1),
                "predict_single_ms": round(single_seconds * 1e3, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("fit_time_s").reset_index(drop=True)


def main() -> None:
    """
    Run the efficiency benchmark and export the results table
    plus an environment summary for reproducibility.
    """
    ensure_output_dirs()
    results = benchmark_models()
    results.to_csv(TABLES_DIR / "efficiency_benchmark.csv", index=False)

    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "note": "Single-machine, CPU-only timings; n_jobs=-1 where the model supports it.",
    }
    (TABLES_DIR / "efficiency_benchmark_environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Efficiency benchmark complete.")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
