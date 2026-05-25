from __future__ import annotations

import json

from neo_hazard.config import BASE_NUMERIC_FEATURES, DATA_PATH, FIGURES_DIR, TABLES_DIR, TARGET, ensure_output_dirs
from neo_hazard.data import (
    class_distribution,
    constant_value_table,
    correlation_table,
    load_neo_data,
    missing_value_table,
    numeric_summary,
    summarize_dataset,
)
from neo_hazard.plots import (
    save_correlation_heatmap,
    save_numeric_distributions,
    save_target_distribution,
)


def main() -> None:
    ensure_output_dirs()
    df = load_neo_data(DATA_PATH)

    summary = summarize_dataset(df)
    (TABLES_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    missing_value_table(df).to_csv(TABLES_DIR / "missing_values.csv", index=False)
    constant_value_table(df).to_csv(TABLES_DIR / "constant_values.csv", index=False)
    class_distribution(df).to_csv(TABLES_DIR / "class_distribution.csv", index=False)
    numeric_summary(df).to_csv(TABLES_DIR / "numeric_summary.csv")

    corr = correlation_table(df)
    corr.to_csv(TABLES_DIR / "correlation_matrix.csv")

    save_target_distribution(df, TARGET, FIGURES_DIR / "target_distribution.png")
    save_numeric_distributions(df, BASE_NUMERIC_FEATURES, FIGURES_DIR / "numeric_distributions.png")
    save_correlation_heatmap(corr, FIGURES_DIR / "correlation_heatmap.png")

    print("EDA complete.")
    print(f"Rows: {summary['rows']}, columns: {summary['columns']}")
    print(f"Outputs: {TABLES_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
