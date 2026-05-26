from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neo_hazard.config import BASE_NUMERIC_FEATURES, EXPECTED_COLUMNS, TARGET


def load_neo_data(path: Path) -> pd.DataFrame:
    """
    Load the NEO CSV, validate required columns,
    and normalize target/numeric dtypes.
    """
    df = pd.read_csv(path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = df.copy()
    for column in BASE_NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[TARGET].dtype != bool:
        df[TARGET] = df[TARGET].map({True: True, False: False, "True": True, "False": False})
    if df[TARGET].isna().any():
        raise ValueError("Target column contains values other than True/False.")

    return df


def summarize_dataset(df: pd.DataFrame) -> dict[str, object]:
    """
    Return high-level dataset counts used in reports and sanity checks.
    """
    target_counts = df[TARGET].value_counts().to_dict()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "hazardous_true": int(target_counts.get(True, 0)),
        "hazardous_false": int(target_counts.get(False, 0)),
        "hazardous_rate": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for the base numeric features.
    """
    summary = df[BASE_NUMERIC_FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T
    return summary.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Pearson correlation matrix with the boolean target encoded as 0/1.
    """
    corr_df = df[BASE_NUMERIC_FEATURES + [TARGET]].copy()
    corr_df[TARGET] = corr_df[TARGET].astype(int)
    corr = corr_df.corr(numeric_only=True)
    return corr.round(6)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize class counts and ratios for the hazardous target.
    """
    counts = df[TARGET].value_counts().rename_axis(TARGET).reset_index(name="count")
    counts["ratio"] = counts["count"] / len(df)
    return counts.sort_values(TARGET).reset_index(drop=True)


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing-value counts and ratios for every input column.
    """
    table = df.isna().sum().rename("missing_count").reset_index()
    table = table.rename(columns={"index": "column"})
    table["missing_ratio"] = table["missing_count"] / len(df)
    return table


def constant_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    List columns with very low cardinality
    so constant fields are easy to inspect.
    """
    rows = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)
        if unique_count <= 3:
            values = sorted(map(str, df[column].drop_duplicates().tolist()))
            rows.append({"column": column, "unique_count": unique_count, "values": ", ".join(values)})
    return pd.DataFrame(rows)


def safe_log1p(series: pd.Series) -> pd.Series:
    """
    Apply log1p only after verifying the feature has no negative values.
    """
    if (series < 0).any():
        raise ValueError(f"{series.name} contains negative values and cannot use log1p.")
    return np.log1p(series)
