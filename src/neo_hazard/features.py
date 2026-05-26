from __future__ import annotations

import pandas as pd

from neo_hazard.config import BASE_NUMERIC_FEATURES, CONSTANT_COLUMNS, ID_COLUMNS, TARGET
from neo_hazard.data import safe_log1p


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Create model features, target labels,
    and metadata kept for case explanations.
    """
    features = df[BASE_NUMERIC_FEATURES].copy()

    features["est_diameter_mean"] = (
        features["est_diameter_min"] + features["est_diameter_max"]
    ) / 2.0
    features["est_diameter_range"] = (
        features["est_diameter_max"] - features["est_diameter_min"]
    )
    features["log_est_diameter_mean"] = safe_log1p(features["est_diameter_mean"])
    features["log_relative_velocity"] = safe_log1p(features["relative_velocity"])
    features["log_miss_distance"] = safe_log1p(features["miss_distance"])

    y = df[TARGET].astype(bool)
    metadata = df[ID_COLUMNS + CONSTANT_COLUMNS].copy()
    return features, y, metadata
