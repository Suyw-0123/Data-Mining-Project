from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "neo.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET = "hazardous"
ID_COLUMNS = ["id", "name"]
CONSTANT_COLUMNS = ["orbiting_body", "sentry_object"]
BASE_NUMERIC_FEATURES = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]
EXPECTED_COLUMNS = ID_COLUMNS + BASE_NUMERIC_FEATURES[:4] + CONSTANT_COLUMNS + [
    "absolute_magnitude",
    TARGET,
]


def ensure_output_dirs() -> None:
    for path in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
