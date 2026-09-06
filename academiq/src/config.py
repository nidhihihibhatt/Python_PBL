"""AcademIQ — Configuration loader.

Loads YAML configuration files and provides centralised access
to all project constants (random seed, feature lists, thresholds, paths).
"""

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is the 'academiq' directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_config() -> dict:
    """Load the model configuration."""
    return load_yaml(CONFIGS_DIR / "model_config.yaml")


def get_app_config() -> dict:
    """Load the application configuration."""
    return load_yaml(CONFIGS_DIR / "app_config.yaml")


def get_recommendation_rules() -> list[dict]:
    """Load recommendation rules."""
    config = load_yaml(CONFIGS_DIR / "recommendation_rules.yaml")
    return config.get("rules", [])


# --- Convenience accessors for commonly used values ---

_model_config: dict | None = None


def _mc() -> dict:
    """Lazy-load model config."""
    global _model_config
    if _model_config is None:
        _model_config = get_model_config()
    return _model_config


def get_random_seed() -> int:
    return _mc()["random_seed"]


def get_target_column() -> str:
    return _mc()["target_column"]


def get_risk_threshold() -> int:
    return _mc()["risk_threshold"]


def get_numeric_features() -> list[str]:
    return _mc()["numeric_features"]


def get_ordinal_features() -> dict[str, list[str]]:
    return _mc()["ordinal_features"]


def get_nominal_features() -> list[str]:
    return _mc()["nominal_features"]


def get_all_input_features() -> list[str]:
    """Return all input feature names (before encoding)."""
    return (
        get_numeric_features()
        + list(get_ordinal_features().keys())
        + get_nominal_features()
    )


def get_raw_data_path() -> Path:
    return DATA_DIR / "raw" / "StudentPerformanceFactors.csv"
