"""AcademIQ — Preprocessing pipeline builder.

Constructs sklearn Pipelines that encapsulate ALL preprocessing:
  1. ColumnTransformer (ordinal encoding, one-hot encoding, numeric passthrough)
  2. FeatureEngineer (optional engineered features)
  3. Scaler (model-dependent: StandardScaler for linear, passthrough for trees)
  4. Model

This guarantees training and inference use identical transformations.
"""

import logging
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.config import (
    get_nominal_features,
    get_numeric_features,
    get_ordinal_features,
    get_random_seed,
)
from src.preprocessing.transformers import FeatureEngineer

logger = logging.getLogger(__name__)

# Models that do NOT require feature scaling
TREE_MODELS = (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    DummyRegressor,
    DummyClassifier,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer that handles all encoding.

    - Numeric features: impute with median, pass through
    - Ordinal features: impute with most-frequent, encode as 0/1/2/...
    - Nominal features: impute with most-frequent, one-hot encode (drop first)

    Returns:
        Fitted-once ColumnTransformer with named outputs.
    """
    numeric_features = get_numeric_features()
    ordinal_features = get_ordinal_features()
    nominal_features = get_nominal_features()

    ordinal_cols = list(ordinal_features.keys())
    ordinal_categories = list(ordinal_features.values())

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="infrequent_if_exist",
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("ord", ordinal_pipeline, ordinal_cols),
            ("nom", nominal_pipeline, nominal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(
    model: Any,
    use_feature_engineering: bool = True,
    use_scaler: bool | None = None,
) -> Pipeline:
    """Build a complete preprocessing + model pipeline.

    Args:
        model: sklearn estimator (regressor or classifier).
        use_feature_engineering: Whether to include FeatureEngineer step.
        use_scaler: Whether to include StandardScaler.
            If None, auto-detect: True for linear models, False for tree models.

    Returns:
        sklearn Pipeline ready for fit/predict.
    """
    if use_scaler is None:
        use_scaler = not isinstance(model, TREE_MODELS)

    steps = [("preprocessor", build_preprocessor())]

    if use_feature_engineering:
        steps.append(("feature_engineer", FeatureEngineer()))

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    pipeline = Pipeline(steps)
    logger.info(
        "Built pipeline: %s",
        " -> ".join(name for name, _ in steps),
    )
    return pipeline


# --- Model constructors (using config-defined seed) ---


def get_regression_models() -> dict[str, Any]:
    """Return a dict of regression model instances."""
    seed = get_random_seed()
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "HistGBM": HistGradientBoostingRegressor(random_state=seed),
    }


def get_classification_models() -> dict[str, Any]:
    """Return a dict of classification model instances."""
    seed = get_random_seed()
    return {
        "DummyClassifier": DummyClassifier(strategy="stratified", random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "HistGBM": HistGradientBoostingClassifier(random_state=seed),
    }
