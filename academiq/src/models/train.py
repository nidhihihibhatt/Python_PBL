"""AcademIQ — Model training orchestration.

Provides functions to train regression and classification models
with cross-validation and hyperparameter tuning, compare models,
and serialise the best pipelines.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    get_model_config,
    get_random_seed,
    get_risk_threshold,
    get_target_column,
)
from src.preprocessing.pipeline import (
    build_pipeline,
    get_classification_models,
    get_regression_models,
)

logger = logging.getLogger(__name__)


def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate features and targets from the raw DataFrame.

    Returns:
        X: Feature DataFrame (all columns except target).
        y_reg: Regression target (Exam_Score).
        y_cls: Classification target (At_Risk: 1 if below threshold, else 0).
    """
    target_col = get_target_column()
    threshold = get_risk_threshold()

    X = df.drop(columns=[target_col])
    y_reg = df[target_col].copy()
    y_cls = (df[target_col] < threshold).astype(int)
    y_cls.name = "At_Risk"

    logger.info(
        "Data prepared: X=%s, y_reg=%s, At_Risk distribution: %s",
        X.shape,
        y_reg.shape,
        y_cls.value_counts().to_dict(),
    )
    return X, y_reg, y_cls


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_feature_engineering: bool = True,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Train and compare all regression models.

    Args:
        X_train, y_train: Training data (raw features + target).
        X_test, y_test: Holdout test data.
        use_feature_engineering: Whether to include engineered features.
        cv_folds: Number of CV folds.

    Returns:
        DataFrame with comparison results for all models.
    """
    models = get_regression_models()
    config = get_model_config()
    seed = get_random_seed()
    results = []

    for name, model in models.items():
        logger.info("Training regression model: %s", name)
        start_time = time.time()

        pipeline = build_pipeline(model, use_feature_engineering=use_feature_engineering)

        # Get hyperparameter grid from config
        model_config = config.get("regression_models", {}).get(name, {})
        param_grid = model_config.get("params", {})

        if param_grid:
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_folds,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_
            cv_mae = -grid.best_score_
        else:
            # No tuning needed (e.g. DummyRegressor)
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}

            # Manual cross-validation for models without param grid
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="neg_mean_absolute_error",
                return_train_score=False,
            )
            cv_mae = -cv_results["test_score"].mean()

        train_time = time.time() - start_time

        # Evaluate on test set
        from src.models.evaluate import compute_regression_metrics

        y_pred = best_pipeline.predict(X_test)
        test_metrics = compute_regression_metrics(y_test, y_pred)

        # Cross-validation with multiple metrics for detailed reporting
        cv_multi = cross_validate(
            best_pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            return_train_score=False,
        )

        result = {
            "Model": name,
            "CV_MAE": round(-cv_multi["test_mae"].mean(), 4),
            "CV_MAE_std": round(cv_multi["test_mae"].std(), 4),
            "CV_RMSE": round(-cv_multi["test_rmse"].mean(), 4),
            "CV_RMSE_std": round(cv_multi["test_rmse"].std(), 4),
            "CV_R2": round(cv_multi["test_r2"].mean(), 4),
            "CV_R2_std": round(cv_multi["test_r2"].std(), 4),
            "Test_MAE": test_metrics["mae"],
            "Test_RMSE": test_metrics["rmse"],
            "Test_R2": test_metrics["r2"],
            "Train_Time_s": round(train_time, 2),
            "Best_Params": str(best_params),
            "Pipeline": best_pipeline,
        }
        results.append(result)
        logger.info(
            "  %s: CV MAE=%.4f, Test MAE=%.4f, Test R²=%.4f",
            name,
            result["CV_MAE"],
            result["Test_MAE"],
            result["Test_R2"],
        )

    results_df = pd.DataFrame(results).sort_values("CV_MAE", ascending=True)
    return results_df


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_feature_engineering: bool = True,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Train and compare all classification models.

    Args:
        X_train, y_train: Training data (raw features + At_Risk target).
        X_test, y_test: Holdout test data.
        use_feature_engineering: Whether to include engineered features.
        cv_folds: Number of CV folds.

    Returns:
        DataFrame with comparison results for all models.
    """
    models = get_classification_models()
    config = get_model_config()
    seed = get_random_seed()
    results = []

    for name, model in models.items():
        logger.info("Training classification model: %s", name)
        start_time = time.time()

        pipeline = build_pipeline(model, use_feature_engineering=use_feature_engineering)

        model_config = config.get("classification_models", {}).get(name, {})
        param_grid = model_config.get("params", {})

        if param_grid:
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
                scoring="average_precision",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}

        train_time = time.time() - start_time

        # Evaluate on test set
        from src.models.evaluate import compute_classification_metrics

        y_pred = best_pipeline.predict(X_test)
        y_prob = None
        if hasattr(best_pipeline, "predict_proba"):
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        test_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        # Cross-validation
        cv_scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
            "pr_auc": "average_precision",
        }
        cv_multi = cross_validate(
            best_pipeline,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
            scoring=cv_scoring,
            return_train_score=False,
        )

        result = {
            "Model": name,
            "CV_PR_AUC": round(cv_multi["test_pr_auc"].mean(), 4),
            "CV_PR_AUC_std": round(cv_multi["test_pr_auc"].std(), 4),
            "CV_F1": round(cv_multi["test_f1"].mean(), 4),
            "CV_Recall": round(cv_multi["test_recall"].mean(), 4),
            "CV_Precision": round(cv_multi["test_precision"].mean(), 4),
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Precision": test_metrics["precision"],
            "Test_Recall": test_metrics["recall"],
            "Test_F1": test_metrics["f1"],
            "Test_ROC_AUC": test_metrics.get("roc_auc"),
            "Test_PR_AUC": test_metrics.get("pr_auc"),
            "Train_Time_s": round(train_time, 2),
            "Best_Params": str(best_params),
            "Pipeline": best_pipeline,
        }
        results.append(result)
        logger.info(
            "  %s: CV PR-AUC=%.4f, Test Recall=%.4f, Test F1=%.4f",
            name,
            result["CV_PR_AUC"],
            result["Test_Recall"],
            result["Test_F1"],
        )

    results_df = pd.DataFrame(results).sort_values("CV_PR_AUC", ascending=False)
    return results_df


def save_model(pipeline: Any, filename: str) -> Path:
    """Save a fitted pipeline to the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(pipeline, path)
    logger.info("Model saved: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_model(filename: str) -> Any:
    """Load a fitted pipeline from the models directory."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_metrics(metrics: dict, filename: str = "metrics.json") -> Path:
    """Save metrics dictionary as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved: %s", path)
    return path


def save_model_metadata(
    model_name: str,
    version: str,
    feature_set: str,
    metrics: dict,
) -> Path:
    """Save model metadata for reproducibility tracking."""
    metadata = {
        "model_name": model_name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": get_random_seed(),
        "risk_threshold": get_risk_threshold(),
        "feature_set": feature_set,
        "metrics_summary": metrics,
    }
    return save_metrics(metadata, "model_metadata.json")
