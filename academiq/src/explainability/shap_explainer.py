"""AcademIQ — SHAP Explainability module.

Provides global and local SHAP explanations for tree-based and linear models.
Uses TreeExplainer for tree models (exact, fast) and LinearExplainer for
linear models.
"""

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


class ShapExplainer:
    """Wrapper for SHAP explanations.

    Handles the complexity of extracting the model from a Pipeline and
    creating the appropriate SHAP explainer.
    """

    def __init__(self, pipeline: Any, X_background: pd.DataFrame):
        """Initialise the SHAP explainer.

        Args:
            pipeline: Fitted sklearn Pipeline.
            X_background: Background dataset (raw features) for SHAP.
                Should be a representative sample (100-200 rows).
        """
        self.pipeline = pipeline
        self.feature_names = self._get_feature_names(pipeline, X_background)

        # Transform the background data through all preprocessing steps
        self.X_background_transformed = self._transform_features(X_background)

        # Extract the final model from the pipeline
        self.model = pipeline.named_steps["model"]

        # Create the appropriate explainer
        self.explainer = self._create_explainer()

    def _get_feature_names(self, pipeline: Any, X: pd.DataFrame) -> list[str]:
        """Get the feature names after all transformations."""
        try:
            # Walk through the pipeline steps (excluding model) and collect names
            preprocessor = pipeline.named_steps.get("preprocessor")
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                names = list(preprocessor.get_feature_names_out())
                # If there's a scaler step, names pass through unchanged
                # If there's a feature_engineer step, append engineered names
                fe = pipeline.named_steps.get("feature_engineer")
                if fe is not None:
                    names = list(fe.get_feature_names_out(names))
                return names

            # Fallback: try get_feature_names_out from the last pre-model step
            pipeline_without_model = pipeline[:-1]
            last_step = pipeline_without_model[-1]
            if hasattr(last_step, "get_feature_names_out"):
                return list(last_step.get_feature_names_out())
        except Exception:
            pass
        return [f"feature_{i}" for i in range(self.X_background_transformed.shape[1])]

    def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform raw features through all pipeline steps except the model."""
        pipeline_without_model = self.pipeline[:-1]
        X_transformed = pipeline_without_model.transform(X)
        if isinstance(X_transformed, pd.DataFrame):
            return X_transformed.values
        return X_transformed

    def _create_explainer(self) -> shap.Explainer:
        """Create the appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__

        if hasattr(self.model, "estimators_") or "Gradient" in model_type or "Forest" in model_type:
            logger.info("Using TreeExplainer for %s", model_type)
            return shap.TreeExplainer(self.model)
        elif hasattr(self.model, "coef_"):
            logger.info("Using LinearExplainer for %s", model_type)
            return shap.LinearExplainer(self.model, self.X_background_transformed)
        else:
            logger.info("Using KernelExplainer for %s (may be slow)", model_type)
            return shap.KernelExplainer(
                self.model.predict, self.X_background_transformed[:50]
            )

    def global_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """Compute SHAP values for a dataset (global explanation).

        Args:
            X: Raw features (before preprocessing).

        Returns:
            shap.Explanation object with values and feature names.
        """
        X_transformed = self._transform_features(X)
        shap_values = self.explainer.shap_values(X_transformed)

        return shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value
            if hasattr(self.explainer, "expected_value")
            else np.mean(shap_values, axis=0),
            data=X_transformed,
            feature_names=self.feature_names,
        )

    def local_shap_values(self, X_single: pd.DataFrame) -> dict:
        """Compute SHAP values for a single student.

        Args:
            X_single: Single-row DataFrame with raw features.

        Returns:
            Dict with shap_values, feature_names, base_value,
            top_positive, top_negative.
        """
        X_transformed = self._transform_features(X_single)
        shap_values = self.explainer.shap_values(X_transformed)

        if shap_values.ndim > 1:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        base_value = (
            self.explainer.expected_value
            if isinstance(self.explainer.expected_value, (int, float))
            else self.explainer.expected_value[0]
            if hasattr(self.explainer.expected_value, "__len__")
            else float(np.mean(shap_vals))
        )

        # Create sorted feature contributions
        contributions = sorted(
            zip(self.feature_names, shap_vals, X_transformed[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_positive = [
            {"feature": f, "shap_value": round(float(v), 4), "feature_value": round(float(fv), 4)}
            for f, v, fv in contributions
            if v > 0
        ][:5]

        top_negative = [
            {"feature": f, "shap_value": round(float(v), 4), "feature_value": round(float(fv), 4)}
            for f, v, fv in contributions
            if v < 0
        ][:5]

        return {
            "shap_values": dict(zip(self.feature_names, [round(float(v), 4) for v in shap_vals])),
            "base_value": round(float(base_value), 4),
            "top_positive_contributors": top_positive,
            "top_negative_contributors": top_negative,
        }

    def mean_abs_shap(self, X: pd.DataFrame) -> pd.Series:
        """Compute mean absolute SHAP values (global feature importance).

        Args:
            X: Raw features DataFrame.

        Returns:
            Series with feature names as index, sorted descending.
        """
        X_transformed = self._transform_features(X)
        shap_values = self.explainer.shap_values(X_transformed)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.Series(mean_abs, index=self.feature_names)
        return importance.sort_values(ascending=False)


def save_explainer_background(X_background: pd.DataFrame, filename: str = "shap_background.joblib"):
    """Save background data for SHAP (used by dashboard)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(X_background, path)
    logger.info("SHAP background saved: %s", path)
    return path
