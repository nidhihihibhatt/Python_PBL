"""AcademIQ — Prediction service.

Single entry point for inference. Accepts raw student features and returns
a complete prediction report: predicted score, prediction interval,
risk probability, risk category, SHAP explanations, and recommendations.

All preprocessing is handled by the serialised Pipeline — no manual
encoding, scaling, or feature engineering at inference time.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, get_model_config, get_risk_threshold, REPORTS_DIR
from src.data.loader import validate_data
from src.explainability.shap_explainer import ShapExplainer
from src.recommendations.engine import generate_recommendations

logger = logging.getLogger(__name__)


class PredictionService:
    """End-to-end prediction service.

    Usage:
        service = PredictionService()
        result = service.predict(student_features_dict)
    """

    def __init__(self, models_dir: Path | None = None):
        """Load all model artifacts.

        Args:
            models_dir: Optional override for models directory.
        """
        self.models_dir = models_dir or MODELS_DIR
        self.config = get_model_config()
        
        # Load dynamic threshold from metadata, fallback to config
        self.risk_threshold = get_risk_threshold()
        try:
            meta_path = REPORTS_DIR / "model_metadata.json"
            if meta_path.exists():
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.risk_threshold = meta.get("metrics", {}).get("classification_threshold", self.risk_threshold)
        except Exception:
            pass

        # Load pipelines
        self.reg_pipeline = self._load("regression_pipeline.joblib")
        self.cls_pipeline = self._load_optional("classification_pipeline.joblib")
        self.quantile_lo = self._load_optional("quantile_lo_pipeline.joblib")
        self.quantile_hi = self._load_optional("quantile_hi_pipeline.joblib")

        # Load SHAP background
        self.shap_background = self._load_optional("shap_background.joblib")
        self._shap_explainer = None

    def _load(self, filename: str) -> Any:
        """Load a required model artifact."""
        path = self.models_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}. "
                f"Run 'python train.py' first."
            )
        return joblib.load(path)

    def _load_optional(self, filename: str) -> Any | None:
        """Load an optional model artifact."""
        path = self.models_dir / filename
        if path.exists():
            return joblib.load(path)
        logger.info("Optional artifact not found: %s", path)
        return None

    @property
    def shap_explainer(self) -> ShapExplainer | None:
        """Lazy-initialise SHAP explainer."""
        if self._shap_explainer is None and self.shap_background is not None:
            try:
                self._shap_explainer = ShapExplainer(
                    self.reg_pipeline, self.shap_background
                )
            except Exception as e:
                logger.warning("Failed to initialise SHAP explainer: %s", e)
        return self._shap_explainer

    def predict(self, features: dict | pd.DataFrame) -> dict:
        """Generate a complete prediction report for a student.

        Args:
            features: Student features as a dict or single-row DataFrame.

        Returns:
            Dict with predicted_score, prediction_interval, risk_probability,
            risk_category, shap_explanation, recommendations.
        """
        # Convert dict to DataFrame
        if isinstance(features, dict):
            X = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            X = pd.DataFrame([features])
        else:
            X = features.copy()

        # --- Validate input ---
        report = validate_data(X, require_target=False)
        if not report.is_valid:
            errors = [i.message for i in report.issues if i.severity == "error"]
            return {"error": f"Input validation failed: {'; '.join(errors)}"}

        # --- Regression prediction ---
        predicted_score = float(self.reg_pipeline.predict(X)[0])

        # --- Prediction interval (if quantile models available) ---
        interval = None
        if self.quantile_lo is not None and self.quantile_hi is not None:
            lo = float(self.quantile_lo.predict(X)[0])
            hi = float(self.quantile_hi.predict(X)[0])
            interval = {
                "lower": round(lo, 1),
                "upper": round(hi, 1),
                "coverage": "80%",
                "method": "quantile_regression",
            }

        # --- Risk assessment ---
        risk_prob = None
        risk_category = None

        if self.cls_pipeline is not None:
            try:
                risk_prob = float(self.cls_pipeline.predict_proba(X)[0, 1])
            except Exception:
                # Fall back to regression-derived risk
                risk_prob = None

        if risk_prob is None:
            # Regression-derived risk: simple threshold
            risk_prob = 1.0 if predicted_score < self.risk_threshold else 0.0

        # Determine risk category from probability
        cat_config = self.config.get("risk_categories", {})
        high_min = cat_config.get("high_risk_min_prob", 0.7)
        mod_min = cat_config.get("moderate_risk_min_prob", 0.3)

        if risk_prob >= high_min:
            risk_category = "High Risk"
        elif risk_prob >= mod_min:
            risk_category = "Moderate Risk"
        else:
            risk_category = "Low Risk"

        # --- SHAP explanation ---
        shap_explanation = None
        shap_values_dict = {}

        if self.shap_explainer is not None:
            try:
                shap_explanation = self.shap_explainer.local_shap_values(X)
                shap_values_dict = shap_explanation.get("shap_values", {})
            except Exception as e:
                logger.warning("SHAP explanation failed: %s", e)

        # --- Recommendations ---
        # Build feature dict from the input
        feature_dict = X.iloc[0].to_dict()
        recommendations = generate_recommendations(
            features=feature_dict,
            shap_values=shap_values_dict,
        )

        return {
            "predicted_score": round(predicted_score, 1),
            "prediction_interval": interval,
            "risk_probability": round(risk_prob, 3) if risk_prob is not None else None,
            "risk_category": risk_category,
            "shap_explanation": shap_explanation,
            "recommendations": recommendations,
            "input_features": feature_dict,
        }

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for multiple students.

        Args:
            X: DataFrame with student features (one row per student).

        Returns:
            DataFrame with predicted_score, risk_probability, risk_category
            for each student.
        """
        scores = self.reg_pipeline.predict(X)
        result = pd.DataFrame({
            "predicted_score": np.round(scores, 1),
        })

        if self.cls_pipeline is not None:
            try:
                probs = self.cls_pipeline.predict_proba(X)[:, 1]
                result["risk_probability"] = np.round(probs, 3)
            except Exception:
                result["risk_probability"] = (scores < self.risk_threshold).astype(float)
        else:
            result["risk_probability"] = (scores < self.risk_threshold).astype(float)

        # Risk categories
        cat_config = self.config.get("risk_categories", {})
        high_min = cat_config.get("high_risk_min_prob", 0.7)
        mod_min = cat_config.get("moderate_risk_min_prob", 0.3)

        def categorise(p):
            if p >= high_min:
                return "High Risk"
            elif p >= mod_min:
                return "Moderate Risk"
            return "Low Risk"

        result["risk_category"] = result["risk_probability"].apply(categorise)

        # Prediction intervals
        if self.quantile_lo is not None and self.quantile_hi is not None:
            result["interval_lower"] = np.round(self.quantile_lo.predict(X), 1)
            result["interval_upper"] = np.round(self.quantile_hi.predict(X), 1)

        return result
