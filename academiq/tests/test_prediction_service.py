"""End-to-end tests using trained models.

These tests require models to be trained first (run train.py).
They verify the complete prediction pipeline from raw input to output.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR

# Skip all tests if models not trained
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "regression_pipeline.joblib").exists(),
    reason="Models not trained. Run 'python train.py' first.",
)


class TestPredictionService:
    """End-to-end prediction service tests."""

    def test_service_loads(self):
        from src.inference.service import PredictionService
        service = PredictionService()
        assert service.reg_pipeline is not None

    def test_predict_returns_score(self, sample_student):
        from src.inference.service import PredictionService
        service = PredictionService()
        result = service.predict(sample_student)
        assert "predicted_score" in result
        assert isinstance(result["predicted_score"], float)
        assert 50 <= result["predicted_score"] <= 110

    def test_predict_returns_risk(self, sample_student):
        from src.inference.service import PredictionService
        service = PredictionService()
        result = service.predict(sample_student)
        assert "risk_category" in result
        assert result["risk_category"] in ["High Risk", "Moderate Risk", "Low Risk"]

    def test_predict_returns_interval(self, sample_student):
        from src.inference.service import PredictionService
        service = PredictionService()
        result = service.predict(sample_student)
        interval = result.get("prediction_interval")
        if interval is not None:
            assert interval["lower"] <= interval["upper"]

    def test_predict_returns_shap(self, sample_student):
        from src.inference.service import PredictionService
        service = PredictionService()
        result = service.predict(sample_student)
        shap_exp = result.get("shap_explanation")
        if shap_exp is not None:
            assert "top_positive_contributors" in shap_exp
            assert "top_negative_contributors" in shap_exp

    def test_batch_predict(self, sample_dataframe):
        from src.inference.service import PredictionService
        service = PredictionService()
        preds = service.predict_batch(sample_dataframe)
        assert len(preds) == len(sample_dataframe)
        assert "predicted_score" in preds.columns
        assert "risk_category" in preds.columns

    def test_invalid_input_handled(self):
        from src.inference.service import PredictionService
        service = PredictionService()
        result = service.predict({"Hours_Studied": 20})  # Missing most features
        assert "error" in result

    def test_at_risk_student_gets_high_risk(self):
        from src.inference.service import PredictionService
        service = PredictionService()
        # Create a student likely to be at risk
        student = {
            "Hours_Studied": 1,
            "Attendance": 20,
            "Parental_Involvement": "Low",
            "Access_to_Resources": "Low",
            "Extracurricular_Activities": "No",
            "Sleep_Hours": 4,
            "Previous_Scores": 40,
            "Motivation_Level": "Low",
            "Internet_Access": "No",
            "Tutoring_Sessions": 0,
            "Family_Income": "Low",
            "Teacher_Quality": "Low",
            "School_Type": "Public",
            "Peer_Influence": "Negative",
            "Physical_Activity": 0,
            "Learning_Disabilities": "Yes",
            "Parental_Education_Level": "High School",
            "Distance_from_Home": "Far",
            "Gender": "Male",
        }
        result = service.predict(student)
        # Should have a lower score and higher risk
        assert result["predicted_score"] < 70
        assert result["risk_category"] in ["High Risk", "Moderate Risk"]
