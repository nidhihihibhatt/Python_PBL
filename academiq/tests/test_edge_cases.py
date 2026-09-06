"""Tests for edge cases and error handling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import validate_data
from src.preprocessing.pipeline import build_pipeline
from sklearn.dummy import DummyRegressor


class TestEdgeCases:
    """Tests for boundary values and unusual inputs."""

    def test_all_features_at_minimum(self):
        student = {
            "Hours_Studied": 0,
            "Attendance": 0,
            "Parental_Involvement": "Low",
            "Access_to_Resources": "Low",
            "Extracurricular_Activities": "No",
            "Sleep_Hours": 0,
            "Previous_Scores": 0,
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
        X = pd.DataFrame([student] * 5)
        y = np.array([60, 61, 62, 63, 64])
        pipeline = build_pipeline(DummyRegressor())
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == 5
        assert all(np.isfinite(preds))

    def test_all_features_at_maximum(self):
        student = {
            "Hours_Studied": 44,
            "Attendance": 100,
            "Parental_Involvement": "High",
            "Access_to_Resources": "High",
            "Extracurricular_Activities": "Yes",
            "Sleep_Hours": 12,
            "Previous_Scores": 100,
            "Motivation_Level": "High",
            "Internet_Access": "Yes",
            "Tutoring_Sessions": 8,
            "Family_Income": "High",
            "Teacher_Quality": "High",
            "School_Type": "Private",
            "Peer_Influence": "Positive",
            "Physical_Activity": 6,
            "Learning_Disabilities": "No",
            "Parental_Education_Level": "Postgraduate",
            "Distance_from_Home": "Near",
            "Gender": "Female",
        }
        X = pd.DataFrame([student] * 5)
        y = np.array([80, 82, 85, 88, 90])
        pipeline = build_pipeline(DummyRegressor())
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == 5

    def test_missing_values_handled_by_pipeline(self, sample_student):
        student = sample_student.copy()
        student["Teacher_Quality"] = None
        student["Parental_Education_Level"] = None
        student["Distance_from_Home"] = None
        X = pd.DataFrame([student] * 5)
        y = np.array([65, 67, 70, 68, 72])
        pipeline = build_pipeline(DummyRegressor())
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == 5
        assert all(np.isfinite(preds))

    def test_validation_catches_wrong_column_names(self):
        df = pd.DataFrame({
            "WRONG_COL_1": [1, 2, 3],
            "WRONG_COL_2": [4, 5, 6],
            "Exam_Score": [65, 70, 75],
        })
        report = validate_data(df)
        assert report.is_valid is False

    def test_validation_catches_unknown_category(self, sample_student):
        student = sample_student.copy()
        student["Gender"] = "NonBinary"
        student["Exam_Score"] = 70
        df = pd.DataFrame([student])
        report = validate_data(df)
        error_cols = [i.column for i in report.issues if i.severity == "error"]
        assert "Gender" in error_cols

    def test_single_row_dataframe(self, sample_student):
        X = pd.DataFrame([sample_student])
        y = np.array([68])
        pipeline = build_pipeline(DummyRegressor())
        pipeline.fit(X, y)
        pred = pipeline.predict(X)
        assert len(pred) == 1
