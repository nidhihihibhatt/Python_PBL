"""Tests for data validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_data, validate_data


class TestDataLoader:
    """Tests for the data loading function."""

    def test_load_data_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_data_has_expected_shape(self):
        df = load_data()
        assert df.shape == (6607, 20)

    def test_load_data_has_target_column(self):
        df = load_data()
        assert "Exam_Score" in df.columns


class TestDataValidator:
    """Tests for the data validation function."""

    def test_valid_data_passes(self):
        df = load_data()
        report = validate_data(df)
        assert report.is_valid is True

    def test_missing_column_detected(self, sample_dataframe_with_target):
        df = sample_dataframe_with_target.drop(columns=["Attendance"])
        report = validate_data(df)
        assert report.is_valid is False
        assert any("Attendance" in i.message for i in report.issues)

    def test_invalid_category_detected(self, sample_dataframe_with_target):
        df = sample_dataframe_with_target.copy()
        df.loc[0, "Gender"] = "Unknown"
        report = validate_data(df)
        assert any(i.category == "invalid_category" for i in report.issues)

    def test_empty_dataframe_detected(self):
        df = pd.DataFrame(columns=["Exam_Score", "Hours_Studied", "Attendance"])
        report = validate_data(df)
        # Will fail due to missing columns or empty data
        assert report.is_valid is False

    def test_missing_values_reported(self, sample_dataframe_with_target):
        df = sample_dataframe_with_target.copy()
        df.loc[0, "Teacher_Quality"] = None
        report = validate_data(df)
        assert any(i.category == "missing_values" for i in report.issues)

    def test_validation_without_target(self, sample_dataframe):
        report = validate_data(sample_dataframe, require_target=False)
        assert report.is_valid is True

    def test_report_summary_is_string(self):
        df = load_data()
        report = validate_data(df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Validation" in summary
