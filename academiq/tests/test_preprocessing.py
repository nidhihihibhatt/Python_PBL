"""Tests for preprocessing pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.pipeline import build_pipeline, build_preprocessor
from sklearn.dummy import DummyRegressor


class TestPreprocessor:
    """Tests for the ColumnTransformer preprocessor."""

    def test_preprocessor_fits_and_transforms(self, sample_dataframe):
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(sample_dataframe)
        assert result is not None
        assert result.shape[0] == len(sample_dataframe)

    def test_preprocessor_produces_no_nans(self, sample_dataframe):
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(sample_dataframe)
        assert not np.any(np.isnan(result))

    def test_preprocessor_consistent_output(self, sample_dataframe):
        preprocessor = build_preprocessor()
        preprocessor.fit(sample_dataframe)
        result1 = preprocessor.transform(sample_dataframe)
        result2 = preprocessor.transform(sample_dataframe)
        np.testing.assert_array_equal(result1, result2)


class TestPipeline:
    """Tests for the full Pipeline (preprocessor + model)."""

    def test_pipeline_builds_successfully(self):
        model = DummyRegressor()
        pipeline = build_pipeline(model)
        assert pipeline is not None
        assert "preprocessor" in dict(pipeline.steps)
        assert "model" in dict(pipeline.steps)

    def test_pipeline_fits_and_predicts(self, sample_dataframe):
        model = DummyRegressor()
        pipeline = build_pipeline(model)
        y = np.array([65, 67, 70, 68, 72])
        pipeline.fit(sample_dataframe, y)
        predictions = pipeline.predict(sample_dataframe)
        assert len(predictions) == len(sample_dataframe)

    def test_pipeline_with_feature_engineering(self, sample_dataframe):
        model = DummyRegressor()
        pipeline = build_pipeline(model, use_feature_engineering=True)
        y = np.array([65, 67, 70, 68, 72])
        pipeline.fit(sample_dataframe, y)
        predictions = pipeline.predict(sample_dataframe)
        assert len(predictions) == len(sample_dataframe)

    def test_pipeline_without_feature_engineering(self, sample_dataframe):
        model = DummyRegressor()
        pipeline = build_pipeline(model, use_feature_engineering=False)
        y = np.array([65, 67, 70, 68, 72])
        pipeline.fit(sample_dataframe, y)
        predictions = pipeline.predict(sample_dataframe)
        assert len(predictions) == len(sample_dataframe)

    def test_pipeline_single_row_prediction(self, sample_student):
        model = DummyRegressor()
        pipeline = build_pipeline(model)
        X_train = pd.DataFrame([sample_student] * 10)
        y_train = np.array([65, 67, 70, 68, 72, 63, 66, 69, 71, 74])
        pipeline.fit(X_train, y_train)

        X_single = pd.DataFrame([sample_student])
        predictions = pipeline.predict(X_single)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.integer, np.floating))

    def test_tree_model_gets_no_scaler(self):
        from sklearn.ensemble import RandomForestRegressor
        pipeline = build_pipeline(RandomForestRegressor())
        step_names = [name for name, _ in pipeline.steps]
        assert "scaler" not in step_names

    def test_linear_model_gets_scaler(self):
        from sklearn.linear_model import Ridge
        pipeline = build_pipeline(Ridge())
        step_names = [name for name, _ in pipeline.steps]
        assert "scaler" in step_names
