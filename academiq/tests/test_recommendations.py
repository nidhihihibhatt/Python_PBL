"""Tests for recommendation engine."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommendations.engine import evaluate_condition, generate_recommendations


class TestConditionEvaluation:
    """Tests for rule condition evaluation."""

    def test_less_than_true(self):
        assert evaluate_condition(10, "<", 15) is True

    def test_less_than_false(self):
        assert evaluate_condition(20, "<", 15) is False

    def test_equals_true(self):
        assert evaluate_condition(0, "==", 0) is True

    def test_greater_than(self):
        assert evaluate_condition(30, ">", 25) is True

    def test_invalid_operator(self):
        assert evaluate_condition(10, "LIKE", 10) is False


class TestRecommendationEngine:
    """Tests for recommendation generation."""

    def test_recommendations_trigger_correctly(self, sample_shap_values):
        features = {
            "Hours_Studied": 10,
            "Attendance": 60,
            "Previous_Scores": 55,
            "Tutoring_Sessions": 0,
            "Parental_Involvement": 0,
            "Access_to_Resources": 0,
            "Motivation_Level": 0,
            "Sleep_Hours": 5,
        }
        recs = generate_recommendations(features, sample_shap_values)
        assert len(recs) > 0
        assert all("rule_id" in r for r in recs)
        assert all("message" in r for r in recs)

    def test_no_trigger_when_shap_disagrees(self):
        features = {"Hours_Studied": 10, "Attendance": 60}
        # SHAP says these features help (positive) — rule should NOT trigger
        shap_vals = {"Hours_Studied": 0.5, "Attendance": 0.8}
        recs = generate_recommendations(features, shap_vals)
        # Rules expect negative SHAP, so none should trigger
        assert len(recs) == 0

    def test_priority_sorting(self, sample_shap_values):
        features = {
            "Hours_Studied": 10,
            "Attendance": 60,
            "Motivation_Level": 0,
            "Sleep_Hours": 5,
        }
        recs = generate_recommendations(features, sample_shap_values)
        if len(recs) >= 2:
            priorities = [r["priority"] for r in recs]
            priority_order = {"high": 0, "medium": 1, "low": 2}
            numeric = [priority_order.get(p, 99) for p in priorities]
            assert numeric == sorted(numeric)

    def test_empty_features_returns_empty(self):
        recs = generate_recommendations({}, {})
        assert recs == []

    def test_max_recommendations_limit(self, sample_shap_values):
        features = {
            "Hours_Studied": 10,
            "Attendance": 60,
            "Previous_Scores": 55,
            "Tutoring_Sessions": 0,
            "Parental_Involvement": 0,
            "Access_to_Resources": 0,
            "Motivation_Level": 0,
            "Sleep_Hours": 5,
        }
        recs = generate_recommendations(features, sample_shap_values, max_recommendations=2)
        assert len(recs) <= 2

    def test_recommendation_has_reason(self, sample_shap_values):
        features = {"Attendance": 60}
        recs = generate_recommendations(features, sample_shap_values)
        for r in recs:
            assert "reason" in r
            assert "Rule" in r["reason"]

    def test_all_good_student_no_recommendations(self):
        features = {
            "Hours_Studied": 30,
            "Attendance": 95,
            "Previous_Scores": 85,
            "Tutoring_Sessions": 5,
            "Parental_Involvement": 2,
            "Access_to_Resources": 2,
            "Motivation_Level": 2,
            "Sleep_Hours": 8,
        }
        # Positive SHAP values (all features help)
        shap_vals = {k: 0.5 for k in features}
        recs = generate_recommendations(features, shap_vals)
        assert len(recs) == 0
