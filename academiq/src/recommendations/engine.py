"""AcademIQ — Rule-based recommendation engine.

Generates transparent, traceable recommendations based on:
  1. Student feature values
  2. SHAP contributions (model-specific)
  3. Explicit rules defined in YAML configuration

Each recommendation triggers ONLY when:
  - The feature condition is met, AND
  - The SHAP value confirms the feature negatively impacts the prediction.

This prevents generic advice disconnected from the model's assessment.
"""

import logging
import operator
from typing import Any

from src.config import get_recommendation_rules

logger = logging.getLogger(__name__)

# Operator mapping for condition evaluation
OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

# Priority ordering
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def evaluate_condition(value: Any, op: str, threshold: Any) -> bool:
    """Evaluate a single condition.

    Args:
        value: Feature value.
        op: Operator string (e.g. "<", ">=", "==").
        threshold: Threshold to compare against.

    Returns:
        True if condition is met.
    """
    try:
        op_func = OPERATORS.get(op)
        if op_func is None:
            logger.warning("Unknown operator: %s", op)
            return False
        return bool(op_func(float(value), float(threshold)))
    except (ValueError, TypeError):
        return False


def generate_recommendations(
    features: dict[str, Any],
    shap_values: dict[str, float],
    rules: list[dict] | None = None,
    max_recommendations: int = 5,
) -> list[dict]:
    """Generate recommendations for a single student.

    Args:
        features: Dict of feature name → value (raw or encoded).
        shap_values: Dict of feature name → SHAP value.
        rules: Optional override for recommendation rules.
            If None, loads from config.
        max_recommendations: Maximum number of recommendations to return.

    Returns:
        List of triggered recommendation dicts, sorted by priority.
        Each dict contains: rule_id, feature, value, shap_value,
        priority, category, message, reason.
    """
    if rules is None:
        rules = get_recommendation_rules()

    triggered = []

    for rule in rules:
        feature = rule["feature"]
        rule_id = rule["id"]

        # Get feature value
        value = features.get(feature)
        if value is None:
            continue

        # Check condition
        condition_met = evaluate_condition(
            value, rule["operator"], rule["threshold"]
        )
        if not condition_met:
            continue

        # Check SHAP direction
        shap_val = shap_values.get(feature, 0.0)
        expected_dir = rule.get("expected_shap_direction", "negative")

        shap_confirms = False
        if expected_dir == "negative" and shap_val < 0:
            shap_confirms = True
        elif expected_dir == "positive" and shap_val > 0:
            shap_confirms = True

        if not shap_confirms:
            continue

        # Rule triggered
        triggered.append({
            "rule_id": rule_id,
            "feature": feature,
            "feature_value": value,
            "shap_value": round(float(shap_val), 4),
            "priority": rule["priority"],
            "category": rule["category"],
            "message": rule["message"].strip(),
            "reason": (
                f"Rule {rule_id} triggered: {feature}={value} "
                f"({rule['operator']} {rule['threshold']}) and SHAP "
                f"confirms negative contribution ({shap_val:.4f})"
            ),
        })

    # Sort by priority, then by absolute SHAP value (most impactful first)
    triggered.sort(
        key=lambda r: (
            PRIORITY_ORDER.get(r["priority"], 99),
            -abs(r["shap_value"]),
        )
    )

    return triggered[:max_recommendations]
