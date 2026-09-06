"""AcademIQ — Fairness / responsible AI analysis.

Evaluates model performance across demographic subgroups to identify
potential disparities. Does NOT declare the model "fair" or "unfair" —
it reports metrics transparently.
"""

import logging

import numpy as np
import pandas as pd

from src.models.evaluate import (
    compute_classification_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)

# Internal alert threshold — NOT a legal fairness standard
DISPARITY_ALERT_THRESHOLD_PP = 10  # percentage points


def analyze_regression_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    group_name: str,
) -> dict:
    """Analyse regression performance across subgroups.

    Returns:
        Dict with overall metrics, per-group metrics, and disparity alerts.
    """
    overall = compute_regression_metrics(y_true, y_pred)

    subgroup_results = {}
    for group_val in sorted(groups.unique()):
        mask = groups == group_val
        if mask.sum() < 10:
            continue
        metrics = compute_regression_metrics(y_true[mask], y_pred[mask])
        metrics["n"] = int(mask.sum())
        # Mean residual (bias check)
        residuals = y_true[mask] - y_pred[mask]
        metrics["mean_residual"] = round(float(residuals.mean()), 4)
        subgroup_results[str(group_val)] = metrics

    # Check for MAE disparities
    maes = {k: v["mae"] for k, v in subgroup_results.items()}
    alerts = []
    if maes:
        max_mae = max(maes.values())
        min_mae = min(maes.values())
        if max_mae - min_mae > 0.5:  # More than 0.5 score points difference
            alerts.append(
                f"MAE varies by {max_mae - min_mae:.2f} points across "
                f"{group_name} subgroups (max: {max_mae:.2f}, min: {min_mae:.2f})"
            )

    return {
        "group_name": group_name,
        "overall": overall,
        "subgroups": subgroup_results,
        "alerts": alerts,
    }


def analyze_classification_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    groups: pd.Series,
    group_name: str,
) -> dict:
    """Analyse classification performance across subgroups.

    Returns:
        Dict with overall metrics, per-group metrics, and disparity alerts.
    """
    overall = compute_classification_metrics(y_true, y_pred, y_prob)

    subgroup_results = {}
    for group_val in sorted(groups.unique()):
        mask = groups == group_val
        if mask.sum() < 10:
            continue
        metrics = compute_classification_metrics(
            y_true[mask],
            y_pred[mask],
            y_prob[mask] if y_prob is not None else None,
        )
        metrics["n"] = int(mask.sum())

        # False positive/negative rates
        cm = np.array(metrics["confusion_matrix"])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["fpr"] = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 4)
            metrics["fnr"] = round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0, 4)

        subgroup_results[str(group_val)] = metrics

    # Check for recall disparities
    recalls = {k: v["recall"] for k, v in subgroup_results.items()}
    alerts = []
    if recalls:
        max_recall = max(recalls.values())
        min_recall = min(recalls.values())
        diff_pp = (max_recall - min_recall) * 100
        if diff_pp > DISPARITY_ALERT_THRESHOLD_PP:
            alerts.append(
                f"⚠️ Recall varies by {diff_pp:.1f} percentage points across "
                f"{group_name} subgroups. This exceeds the internal alert "
                f"threshold of {DISPARITY_ALERT_THRESHOLD_PP}pp. "
                f"(Note: this threshold is a project-defined internal alert, "
                f"not a universal fairness standard.)"
            )

    return {
        "group_name": group_name,
        "overall": overall,
        "subgroups": subgroup_results,
        "alerts": alerts,
    }


def full_fairness_report(
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_true_cls: np.ndarray | None,
    y_pred_cls: np.ndarray | None,
    y_prob_cls: np.ndarray | None,
    df: pd.DataFrame,
    protected_attributes: list[str] = None,
) -> dict:
    """Generate a complete fairness analysis report.

    Args:
        y_true_reg: True regression values.
        y_pred_reg: Predicted regression values.
        y_true_cls: True classification labels (optional).
        y_pred_cls: Predicted classification labels (optional).
        y_prob_cls: Predicted classification probabilities (optional).
        df: Original DataFrame containing the protected attributes.
        protected_attributes: List of column names to analyse.

    Returns:
        Dict with regression and classification fairness results.
    """
    if protected_attributes is None:
        protected_attributes = ["Gender", "Family_Income"]

    report = {"regression": {}, "classification": {}}

    for attr in protected_attributes:
        if attr not in df.columns:
            logger.warning("Protected attribute '%s' not found in data", attr)
            continue

        groups = df[attr]

        report["regression"][attr] = analyze_regression_fairness(
            y_true_reg, y_pred_reg, groups, attr
        )

        if y_true_cls is not None and y_pred_cls is not None:
            report["classification"][attr] = analyze_classification_fairness(
                y_true_cls, y_pred_cls, y_prob_cls, groups, attr
            )

    return report
