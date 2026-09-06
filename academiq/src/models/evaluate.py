"""AcademIQ — Model evaluation metrics.

Computes regression and classification metrics.
All functions return plain dicts suitable for JSON serialisation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Compute regression evaluation metrics.

    Returns:
        Dict with mae, rmse, r2, max_error, median_ae.
    """
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "max_error": round(float(np.max(np.abs(y_true - y_pred))), 4),
        "median_ae": round(float(np.median(np.abs(y_true - y_pred))), 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute classification evaluation metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dict with accuracy, precision, recall, f1, and optionally roc_auc,
        pr_auc, brier_score.
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = round(
                float(average_precision_score(y_true, y_prob)), 4
            )
        except ValueError:
            metrics["pr_auc"] = None
        try:
            metrics["brier_score"] = round(
                float(brier_score_loss(y_true, y_prob)), 4
            )
        except ValueError:
            metrics["brier_score"] = None

    return metrics


def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Analyse precision/recall/F1 across probability thresholds.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        thresholds: Optional array of thresholds to evaluate.
            If None, uses [0.1, 0.2, ..., 0.9].

    Returns:
        List of dicts with threshold, precision, recall, f1, fp, fn counts.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    results = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "threshold": round(float(t), 2),
            "precision": round(float(precision_score(y_true, y_pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred_t, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred_t, zero_division=0)), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        })

    return results


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    task: str = "regression",
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute metrics for each subgroup.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        groups: Array of group labels (same length as y_true).
        task: "regression" or "classification".
        y_prob: Predicted probabilities (for classification).

    Returns:
        Dict mapping group name to metric dict.
    """
    unique_groups = np.unique(groups)
    results = {}

    for group in unique_groups:
        mask = groups == group
        n = int(mask.sum())

        if task == "regression":
            metrics = compute_regression_metrics(y_true[mask], y_pred[mask])
        else:
            metrics = compute_classification_metrics(
                y_true[mask],
                y_pred[mask],
                y_prob[mask] if y_prob is not None else None,
            )

        metrics["n"] = n
        results[str(group)] = metrics

    return results
