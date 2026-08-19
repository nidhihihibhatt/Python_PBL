"""AcademIQ — Main training script.

Runs the complete training pipeline:
  1. Load and validate data
  2. Train/test split
  3. Ablation study (with/without engineered features)
  4. Regression model comparison
  5. Classification model comparison
  6. Probability Calibration
  7. Regression-vs-classifier experiment & Dynamic Threshold Selection
  8. Train quantile regression (uncertainty tuning)
  9. SHAP explainability
  10. Fairness analysis
  11. Save all artifacts and metrics
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split
from sklearn.metrics import brier_score_loss, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODELS_DIR, REPORTS_DIR, get_model_config, get_random_seed, get_risk_threshold
from src.data.loader import load_data, validate_data
from src.explainability.shap_explainer import ShapExplainer, save_explainer_background
from src.fairness.analysis import full_fairness_report
from src.models.evaluate import compute_classification_metrics, compute_regression_metrics, threshold_analysis
from src.models.train import prepare_data, save_metrics, save_model, save_model_metadata, train_classification_models, train_regression_models
from src.preprocessing.pipeline import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log", mode="w")],
)
logger = logging.getLogger("academiq.train")


def main():
    start_time = time.time()
    seed = get_random_seed()
    np.random.seed(seed)
    all_metrics = {}

    # STEP 1: Load and validate
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and validating data")
    df = load_data()
    report = validate_data(df)
    if not report.is_valid:
        logger.error("Data validation FAILED.")
        sys.exit(1)

    # STEP 2: Prepare data
    logger.info("=" * 60)
    logger.info("STEP 2: Preparing data and splitting")
    X, y_reg, y_cls = prepare_data(df)
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=seed, stratify=y_cls
    )

    # STEP 3: Ablation study
    logger.info("=" * 60)
    logger.info("STEP 3: Ablation study — Feature engineering impact")
    ablation_model = HistGradientBoostingRegressor(random_state=seed)
    
    pipeline_a = build_pipeline(ablation_model, use_feature_engineering=False)
    cv_a = cross_validate(pipeline_a, X_train, y_train_reg, cv=5, scoring="neg_mean_absolute_error")
    mae_a = -cv_a["test_score"].mean()

    pipeline_b = build_pipeline(ablation_model, use_feature_engineering=True)
    cv_b = cross_validate(pipeline_b, X_train, y_train_reg, cv=5, scoring="neg_mean_absolute_error")
    mae_b = -cv_b["test_score"].mean()

    improvement = mae_a - mae_b
    use_engineering = improvement > 0.05

    ablation_results = {
        "set_a_baseline": {"cv_mae": round(mae_a, 4)},
        "set_b_engineered": {"cv_mae": round(mae_b, 4)},
        "improvement": round(improvement, 4),
        "decision": "KEEP engineered features" if use_engineering else "DROP engineered features",
    }
    all_metrics["ablation_study"] = ablation_results
    logger.info("Ablation: Baseline MAE=%.4f, Engineered MAE=%.4f", mae_a, mae_b)

    # STEP 4: Regression
    logger.info("=" * 60)
    logger.info("STEP 4: Regression model comparison")
    reg_results = train_regression_models(X_train, y_train_reg, X_test, y_test_reg, use_feature_engineering=use_engineering)
    best_reg_name = reg_results.iloc[0]["Model"]
    best_reg_pipeline = reg_results.iloc[0]["Pipeline"]
    save_model(best_reg_pipeline, "regression_pipeline.joblib")
    display_cols = [c for c in reg_results.columns if c != "Pipeline"]
    all_metrics["regression"] = {"best_model": best_reg_name, "results": reg_results[display_cols].to_dict(orient="records")}

    # STEP 5: Classification
    logger.info("=" * 60)
    logger.info("STEP 5: Classification model comparison")
    cls_results = train_classification_models(X_train, y_train_cls, X_test, y_test_cls, use_feature_engineering=use_engineering)
    best_cls_name = cls_results.iloc[0]["Model"]
    uncalibrated_pipeline = cls_results.iloc[0]["Pipeline"]

    # STEP 6: Probability Calibration
    logger.info("=" * 60)
    logger.info("STEP 6: Probability Calibration")
    
    # Calculate uncalibrated Brier score using cross_val_predict to avoid leakage
    y_prob_uncalibrated_cv = cross_val_predict(uncalibrated_pipeline, X_train, y_train_cls, cv=5, method="predict_proba")[:, 1]
    brier_uncalibrated = brier_score_loss(y_train_cls, y_prob_uncalibrated_cv)
    
    calibrated_pipeline = CalibratedClassifierCV(estimator=uncalibrated_pipeline, cv=5, method="sigmoid")
    calibrated_pipeline.fit(X_train, y_train_cls)
    
    # Evaluate calibrated Brier score on CV predictions
    y_prob_calibrated_cv = cross_val_predict(calibrated_pipeline, X_train, y_train_cls, cv=5, method="predict_proba")[:, 1]
    brier_calibrated = brier_score_loss(y_train_cls, y_prob_calibrated_cv)
    
    logger.info("Brier Score (Uncalibrated): %.4f", brier_uncalibrated)
    logger.info("Brier Score (Calibrated): %.4f", brier_calibrated)
    
    # If calibration improves the Brier score by a margin, or is comparable, we keep it.
    if brier_calibrated <= brier_uncalibrated * 1.05:
        best_cls_pipeline = calibrated_pipeline
        all_metrics["calibration"] = {"applied": True, "brier_uncalibrated": brier_uncalibrated, "brier_calibrated": brier_calibrated}
        logger.info("Calibration applied. Brier score verified.")
    else:
        best_cls_pipeline = uncalibrated_pipeline
        all_metrics["calibration"] = {"applied": False, "brier_uncalibrated": brier_uncalibrated, "brier_calibrated": brier_calibrated}
        logger.info("Calibration rejected (did not improve Brier score).")
        
    save_model(best_cls_pipeline, "classification_pipeline.joblib")
    display_cols_cls = [c for c in cls_results.columns if c != "Pipeline"]
    all_metrics["classification"] = {"best_model": best_cls_name, "results": cls_results[display_cols_cls].to_dict(orient="records")}

    # STEP 7: Dynamic Threshold & Classifier Vs Reg
    logger.info("=" * 60)
    logger.info("STEP 7: Dynamic Threshold Selection")
    
    # Use CV probabilities to find optimal threshold (90% recall objective for early warning)
    y_prob_train_cv = cross_val_predict(best_cls_pipeline, X_train, y_train_cls, cv=5, method="predict_proba")[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train_cls, y_prob_train_cv)
    
    optimal_threshold = 0.5
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= 0.90:
            optimal_threshold = float(t)
            
    logger.info("Selected optimal threshold: %.3f (based on >=90%% recall objective on CV data)", optimal_threshold)
    all_metrics["classification"]["selected_threshold"] = optimal_threshold

    # Evaluate on test set with selected threshold
    y_prob_cls = best_cls_pipeline.predict_proba(X_test)[:, 1]
    y_risk_from_cls = (y_prob_cls >= optimal_threshold).astype(int)
    metrics_b = compute_classification_metrics(y_test_cls.values, y_risk_from_cls, y_prob_cls)
    
    # Regression-derived baseline
    y_pred_reg = best_reg_pipeline.predict(X_test)
    y_risk_from_reg = (y_pred_reg < get_risk_threshold()).astype(int)
    metrics_a = compute_classification_metrics(y_test_cls.values, y_risk_from_reg)
    
    all_metrics["regression_vs_classifier"] = {
        "approach_a_regression_derived": metrics_a,
        "approach_b_dedicated_classifier": metrics_b,
    }

    # STEP 8: Quantile regression tuning (Uncertainty)
    logger.info("=" * 60)
    logger.info("STEP 8: Prediction Intervals (Tuning for Empirical 80% Coverage)")
    
    # To get 80% coverage, nominally [0.1, 0.9] is expected, but GBMs underdisperse.
    # We will test [0.1, 0.9], [0.05, 0.95], and [0.02, 0.98] via CV
    candidate_pairs = [(0.1, 0.9), (0.05, 0.95), (0.02, 0.98)]
    best_pair = (0.1, 0.9)
    best_diff = 1.0
    best_empirical = 0.0
    
    for (lo_q, hi_q) in candidate_pairs:
        lo_m = build_pipeline(HistGradientBoostingRegressor(loss="quantile", quantile=lo_q, random_state=seed), use_feature_engineering=use_engineering)
        hi_m = build_pipeline(HistGradientBoostingRegressor(loss="quantile", quantile=hi_q, random_state=seed), use_feature_engineering=use_engineering)
        
        y_pred_lo_cv = cross_val_predict(lo_m, X_train, y_train_reg, cv=5)
        y_pred_hi_cv = cross_val_predict(hi_m, X_train, y_train_reg, cv=5)
        
        cov = np.mean((y_train_reg.values >= y_pred_lo_cv) & (y_train_reg.values <= y_pred_hi_cv))
        diff = abs(cov - 0.80)
        logger.info("Quantile pair (%.2f, %.2f): Empirical CV Coverage = %.2f%%", lo_q, hi_q, cov * 100)
        if diff < best_diff:
            best_diff = diff
            best_pair = (lo_q, hi_q)
            best_empirical = cov
            
    logger.info("Selected quantiles (%.2f, %.2f) with CV coverage %.2f%%", best_pair[0], best_pair[1], best_empirical * 100)
    
    # Train final quantile models on full training data
    lo_model = HistGradientBoostingRegressor(loss="quantile", quantile=best_pair[0], random_state=seed)
    hi_model = HistGradientBoostingRegressor(loss="quantile", quantile=best_pair[1], random_state=seed)
    lo_pipeline = build_pipeline(lo_model, use_feature_engineering=use_engineering)
    hi_pipeline = build_pipeline(hi_model, use_feature_engineering=use_engineering)
    
    lo_pipeline.fit(X_train, y_train_reg)
    hi_pipeline.fit(X_train, y_train_reg)
    save_model(lo_pipeline, "quantile_lo_pipeline.joblib")
    save_model(hi_pipeline, "quantile_hi_pipeline.joblib")
    
    # Test coverage
    y_lo = lo_pipeline.predict(X_test)
    y_hi = hi_pipeline.predict(X_test)
    test_cov = np.mean((y_test_reg.values >= y_lo) & (y_test_reg.values <= y_hi))
    avg_width = np.mean(y_hi - y_lo)
    
    all_metrics["prediction_intervals"] = {
        "nominal_quantiles": best_pair,
        "empirical_cv_coverage": round(float(best_empirical), 4),
        "empirical_test_coverage": round(float(test_cov), 4),
        "avg_interval_width": round(float(avg_width), 2),
    }

    # STEP 9: SHAP Explainability
    logger.info("=" * 60)
    logger.info("STEP 9: SHAP explainability")
    X_background = X_train.sample(n=min(200, len(X_train)), random_state=seed)
    save_explainer_background(X_background)
    try:
        explainer = ShapExplainer(best_reg_pipeline, X_background)
        shap_importance = explainer.mean_abs_shap(X_background)
        all_metrics["shap_feature_importance"] = {k: round(float(v), 4) for k, v in shap_importance.head(10).items()}
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)
        all_metrics["shap_feature_importance"] = {"error": str(e)}

    # STEP 10: Fairness
    logger.info("=" * 60)
    logger.info("STEP 10: Fairness analysis")
    try:
        fairness_report = full_fairness_report(
            y_true_reg=y_test_reg.values, y_pred_reg=y_pred_reg,
            y_true_cls=y_test_cls.values, y_pred_cls=y_risk_from_cls, y_prob_cls=y_prob_cls,
            df=X_test, protected_attributes=["Gender", "Family_Income"],
        )
        all_metrics["fairness"] = fairness_report
    except Exception as e:
        logger.warning("Fairness analysis failed: %s", e)
        all_metrics["fairness"] = {"error": str(e)}

    # STEP 11: Save artifacts
    save_metrics(all_metrics)
    save_model_metadata(
        model_name=best_reg_name,
        version="1.1.0",
        feature_set="engineered" if use_engineering else "baseline",
        metrics={
            "regression_test_mae": all_metrics["regression"]["results"][0]["Test_MAE"],
            "classification_test_recall": metrics_b["recall"],
            "classification_threshold": optimal_threshold
        },
    )
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %.1f seconds", elapsed)

if __name__ == "__main__":
    main()
