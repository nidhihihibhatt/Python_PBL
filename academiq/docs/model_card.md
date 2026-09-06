# AcademIQ — Model Card

## Model Details

| Property | Value |
|----------|-------|
| **Name** | AcademIQ Student Performance Prediction System |
| **Version** | 1.1.0 |
| **Type** | Regression + Classification (risk detection) |
| **Framework** | scikit-learn |
| **Developer** | Student project |
| **Date** | August 2026 |

## Intended Use

### Primary Use
Decision-support tool for academic administrators and educators to:
- Estimate expected student exam performance
- Identify students who may benefit from additional academic attention
- Understand which observable factors are associated with predictions
- Generate transparent, rule-based intervention suggestions

### Out-of-Scope Uses
- **Automated academic decisions** — This system should NOT automatically determine grades or intervention actions without human review.
- **Causal inference** — The model identifies statistical associations, not causal relationships.

## System Architecture Highlights
- **Pipeline Consistency**: A strictly modular `sklearn.pipeline.Pipeline` orchestrates imputation, encoding, and feature scaling to guarantee training/inference parity.
- **Ablation Validated**: Engineered features are dynamically included/excluded based on automated cross-validation ablation studies.
- **Dynamic Thresholding**: The classification threshold is not hardcoded; it is chosen dynamically during cross-validation to guarantee a recall of ≥90% for early warning.
- **Probability Calibration**: Probabilities are evaluated against a sigmoid-calibrated version (`CalibratedClassifierCV`) via Brier Score to ensure reliable risk probabilities.
- **Uncertainty**: 80% prediction intervals are provided via Quantile Regression (`HistGradientBoostingRegressor`). The nominal quantiles are dynamically tuned on cross-validation data to achieve an empirical 80% coverage (mitigating gradient boosting underdispersion).

## Limitations

### Data Limitations
1. **Synthetic data**: The dataset appears highly idealized. Real-world relationships are noisier.
2. **Cross-sectional**: No temporal dimension — cannot track improvement over time.
3. **Narrow score distribution**: The Exam_Score distribution is tight, making ultra-precise point predictions challenging.
4. **Limited demographics**: Cannot assess fairness across race or ethnicity (data unavailable).

### Model Limitations
1. **Association, not causation**: All explanations (SHAP) reflect statistical associations, not causal mechanisms.
2. **Threshold is project-defined**: The baseline risk threshold (65) is a project design choice.
3. **Missing feature interactions**: Linear models cannot capture complex interactions without manual engineering.
4. **Prediction intervals are approximate**: While tuned to 80%, intervals on unseen long-tail distributions may still degrade.

## Ethical Considerations
1. **Student welfare**: Predictions should be used to support students, not penalise them. A "high risk" prediction should trigger support mechanisms, not academic sanctions.
2. **Label bias**: The model may reproduce historical socioeconomic biases present in the training data.
3. **Privacy**: Student-level predictions should be handled with appropriate data protection measures.
