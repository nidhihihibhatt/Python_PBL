# AcademIQ: Final Verification Audit

## A. Executive Verdict
**Classification: READY AFTER FIXES**

The AcademIQ project exhibits a highly professional architecture. The transition from a basic Jupyter Notebook to a modular, pipeline-driven ML system is exceptional. The critical data leakage and encoding mismatches from the original project have been completely eliminated. The integration of ablation studies, threshold optimization, and dual-gated recommendation engines are strong technical achievements that stand out for a portfolio project.

However, a rigorous audit revealed three critical omissions/weaknesses that must be addressed before presenting this to recruiters or pushing to a public GitHub repository. Most notably, the local environment was missing UI dependencies, and explicit probability calibration was skipped.

---

## B. Verified Features (PASS)
- **Training ↔ Inference Consistency [PASS]:** The entire preprocessing chain (Imputation → Ordinal Encoding → OneHot Encoding → Scaling) is serialized inside an `sklearn.pipeline.Pipeline`. Raw dictionary inputs passed to `PredictionService.predict()` traverse the exact same graph as training data. No manual encoding mismatch exists.
- **Data Pipeline & Feature Engineering [PASS]:** The `FeatureEngineer` class dynamically creates features. The ablation study correctly measured `0.0` improvement and strictly chose to *drop* the engineered features to prevent overfitting. This is a massive interview win.
- **Classification vs. Regression Experiment [PASS]:** The pipeline empirically proved that regression-derived risk thresholding (Recall=0.99) outperformed a dedicated classifier (Recall=0.95), justifying the architecture.
- **Recommendation Engine [PASS]:** The dual-gate logic works. Verification confirmed recommendations only trigger when a raw feature meets a threshold *and* SHAP confirms a negative contribution.
- **Reproducibility [PASS]:** `model_config.yaml` cleanly separates parameters. The `.gitignore` successfully prevents model weights from bloating the repo.

---

## C. Failed Claims (FAIL)
- **Probability Calibration [FAIL]:** The `interview_prep.md` document claims the system evaluates calibration and applies Platt scaling/Isotonic regression. **This is false.** The code uses raw `LogisticRegression` probabilities. While Logistic Regression is naturally well-calibrated, the explicit calibration step (`CalibratedClassifierCV`) is missing from `train.py`.
- **Dashboard Execution [FAIL]:** The `app/app.py` Streamlit dashboard is fully coded, but during testing, `streamlit` and `plotly` were missing from the local environment (despite being in `requirements.txt`). The dashboard could not be natively verified in the background.

---

## D. Actual Model Metrics (PASS)
The models were run via 5-fold Cross Validation. The dynamic selection correctly chose **Ridge** and **LogisticRegression**.

| Model (Regression) | CV MAE | Test MAE | Test RMSE | Test R² | Train Time |
|-------------------|--------|----------|-----------|---------|------------|
| Ridge | 0.4767 | 0.5218 | 2.2665 | 0.6898 | 6.49s |
| Lasso | 0.4769 | 0.5219 | 2.2666 | 0.6897 | 0.73s |
| HistGBM | 0.8379 | 0.8754 | 2.3756 | 0.6592 | 6.70s |
| Dummy Baseline | 2.8226 | 2.9407 | 4.0720 | -0.0013 | 0.53s |

---

## E. Actual Uncertainty Coverage (WARNING)
- **Claim:** 80% Prediction Interval via Quantile Regression (`lower_quantile=0.1`, `upper_quantile=0.9`).
- **Actual Verification:** The empirical coverage on the test set is **68.84%** with an average width of 2.41 points. 
- **Verdict:** The intervals are underdispersed (too tight). This is common for HistGradientBoosting on tight score distributions. **Action required:** Either update the documentation to reflect ~69% coverage, or widen the quantiles (e.g., 0.05 and 0.95) to hit the 80% target.

---

## F. Actual SHAP Verification (PASS)
- **Execution:** Verified via `verify_audit.py`.
- **Result:** Successfully extracted feature names (the previous `x0` bug is fixed). SHAP outputs exactly match the pipeline's internal feature names (e.g., `Attendance` = +1.06 impact, `Parental_Involvement` = -1.01 impact). 
- **Claim check:** The `model_card.md` correctly restricts SHAP to *statistical associations*, avoiding causal traps.

---

## G. Actual Fairness Results (PASS)
Fairness analysis ran successfully on the regression target. Disparities are below the strict internal alert threshold:
- **Gender:** Female (MAE=0.46), Male (MAE=0.56)
- **Income:** High (MAE=0.61), Medium (MAE=0.49), Low (MAE=0.52)
- Limitations are accurately documented in the `fairness_report.md`.

---

## H. Actual Test Results (PASS)
- **Suite:** Executed `python -m pytest tests/ -v`.
- **Result:** **46 Passed, 0 Failed, 0 Skipped.**
- **Execution Time:** ~25 seconds.
- **Coverage:** Tests end-to-end prediction, missing value handling, exact boundary evaluation for rules, and data validation. Extremely strong for a portfolio project.

---

## I. Code-Quality & GitHub Readiness (PASS/WARNING)
- **Positives:** Code is highly modular (`src/data`, `src/models`, etc.). No hardcoded model weights. No notebook checkpoints committed.
- **Weaknesses:** Missing type hints on some custom transformers. The raw dataset (`StudentPerformanceFactors.csv`, ~640KB) is not in `.gitignore`. (Acceptable for this size, but should be tracked via DVC or git LFS in a real enterprise project).

---

## J. Interview Risks & Defensibility
If an interviewer probes the project as currently written:
1. **Risk:** "You mentioned calibrating probabilities in your interview guide. Walk me through how you implemented Brier Score and Platt Scaling."
   - *Failure:* The code doesn't do this. 
2. **Risk:** "Your dashboard displays a final risk assessment. How did you choose the threshold?"
   - *Failure:* The `train.py` script calculates a beautiful threshold analysis (precision/recall tradeoff), but `configs/model_config.yaml` hardcodes `high_risk_min_prob: 0.7`. The analysis isn't dynamically setting the deployed threshold.
3. **Risk:** "Why does your 80% prediction interval only capture 68% of the actual scores?"
   - *Mitigation:* You must be prepared to answer: "Gradient Boosting trees often struggle with extreme quantiles on low-variance targets, causing underdispersion."

---

## K. Priority Fixes Required (Before GitHub/Interviews)
1. **Install UI Dependencies:** Run `pip install streamlit plotly` and manually visually QA the dashboard.
2. **Fix Calibration Documentation:** Either implement `CalibratedClassifierCV` in `train.py`, or remove the claim from `interview_prep.md`.
3. **Align Threshold Logic:** Document exactly *why* 0.7 was chosen in the config file (e.g., "Chosen from the threshold analysis because it yielded 90% precision").
4. **Clarify Quantile Coverage:** Add a note in the documentation explicitly acknowledging the 69% empirical coverage of the 80% theoretical prediction interval. Being honest about model failures is a massive green flag for senior engineers.
