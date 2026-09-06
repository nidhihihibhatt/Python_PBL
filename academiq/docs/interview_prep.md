# AcademIQ: Technical Interview Preparation

This document provides technically rigorous answers to defend the design decisions in AcademIQ during an ML engineering interview.

## 1. Why was probability calibration implemented, and did it help?
**Q: "You used Logistic Regression for classification. Did you calibrate the probabilities? Why or why not?"**

**A:** We did evaluate it programmatically. Our pipeline evaluates the uncalibrated Logistic Regression Brier score (via out-of-fold `cross_val_predict`) and compares it against a `CalibratedClassifierCV` (sigmoid approach). In our final run, the uncalibrated model achieved a Brier score of **0.0136**, while the calibrated model achieved **0.0154** (worse). Because Logistic Regression is often natively well-calibrated via log-loss, forcing Platt scaling caused slight overfitting on the CV folds. Because the calibration degraded the Brier score, our pipeline dynamically *rejected* the explicit calibration layer to preserve simplicity and avoid overfitting. This automated gatekeeping prevents us from deploying unnecessary complexity.

## 2. Dynamic Threshold Selection
**Q: "How did you choose the classification threshold to categorize a student as 'High Risk'?"**

**A:** Rather than hardcoding 0.5 or blindly picking a threshold on the test set (which causes leakage), we used `cross_val_predict` on the training set to generate unbiased out-of-fold probabilities. We then plotted the Precision-Recall curve and programmatically selected the threshold that achieved a **Recall of ≥ 90%**. Since AcademIQ is an Early-Warning system, false negatives (missing a failing student) are vastly more costly than false positives (giving a passing student extra tutoring). This chosen threshold is saved in the metadata and loaded dynamically during inference.

## 3. Resolving Prediction Interval Underdispersion
**Q: "Your prediction intervals target 80% coverage. Did you encounter any issues hitting that target?"**

**A:** Yes, significantly. We used `HistGradientBoostingRegressor` with a quantile loss function. Initially, asking the tree for the 10th and 90th percentiles only yielded ~68.8% empirical coverage on our test set. This is a known issue: gradient boosted trees often suffer from underdispersion (predicting too tightly) on low-variance targets. 
To fix this honestly without cheating on the test set, we tested wider nominal quantile pairs like [0.05, 0.95] and [0.02, 0.98] using 5-fold CV on the training data. We selected the pair whose *empirical* coverage was closest to 80%, and then applied that model to the test set.

## 4. SHAP: Explanation vs. Causation
**Q: "Your dashboard says 'Attendance' drove the score up by 5 points. Does that mean if the student attends 5 more classes, they will get 5 more points?"**

**A:** No. SHAP values explain the *model's behavior*, not the real-world physics of the classroom. They measure the statistical association the model learned. Attendance is highly correlated with other unmeasured positive behaviors (like paying attention). Forcing a student to sit in a classroom won't magically grant them 5 points if they are asleep. This is why our dashboard includes a prominent causal disclaimer.

## 5. Preventing Data Leakage
**Q: "Walk me through how you prevented target leakage, especially with your imputation and scaling."**

**A:** We strictly encapsulated all preprocessing steps (Imputation, Ordinal Encoding, One-Hot Encoding, and Standard Scaling) inside a single `sklearn.pipeline.Pipeline`. This ensures that when we call `.fit(X_train)`, the mean/variance for scaling and the modes for imputation are learned *only* from the training folds. During inference, we pass the raw dictionary to `.predict()`, which applies those exact stored parameters.

## 6. The Feature Engineering Ablation Study
**Q: "You created features like 'Study_Effort' and 'Engagement'. How do you know they actually work?"**

**A:** We don't assume they work; we measure them. The training pipeline runs an automated ablation study using 5-fold cross-validation. It trains a baseline model on raw features, and another on the engineered features. It calculates the CV MAE difference. In our case, the engineered features provided *zero* measurable improvement (difference < 0.01 MAE). Therefore, the pipeline dynamically rejected them to keep the model simpler and more robust.

## 7. Fairness Evaluation
**Q: "Did you check if your model is biased against certain demographics?"**

**A:** Yes, we evaluated predictive parity across Gender and Family_Income. We computed the Mean Absolute Error (MAE) and Mean Residuals separately for these subgroups. We found slight disparities (e.g., Male MAE was slightly higher than Female MAE). However, since there is no mechanism in the pipeline that uses these protected attributes to threshold recommendations unfairly, the impact is monitored but currently acceptable. 

## 8. Limitations & Production Readiness
**Q: "What is the biggest limitation of this project, and what would it take to put this in production?"**

**A:** The biggest limitation is the dataset itself—it is a static, highly idealized synthetic/academic dataset with no longitudinal tracking (time-series data of a student's grades over a semester). 
To make this truly production-ready, we would need:
1. **Longitudinal features** (e.g., "Change in attendance over last 3 weeks").
2. **A Feature Store** to serve up-to-date aggregates during inference.
3. **Model Monitoring** (using evidently/whylogs) to track data drift over academic semesters.
