AcademIQ: Academic Early Warning System

Overview
AcademIQ is an end-to-end machine learning pipeline that predicts student academic performance and flags students who might need early intervention.
Standard prediction models usually just output a final score and stop there, which isn't very helpful for making actual decisions. I built this system to go a step further and answer four specific questions:

What score is the student likely to get? (Regression)
Are they at risk of falling behind? (Classification)
Why did the model make this prediction? (Explainability)
What can be done to help? (Recommendation Engine)

Features
1. Score Prediction
The system uses regression models to estimate exam scores based on academic, behavioral, and demographic data. Instead of just assuming the model is good, performance is rigorously evaluated (R², MAE, RMSE) and compared against a simple baseline to prove the ML actually adds value.

2. Risk Detection
A classification pipeline acts as an early-warning system to flag at-risk students. The classification threshold isn't just left at the default probability—it is specifically tuned for an early-warning use case based on Precision, Recall, and F1-scores.

3. Model Explainability (SHAP)
Predictions shouldn't be black boxes. The project uses SHAP to break down exactly which features (like attendance or study hours) pushed an individual student's prediction up or down. (Note: SHAP explains how the model behaves, but doesn't strictly prove real-world causation).

4. Evidence-Based Recommendations
If a student is flagged as at-risk, the system suggests interventions. To prevent generic advice, it only makes a recommendation if both the student's actual data (e.g., low study hours) AND the individual SHAP values agree that the factor is negatively impacting their score.

5. Ablation Study
Rather than just throwing engineered features into the model and hoping for the best, the project includes an ablation study. It directly compares baseline features against engineered features to prove exactly how much the feature engineering improved the model.
