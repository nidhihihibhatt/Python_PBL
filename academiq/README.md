# AcademIQ: Academic Early Warning System

AcademIQ is an end-to-end machine learning system I built to predict student performance and flag when someone might need early intervention.

The Goal
Most ML projects using student data stop at a basic output: "Predicted score = 72."

I wanted to build something closer to a real-world decision-support tool. A raw score alone doesn't give educators the context they need to actually help a student. To make the model useful, AcademIQ is designed to answer five specific questions:

The Prediction: What is the student's expected score?
The Uncertainty: How confident is the model in this exact number?
The Risk: Is this student actually at risk of falling behind?
The "Why": Which specific factors drove the model to make this prediction? (Using SHAP)
The Action: What targeted interventions could potentially help?

Building for Reliability
Rather than just looking at standard accuracy metrics and calling it a day, a major focus of this project was proving the model's reliability. The system actively evaluates its own probability calibration, tests the empirical coverage of its prediction intervals, and runs subgroup analysis to check for fairness. In a real-world setting, a model shouldn't just be trusted blindly, and this architecture reflects that.