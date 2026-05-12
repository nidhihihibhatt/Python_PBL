import pickle
import pandas as pd
# LOAD MODELS
with open("regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    cols = pickle.load(f)
input_dict = {
    "Hours_Studied": float(input("Hours Studied: ")),
    "Attendance": float(input("Attendance: ")),
    "Parental_Involvement": input("Parental Involvement (Low/Medium/High): "),
    "Access_to_Resources": input("Access to Resources (Low/Medium/High): ")
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

input_encoded = input_encoded.reindex(columns=cols, fill_value=0)
score = reg_model.predict(input_encoded)[0]

threshold = 40   
if score >= threshold:
    result = "Pass"
else:
    result = "Fail"

# Risk level
if score < 40:
    risk = "High Risk"
elif score < 60:
    risk = "Moderate Risk"
else:
    risk = "Low Risk"
print("\n RESULT ")
print("Predicted Score:", round(score, 2))
print("Pass/Fail:", result)
print("Risk Level:", risk)