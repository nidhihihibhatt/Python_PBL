import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference.service import PredictionService
from src.data.loader import load_data

def verify_all():
    service = PredictionService()
    df = load_data()
    
    student = df.iloc[0].drop("Exam_Score").to_dict()
    result = service.predict(student)
    
    print("--- SHAP Verification ---")
    shap_exp = result.get("shap_explanation", {})
    print(f"Top Pos: {shap_exp.get('top_positive_contributors', [])[:2]}")
    print(f"Top Neg: {shap_exp.get('top_negative_contributors', [])[:2]}")
    
    print("\n--- Recommendation Verification ---")
    recs = result.get("recommendations", [])
    for r in recs:
        print(f"Rule: {r['rule_id']} | Priority: {r['priority']} | Reason: {r['reason']}")
    
    print("\n--- Pipeline Preprocessing Verification ---")
    # test raw prediction directly with pipeline
    import pandas as pd
    X_single = pd.DataFrame([student])
    raw_pred = service.reg_pipeline.predict(X_single)[0]
    print(f"Direct pipeline pred: {raw_pred}, Service pred: {result['predicted_score']}")

if __name__ == "__main__":
    verify_all()
