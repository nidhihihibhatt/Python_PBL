import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data.loader import load_data
from src.inference.service import PredictionService
from src.config import get_all_input_features, get_ordinal_features, get_nominal_features, get_numeric_features

st.set_page_config(page_title="Student Analysis — AcademIQ", layout="wide")
st.title("🧑‍🎓 Student Analysis")

@st.cache_resource
def get_service():
    try:
        return PredictionService()
    except Exception as e:
        return None

@st.cache_data
def get_raw_data():
    return load_data()

service = get_service()
if service is None:
    st.warning("⚠️ Models not trained. Run `python train.py` first.")
    st.stop()

df = get_raw_data()

tab1, tab2 = st.tabs(["📋 Predict from Dataset", "✏️ New Student"])

def render_prediction(result):
    """Render a prediction result."""
    if "error" in result:
        st.error(result["error"])
        return

    # Score and Risk
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Score", f"{result['predicted_score']:.1f}")
    with c2:
        risk = result["risk_category"]
        color = {"High Risk": "🔴", "Moderate Risk": "🟡", "Low Risk": "🟢"}.get(risk, "⚪")
        st.metric("Risk Level", f"{color} {risk}")
    with c3:
        prob = result.get("risk_probability")
        if prob is not None:
            st.metric("Risk Probability", f"{prob * 100:.1f}%")

    # Prediction interval
    interval = result.get("prediction_interval")
    if interval:
        st.info(f"📐 **Prediction Interval ({interval['coverage']}):** "
                f"{interval['lower']} – {interval['upper']}")

    # SHAP Explanation
    shap = result.get("shap_explanation")
    if shap:
        st.markdown("---")
        st.markdown("#### Why This Prediction?")

        pos = shap.get("top_positive_contributors", [])
        neg = shap.get("top_negative_contributors", [])

        all_factors = []
        for f in pos:
            all_factors.append({"Feature": f["feature"], "Impact": f["shap_value"], "Direction": "Positive"})
        for f in neg:
            all_factors.append({"Feature": f["feature"], "Impact": f["shap_value"], "Direction": "Negative"})

        if all_factors:
            factors_df = pd.DataFrame(all_factors).sort_values("Impact")
            colors = ["#38a169" if d == "Positive" else "#e53e3e" for d in factors_df["Direction"]]
            fig = go.Figure(go.Bar(
                x=factors_df["Impact"], y=factors_df["Feature"],
                orientation="h", marker_color=colors,
            ))
            fig.update_layout(
                height=max(250, len(all_factors) * 35),
                xaxis_title="SHAP Value (impact on prediction)",
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    recs = result.get("recommendations", [])
    if recs:
        st.markdown("---")
        st.markdown("#### 💡 Recommendations")
        for i, rec in enumerate(recs, 1):
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec["priority"], "⚪")
            with st.expander(f"{priority_icon} {rec['category']}: {rec['message'][:80]}...", expanded=(i <= 2)):
                st.markdown(f"**Priority:** {rec['priority'].title()}")
                st.markdown(f"**Feature:** {rec['feature']} = {rec['feature_value']}")
                st.markdown(f"**SHAP Impact:** {rec['shap_value']:.4f}")
                st.markdown(f"**Reason:** {rec['reason']}")
    else:
        st.success("✅ No specific recommendations — this student's profile looks positive.")

    st.markdown(
        '<div class="causal-disclaimer">⚠️ These explanations show statistical '
        'associations, not causal effects. Correlation ≠ causation.</div>',
        unsafe_allow_html=True,
    )

# --- Tab 1: Dataset Prediction ---
with tab1:
    st.subheader("Select a Student from the Dataset")
    idx = st.number_input("Student Index", min_value=0, max_value=len(df) - 1, value=0, step=1)

    if st.button("🔍 Predict", key="predict_dataset"):
        student_row = df.iloc[idx].drop("Exam_Score", errors="ignore")
        actual = df.iloc[idx].get("Exam_Score", None)

        result = service.predict(student_row.to_dict())

        if actual is not None:
            st.markdown(f"**Actual Exam Score:** {actual}")
        render_prediction(result)

# --- Tab 2: New Student ---
with tab2:
    st.subheader("Enter Student Information")

    with st.form("new_student_form"):
        cols = st.columns(3)

        # Numeric features
        numeric = get_numeric_features()
        inputs = {}
        defaults = {"Hours_Studied": 20, "Attendance": 80, "Sleep_Hours": 7,
                     "Previous_Scores": 70, "Tutoring_Sessions": 2, "Physical_Activity": 3}
        for i, feat in enumerate(numeric):
            with cols[i % 3]:
                inputs[feat] = st.number_input(feat.replace("_", " "), value=defaults.get(feat, 5))

        # Ordinal features
        ordinal = get_ordinal_features()
        for i, (feat, categories) in enumerate(ordinal.items()):
            with cols[i % 3]:
                inputs[feat] = st.selectbox(feat.replace("_", " "), categories, index=1)

        # Nominal features
        nominal = get_nominal_features()
        nominal_options = {
            "Gender": ["Male", "Female"],
            "Extracurricular_Activities": ["Yes", "No"],
            "Internet_Access": ["Yes", "No"],
            "School_Type": ["Public", "Private"],
            "Learning_Disabilities": ["No", "Yes"],
        }
        for i, feat in enumerate(nominal):
            with cols[i % 3]:
                opts = nominal_options.get(feat, ["Yes", "No"])
                inputs[feat] = st.selectbox(feat.replace("_", " "), opts)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        result = service.predict(inputs)
        render_prediction(result)
