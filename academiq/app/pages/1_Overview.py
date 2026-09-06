import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

from src.data.loader import load_data
from src.inference.service import PredictionService
from src.config import REPORTS_DIR, get_risk_threshold

st.set_page_config(page_title="Overview — AcademIQ", layout="wide")
st.title("📊 Executive Dashboard")

@st.cache_resource
def get_service():
    try:
        return PredictionService()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

@st.cache_data
def get_raw_data():
    return load_data()

@st.cache_data
def load_metrics():
    try:
        path = REPORTS_DIR / "metrics.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

service = get_service()
if service is None:
    st.warning("⚠️ Models not trained. Run `python train.py` first.")
    st.stop()

df = get_raw_data()
metrics = load_metrics()

# Drop target for prediction
X = df.drop(columns=["Exam_Score"], errors="ignore")
preds_df = service.predict_batch(X)
preds_df["actual_score"] = df["Exam_Score"].values

# --- KPI Row ---
st.markdown("### Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students Analysed", len(preds_df))
c2.metric("Mean Predicted Score", f"{preds_df['predicted_score'].mean():.1f}")

high_risk = (preds_df["risk_category"] == "High Risk").sum()
c3.metric("High Risk Students", high_risk)
c4.metric("High Risk %", f"{high_risk / len(preds_df) * 100:.1f}%")

st.divider()

# --- Charts Row ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Predicted Score Distribution")
    fig = px.histogram(
        preds_df, x="predicted_score", nbins=30,
        color="risk_category",
        color_discrete_map={"High Risk": "#e53e3e", "Moderate Risk": "#d69e2e", "Low Risk": "#38a169"},
        category_orders={"risk_category": ["High Risk", "Moderate Risk", "Low Risk"]},
    )
    fig.add_vline(x=get_risk_threshold(), line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({get_risk_threshold()})")
    fig.update_layout(bargap=0.05, showlegend=True, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("#### Risk Category Distribution")
    risk_counts = preds_df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Category", "Count"]
    fig2 = px.pie(
        risk_counts, values="Count", names="Category",
        color="Category",
        color_discrete_map={"High Risk": "#e53e3e", "Moderate Risk": "#d69e2e", "Low Risk": "#38a169"},
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- SHAP Feature Importance ---
st.markdown("#### Top Contributing Factors (Global)")
if metrics and "shap_feature_importance" in metrics:
    shap_data = metrics["shap_feature_importance"]
    if "error" not in shap_data:
        shap_sorted = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        names = [s[0] for s in shap_sorted]
        values = [s[1] for s in shap_sorted]
        fig3 = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#4a5568"))
        fig3.update_layout(
            yaxis=dict(autorange="reversed"), height=350,
            xaxis_title="Mean |SHAP Value|", yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("SHAP analysis encountered an error during training.")
else:
    st.info("SHAP feature importance not available. Re-run training.")

# --- Model Summary ---
if metrics:
    st.divider()
    st.markdown("#### Model Summary")
    col_a, col_b = st.columns(2)

    with col_a:
        if "regression" in metrics:
            reg = metrics["regression"]
            st.markdown(f"**Best Regression Model:** {reg.get('best_model', 'N/A')}")
            if "results" in reg and reg["results"]:
                best = reg["results"][0]
                st.metric("Test MAE", f"{best.get('Test_MAE', 'N/A')}")
                st.metric("Test R²", f"{best.get('Test_R2', 'N/A')}")

    with col_b:
        if "classification" in metrics:
            cls = metrics["classification"]
            st.markdown(f"**Best Classification Model:** {cls.get('best_model', 'N/A')}")
            if "results" in cls and cls["results"]:
                best = cls["results"][0]
                st.metric("Test Recall", f"{best.get('Test_Recall', 'N/A')}")
                st.metric("Test PR-AUC", f"{best.get('Test_PR_AUC', 'N/A')}")

st.markdown(
    '<div class="causal-disclaimer">⚠️ These explanations show statistical '
    'associations, not causal effects. Correlation ≠ causation.</div>',
    unsafe_allow_html=True,
)
