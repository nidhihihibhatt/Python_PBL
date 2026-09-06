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

st.set_page_config(page_title="Risk & Early Warning — AcademIQ", layout="wide")
st.title("⚠️ Risk & Early Warning")

@st.cache_resource
def get_service():
    try:
        return PredictionService()
    except Exception:
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

X = df.drop(columns=["Exam_Score"], errors="ignore")
preds_df = service.predict_batch(X)
preds_df["actual_score"] = df["Exam_Score"].values
preds_df.index = df.index

# --- Risk Distribution ---
st.markdown("### Risk Distribution")
col1, col2, col3 = st.columns(3)
high = (preds_df["risk_category"] == "High Risk").sum()
mod = (preds_df["risk_category"] == "Moderate Risk").sum()
low = (preds_df["risk_category"] == "Low Risk").sum()

col1.metric("🔴 High Risk", high)
col2.metric("🟡 Moderate Risk", mod)
col3.metric("🟢 Low Risk", low)

fig = px.bar(
    x=["High Risk", "Moderate Risk", "Low Risk"],
    y=[high, mod, low],
    color=["High Risk", "Moderate Risk", "Low Risk"],
    color_discrete_map={"High Risk": "#e53e3e", "Moderate Risk": "#d69e2e", "Low Risk": "#38a169"},
)
fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count", height=300)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- High Risk Students Table ---
st.markdown("### High-Risk Students Requiring Attention")
high_risk_df = preds_df[preds_df["risk_category"] == "High Risk"].copy()

if len(high_risk_df) > 0:
    display_cols = ["predicted_score", "risk_probability", "actual_score"]
    if "interval_lower" in high_risk_df.columns:
        display_cols.extend(["interval_lower", "interval_upper"])
    st.dataframe(
        high_risk_df[display_cols].sort_values("risk_probability", ascending=False),
        use_container_width=True,
        height=min(400, 40 + len(high_risk_df) * 35),
    )
    st.caption(f"Showing {len(high_risk_df)} high-risk students out of {len(preds_df)} total.")
else:
    st.success("No high-risk students identified at the current threshold.")

st.divider()

# --- Threshold Analysis ---
st.markdown("### Threshold Sensitivity Analysis")
st.markdown(
    "How precision and recall change across different probability thresholds. "
    "Lower thresholds catch more at-risk students (higher recall) but flag more "
    "false positives (lower precision)."
)

if metrics and "threshold_analysis" in metrics:
    thresh_data = metrics["threshold_analysis"]
    thresh_df = pd.DataFrame(thresh_data)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["precision"],
                              mode="lines+markers", name="Precision", line=dict(color="#3182ce")))
    fig2.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["recall"],
                              mode="lines+markers", name="Recall", line=dict(color="#e53e3e")))
    fig2.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["f1"],
                              mode="lines+markers", name="F1", line=dict(color="#38a169", dash="dash")))
    fig2.update_layout(
        xaxis_title="Probability Threshold",
        yaxis_title="Score",
        height=400,
        legend=dict(x=0.7, y=0.1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(thresh_df, use_container_width=True)
else:
    st.info("Threshold analysis not available. Re-run training.")

st.divider()

# --- False Positive / Negative Cost ---
st.markdown("### Understanding Prediction Errors")
st.markdown("""
| Error Type | What Happens | Real-World Impact | Acceptable? |
|-----------|-------------|-------------------|-------------|
| **False Negative** (missed at-risk student) | Student doesn't receive intervention | Potential academic failure | ❌ Worse |
| **False Positive** (flagged safe student) | Student receives unnecessary attention | Minor resource cost, no harm | ⚠️ Better |

For an **early-warning system**, false negatives are more costly. This is why we 
optimise for **recall** (catching at-risk students) while accepting some false positives.
""")

# --- Regression vs Classifier experiment ---
if metrics and "regression_vs_classifier" in metrics:
    st.divider()
    st.markdown("### Regression-Derived Risk vs. Dedicated Classifier")
    exp = metrics["regression_vs_classifier"]

    comp_data = {
        "Approach": ["Regression → Threshold", "Dedicated Classifier"],
        "Recall": [exp["approach_a_regression_derived"]["recall"],
                   exp["approach_b_dedicated_classifier"]["recall"]],
        "F1": [exp["approach_a_regression_derived"]["f1"],
               exp["approach_b_dedicated_classifier"]["f1"]],
        "Precision": [exp["approach_a_regression_derived"]["precision"],
                      exp["approach_b_dedicated_classifier"]["precision"]],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    verdict = "✅ Yes" if exp.get("classifier_adds_value") else "❌ No"
    st.markdown(f"**Does the dedicated classifier add meaningful value?** {verdict}")

st.markdown(
    '<div class="causal-disclaimer">⚠️ These explanations show statistical '
    'associations, not causal effects. Correlation ≠ causation.</div>',
    unsafe_allow_html=True,
)
