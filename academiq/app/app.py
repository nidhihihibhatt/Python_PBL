"""AcademIQ — Streamlit Dashboard Entry Point.

Professional multi-page analytics dashboard with four core sections:
  1. Overview — Executive summary with KPI cards
  2. Student Analysis — Individual prediction + explanation
  3. Risk & Early Warning — Risk distribution + threshold analysis
  4. Model & Methodology — Model performance + fairness + model card
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AcademIQ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Clean, professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 500;
        margin-top: 0.25rem;
        letter-spacing: 0.02em;
    }

    /* Risk badges */
    .risk-high { color: #e53e3e; font-weight: 600; }
    .risk-moderate { color: #d69e2e; font-weight: 600; }
    .risk-low { color: #38a169; font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    /* Disclaimer styling */
    .causal-disclaimer {
        background: #fffbeb;
        border: 1px solid #f6e05e;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #744210;
        margin: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("# 🎓 AcademIQ")
    st.markdown(
        "<p style='color: #a0aec0; font-size: 0.8rem; margin-top: -0.5rem;'>"
        "Academic Intelligence & Early Warning System</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='color: #a0aec0; font-size: 0.75rem;'>"
        "Built with scikit-learn, SHAP, and Streamlit.<br>"
        "This is a decision-support prototype.<br>"
        "It should not automatically determine<br>"
        "a student's academic future.</p>",
        unsafe_allow_html=True,
    )

# --- Landing Page ---
st.markdown("# 🎓 AcademIQ")
st.markdown("### Explainable Student Academic Intelligence & Early Warning System")

st.markdown("---")

st.markdown("""
**AcademIQ** answers five questions about every student:

| # | Question | Method |
|---|----------|--------|
| 1 | **What** is their expected academic performance? | Regression (predicted score) |
| 2 | **How uncertain** is that prediction? | Quantile regression (prediction intervals) |
| 3 | **Who** may require academic attention? | Risk classification (calibrated probabilities) |
| 4 | **Why** did the model produce that assessment? | SHAP explanations (global + local) |
| 5 | **What** transparent actions could be considered? | Rule-based recommendations |

---

👈 **Use the sidebar** to navigate to the system pages.
""")

# Check if models exist
models_dir = Path(__file__).resolve().parent.parent / "models"
if not (models_dir / "regression_pipeline.joblib").exists():
    st.warning(
        "⚠️ **Models not found.** Run `python train.py` from the `academiq/` "
        "directory first to train the models and generate artifacts."
    )
else:
    st.success("✅ Models loaded. Navigate to any section to begin.")
