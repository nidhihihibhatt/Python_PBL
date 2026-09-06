import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import json

from src.config import REPORTS_DIR

st.set_page_config(page_title="Model & Methodology — AcademIQ", layout="wide")
st.title("🔬 Model & Methodology")

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

@st.cache_data
def load_csv(filename):
    path = REPORTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

metrics = load_metrics()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "🧪 Ablation Study",
    "⚖️ Fairness Analysis",
    "📄 Model Card",
])

# --- Tab 1: Model Comparison ---
with tab1:
    st.markdown("### Regression Model Comparison")
    reg_df = load_csv("regression_comparison.csv")
    if reg_df is not None:
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        if metrics and "regression" in metrics:
            st.success(f"✅ **Best Model:** {metrics['regression'].get('best_model', 'N/A')}")
    else:
        st.info("Run `python train.py` to generate model comparison data.")

    st.divider()

    st.markdown("### Classification Model Comparison")
    cls_df = load_csv("classification_comparison.csv")
    if cls_df is not None:
        st.dataframe(cls_df, use_container_width=True, hide_index=True)
        if metrics and "classification" in metrics:
            st.success(f"✅ **Best Model:** {metrics['classification'].get('best_model', 'N/A')}")
    else:
        st.info("Run `python train.py` to generate classification comparison data.")

    # Prediction intervals
    if metrics and "prediction_intervals" in metrics:
        st.divider()
        st.markdown("### Prediction Intervals (Uncertainty)")
        pi = metrics["prediction_intervals"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Target Coverage", pi.get("target_coverage", "N/A"))
        c2.metric("Actual Coverage", f"{pi.get('actual_coverage', 0) * 100:.1f}%")
        c3.metric("Avg Interval Width", f"{pi.get('avg_interval_width', 0):.2f} points")

# --- Tab 2: Ablation Study ---
with tab2:
    st.markdown("### Feature Engineering Ablation Study")
    st.markdown(
        "Does feature engineering improve model performance? "
        "We compare HistGradientBoosting with and without engineered features."
    )

    if metrics and "ablation_study" in metrics:
        ablation = metrics["ablation_study"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Set A: Baseline Features")
            st.metric("CV MAE", f"{ablation['set_a_baseline']['cv_mae']:.4f}")
            st.caption(f"± {ablation['set_a_baseline']['cv_mae_std']:.4f}")

        with c2:
            st.markdown("#### Set B: + Engineered Features")
            st.metric("CV MAE", f"{ablation['set_b_engineered']['cv_mae']:.4f}")
            st.caption(f"± {ablation['set_b_engineered']['cv_mae_std']:.4f}")

        improvement = ablation.get("improvement", 0)
        decision = ablation.get("decision", "Unknown")

        st.divider()
        if improvement > 0:
            st.success(f"Improvement: {improvement:.4f} MAE → **{decision}**")
        else:
            st.warning(f"No improvement ({improvement:.4f} MAE) → **{decision}**")

        st.markdown(
            "**Why this matters for interviews:** An ablation study proves that "
            "feature engineering decisions are evidence-based, not arbitrary. "
            "Reporting a negative result (no improvement) demonstrates scientific rigour."
        )
    else:
        st.info("Ablation study results not available. Re-run training.")

# --- Tab 3: Fairness Analysis ---
with tab3:
    st.markdown("### Responsible AI: Subgroup Performance Analysis")
    st.markdown(
        "Model performance compared across demographic subgroups. "
        "The goal is transparency — identifying potential disparities, "
        "not declaring the model 'fair' or 'unfair'."
    )

    if metrics and "fairness" in metrics:
        fairness = metrics["fairness"]

        for task_type in ["regression", "classification"]:
            if task_type in fairness:
                st.markdown(f"#### {task_type.title()} Fairness")
                for attr, data in fairness[task_type].items():
                    st.markdown(f"**{attr}**")

                    subgroups = data.get("subgroups", {})
                    if subgroups:
                        rows = []
                        for group, m in subgroups.items():
                            row = {"Subgroup": group, "n": m.get("n", 0)}
                            if task_type == "regression":
                                row["MAE"] = m.get("mae", "N/A")
                                row["RMSE"] = m.get("rmse", "N/A")
                                row["Mean Residual"] = m.get("mean_residual", "N/A")
                            else:
                                row["Recall"] = m.get("recall", "N/A")
                                row["Precision"] = m.get("precision", "N/A")
                                row["F1"] = m.get("f1", "N/A")
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    alerts = data.get("alerts", [])
                    for alert in alerts:
                        st.warning(alert)

                st.divider()
    else:
        st.info("Fairness analysis not available. Re-run training.")

    st.markdown(
        "**Important:** This analysis uses a small, likely synthetic dataset with "
        "only Gender and Family_Income available. Real-world fairness evaluation "
        "would require additional demographic dimensions and domain expertise."
    )

# --- Tab 4: Model Card ---
with tab4:
    st.markdown("### Model Card")
    model_card_path = Path(__file__).resolve().parent.parent.parent / "docs" / "model_card.md"
    if model_card_path.exists():
        with open(model_card_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Model card not found at docs/model_card.md")

    # Model metadata
    if metrics:
        st.divider()
        st.markdown("### Training Metadata")
        meta_path = REPORTS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            st.json(meta)
