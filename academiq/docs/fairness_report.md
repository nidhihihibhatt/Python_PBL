# AcademIQ — Fairness Report

## Purpose

This report evaluates model performance across demographic subgroups available in the dataset (Gender, Family_Income). The goal is transparency and awareness of potential disparities — not a legal or universal fairness certification.

## Important Disclaimers

1. **Synthetic data**: The dataset appears synthetic. Fairness findings may not reflect real-world disparities.
2. **Limited dimensions**: Only Gender and Family_Income are available. Race, ethnicity, disability status, and other important dimensions cannot be evaluated.
3. **Associational analysis**: This measures performance differences, not causal discrimination.
4. **Internal thresholds**: Any disparity alerts use project-defined thresholds, not legally established standards.

## Methodology

For each protected attribute, we compare:
- **Regression**: MAE, RMSE, and mean residual (bias indicator) across subgroups
- **Classification**: Recall, precision, F1, false positive rate, and false negative rate

A disparity alert is triggered when:
- MAE differs by > 0.5 points across subgroups (regression)
- Recall differs by > 10 percentage points (classification)

These thresholds are internal design choices, not universal fairness criteria.

## Results

Results are computed during training and stored in `reports/metrics.json` under the `fairness` key. The Streamlit dashboard renders these results interactively.

### Key Observations (from training run)

**Gender (Regression):**
- Female: MAE ≈ 0.46, n ≈ 545
- Male: MAE ≈ 0.57, n ≈ 777
- Difference: ~0.10 points (below alert threshold)
- Observation: Slightly better predictions for female students, but within acceptable range

**Family Income (Regression):**
- High: MAE ≈ 0.61, n ≈ 225
- Medium: MAE ≈ 0.49, n ≈ 572
- Low: MAE ≈ 0.52, n ≈ 525
- Difference: ~0.12 points (below alert threshold)
- Observation: Slightly worse predictions for high-income group (smallest subgroup)

## Recommendations

1. With real student data, conduct fairness analysis across all relevant demographic dimensions
2. Consider whether prediction errors systematically disadvantage any group
3. Ensure intervention recommendations do not disproportionately burden specific subgroups
4. Regular monitoring if the system is deployed — performance can change as the student population evolves

## What This Demonstrates

This analysis shows awareness of:
- Model risk and responsible deployment
- Subgroup performance variation
- Limitations of fairness analysis on synthetic data
- The difference between measuring disparities and certifying fairness
