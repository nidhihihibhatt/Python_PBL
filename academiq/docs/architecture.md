# AcademIQ — System Architecture

## Overview

AcademIQ follows a modular architecture where each component has a single, clear responsibility. The system separates data handling, model training, inference, and presentation.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                       │
│   Streamlit Dashboard (app/)                                    │
│   ├── Overview (KPIs, distributions)                            │
│   ├── Student Analysis (predict, explain, recommend)            │
│   ├── Risk & Early Warning (risk distribution, threshold)       │
│   └── Model & Methodology (comparison, fairness, model card)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                        INFERENCE LAYER                          │
│   PredictionService (src/inference/service.py)                  │
│   ├── Loads serialised Pipeline → raw input → prediction        │
│   ├── Computes risk from classifier probabilities               │
│   ├── Generates SHAP explanations via ShapExplainer             │
│   └── Generates recommendations via engine                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ loads
┌──────────────────────────▼──────────────────────────────────────┐
│                        MODEL ARTIFACTS                          │
│   models/                                                       │
│   ├── regression_pipeline.joblib (complete Pipeline)            │
│   ├── classification_pipeline.joblib                            │
│   ├── quantile_lo_pipeline.joblib (10th percentile)             │
│   ├── quantile_hi_pipeline.joblib (90th percentile)             │
│   └── shap_background.joblib (background data for SHAP)        │
└──────────────────────────▲──────────────────────────────────────┘
                           │ produces
┌──────────────────────────┴──────────────────────────────────────┐
│                        TRAINING LAYER                           │
│   train.py                                                      │
│   ├── Data loading + validation (src/data/)                     │
│   ├── Ablation study (baseline vs engineered features)          │
│   ├── Regression model comparison (5 models × GridSearchCV)     │
│   ├── Classification comparison (4 models × GridSearchCV)       │
│   ├── Probability Calibration (CalibratedClassifierCV)          │
│   ├── Dynamic Threshold Selection (CV precision/recall)         │
│   ├── Regression-vs-classifier experiment                       │
│   ├── Quantile regression (Empirical CV tuned for 80% coverage) │
│   ├── SHAP explainability                                       │
│   └── Fairness analysis                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                        CORE ML LIBRARY                          │
│   src/                                                          │
│   ├── config.py              (YAML config loader)               │
│   ├── data/loader.py         (data loading + validation)        │
│   ├── preprocessing/                                            │
│   │   ├── pipeline.py        (Pipeline builder)                 │
│   │   └── transformers.py    (FeatureEngineer transformer)      │
│   ├── models/                                                   │
│   │   ├── train.py           (training orchestration)           │
│   │   └── evaluate.py        (metrics computation)              │
│   ├── explainability/        (SHAP wrapper)                     │
│   ├── recommendations/       (rule-based engine)                │
│   └── fairness/              (subgroup analysis)                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Single Pipeline Artifact
The entire preprocessing + model chain is saved as one `.joblib` file. This guarantees training/inference consistency — the most critical technical requirement.

### 2. Configuration-Driven
All feature definitions, model hyperparameters, thresholds, and recommendation rules are in YAML files under `configs/`. This separates concerns: changing a threshold doesn't require editing Python code.

### 3. SHAP-Gated Recommendations
Recommendations require both a feature-level condition AND SHAP confirmation. This prevents generic advice that doesn't align with the model's actual assessment.

### 4. Modular Testing
Each module (data validation, preprocessing, recommendations, edge cases) has a dedicated test file. Tests use shared fixtures from `conftest.py`.

### 5. Ablation-Validated Features
Feature engineering is not assumed beneficial — it's validated via an ablation study. The training script automatically decides whether to include engineered features based on measured CV improvement.
