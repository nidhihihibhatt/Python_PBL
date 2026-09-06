# AcademIQ — Original Project Audit (Phase 0)

> **Audit date:** 2026-08-15
> **Original project location:** `D:\AI ML PBL`
> **Status:** All original files preserved and unmodified.

---

## 1. Original Project Structure

| File | Size | Description |
|------|------|-------------|
| `project.ipynb` | 970 KB | Monolithic notebook with all EDA, preprocessing, training, evaluation |
| `predict.py` | 1 KB | CLI prediction script (4 features, inconsistent preprocessing) |
| `predict.ipynb` | 3 KB | Notebook version of predict.py |
| `StudentPerformanceFactors.csv` | 642 KB | Raw dataset (6,607 × 20) |
| `student_performance_clean.csv` | 139 KB | Processed/encoded version of the dataset |
| `regression_model.pkl` | 27.9 MB | Saved regression Pipeline |
| `best_model.pkl` | 27.9 MB | Duplicate of regression_model.pkl |
| `classification_model.pkl` | 6.2 MB | Saved classification Pipeline |
| `columns.pkl` | 1 KB | Saved column names (27 encoded features) |

**Issues:** No README, no requirements.txt, no .gitignore, no tests, no documentation.
Two identical 27.9 MB files (`best_model.pkl` = `regression_model.pkl`).

---

## 2. Dataset Summary

| Property | Value |
|----------|-------|
| Rows | 6,607 |
| Columns | 20 (7 numeric, 13 categorical) |
| Target | `Exam_Score` (integer) |
| Missing values | 235 total across 3 columns |
| Duplicates | 0 |
| Apparent origin | Kaggle (appears synthetic) |

### Missing Values

| Column | Missing Count | % |
|--------|--------------|---|
| `Teacher_Quality` | 78 | 1.2% |
| `Parental_Education_Level` | 90 | 1.4% |
| `Distance_from_Home` | 67 | 1.0% |

### Target Distribution (`Exam_Score`)

| Statistic | Value |
|-----------|-------|
| Mean | 67.24 |
| Median | 67.0 |
| Std | 3.89 |
| Min | 55 |
| Max | 101 |
| Q25 | 65.0 |
| Q75 | 69.0 |
| IQR | 4.0 |

> **Critical observation:** The score distribution is very tight (IQR = 4 points). This fundamentally limits regression accuracy. An MAE of ~2 may approach the noise floor.

### Threshold Analysis

| Threshold | Students Below | Percentage |
|-----------|---------------|------------|
| < 60 | 68 | 1.0% |
| < 63 | 580 | 8.8% |
| < 65 | 1,452 | 22.0% |
| < 67 | 2,882 | 43.6% |

### Feature Correlations with `Exam_Score`

| Feature | Correlation |
|---------|------------|
| Attendance | 0.5811 |
| Hours_Studied | 0.4455 |
| Previous_Scores | 0.1751 |
| Tutoring_Sessions | 0.1565 |
| Physical_Activity | 0.0278 |
| Sleep_Hours | -0.0170 |

### Subgroup Analysis

| Subgroup | Mean Score | Count |
|----------|-----------|-------|
| Gender: Male | 67.23 | 3,814 |
| Gender: Female | 67.24 | 2,793 |
| Income: Low | 66.85 | 2,672 |
| Income: Medium | 67.33 | 2,666 |
| Income: High | 67.84 | 1,269 |

---

## 3. Original Preprocessing Strategy

The notebook (59 code cells) follows this pipeline:

1. **Missing value imputation:** Mode-fill for 3 categorical columns
2. **Outlier capping:** IQR-based capping on `Hours_Studied`, `Attendance`, `Previous_Scores`, and **`Exam_Score`** (target — problematic)
3. **Target creation:** `Pass_Fail = (Exam_Score >= 65).astype(int)`
4. **Ordinal encoding:** Low→0, Medium→1, High→2 for ordered categoricals
5. **One-hot encoding:** `pd.get_dummies(drop_first=True)` for nominal categoricals
6. **Feature matrix:** `X = df_encoded.drop(["Exam_Score", "Pass_Fail"], axis=1)` — correctly excludes both targets
7. **Train/test split:** 80/20, `random_state=42`, stratified on `Pass_Fail`
8. **Feature selection:** `SelectFromModel(RandomForestRegressor, threshold="median")`
9. **Scaling:** `MinMaxScaler`

---

## 4. What Was Good (Preserve These Ideas)

| Strength | Detail |
|----------|--------|
| **No target leakage in feature matrix** | Both `Exam_Score` and `Pass_Fail` correctly dropped from X |
| **sklearn Pipeline usage** | `Pipeline(MinMaxScaler → SelectFromModel → Model)` — correct pattern |
| **Ordinal encoding** for ordered categoricals | Preserves Low < Medium < High relationship |
| **Stratified train/test split** | Maintains class distribution |
| **Baseline model** (DummyRegressor) | Shows understanding that models must beat trivial strategies |
| **5-fold cross-validation** with GridSearchCV | Proper model selection methodology |
| **Dual-task framing** | Both regression and classification attempted |
| **Comprehensive evaluation** | R², MAE, RMSE for regression; Accuracy, Precision, Recall, F1, ROC-AUC for classification |

---

## 5. What Was Wrong (Must Fix)

### 5.1 Critical: Prediction Script is Completely Broken

| Training Pipeline | predict.py |
|-------------------|-----------|
| 27 encoded features (ordinal + one-hot) | 4 raw features only |
| `OrdinalEncoder` for Low/Medium/High | `pd.get_dummies` (different encoding!) |
| `MinMaxScaler` (inside Pipeline) | No scaling |
| `SelectFromModel` (inside Pipeline) | No feature selection |
| Pass/Fail threshold: 65 | Pass/Fail threshold: 40 |
| Classification model saved | Classification model never loaded/used |

**Impact:** Every prediction from `predict.py` is unreliable.

### 5.2 Critical: Target Variable Capped

The notebook applies IQR-based outlier capping to `Exam_Score` (the target). This distorts the learning signal — extreme scorers are the most important students for an early-warning system.

### 5.3 Critical: Only One Classification Model

Only `LogisticRegression` used for classification. No model comparison, no hyperparameter tuning, no GridSearchCV.

### 5.4 High: Limited Regression Model Comparison

Only 3 models compared (LinearRegression, KNN, SVR). Missing ensemble methods (RF, Gradient Boosting) that typically dominate tabular data.

### 5.5 High: No Explainability

No SHAP, no feature explanations. Only RF feature importance (not connected to the final model).

### 5.6 High: No Software Engineering

No tests, no logging, no error handling, no documentation, no README, no requirements.txt, no .gitignore. Everything in one monolithic notebook.

### 5.7 Medium: Duplicate Model Files

`best_model.pkl` and `regression_model.pkl` are identical (27.9 MB each).

---

## 6. Original Model Results

### Saved Models (inspected from .pkl files)

| Model | Pipeline |
|-------|----------|
| **Regression** | `MinMaxScaler → SelectFromModel(RF, threshold="median") → LinearRegression` |
| **Classification** | `MinMaxScaler → SelectFromModel(RF, threshold="median") → LogisticRegression(max_iter=1000)` |

> **Note:** The best regression model selected by GridSearchCV was `LinearRegression`, not RF or SVR. This is recorded from the actual saved Pipeline object.

### Feature Space

The saved `columns.pkl` contains 27 features after encoding:
- 6 numeric features (Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Tutoring_Sessions, Physical_Activity)
- 21 encoded categorical features (one-hot with drop_first + ordinal)

---

## 7. What Will Be Improved

| Area | Current | Target |
|------|---------|--------|
| Architecture | Single monolithic notebook | Modular Python package (`src/`) with clear responsibilities |
| Inference | Broken `predict.py` with inconsistent preprocessing | Single Pipeline artifact; raw input → prediction |
| Regression models | 3 models (LR, KNN, SVR) | 5+ models including Ridge, Lasso, RF, HistGBM |
| Classification | 1 model (LogReg), no CV | 4+ models with CV, threshold analysis, calibration |
| Target handling | `Exam_Score` capped by IQR | Uncapped target; outlier treatment on features only |
| Feature engineering | None | Investigated with mandatory ablation study |
| Explainability | None | SHAP (global + local) |
| Recommendations | None | Rule-based engine tied to SHAP contributions |
| Uncertainty | None | Quantile regression prediction intervals |
| Fairness | None | Subgroup analysis (Gender, Family_Income) |
| Testing | None | pytest suite (≥25 tests) |
| Documentation | None | README, model card, interview prep |
| Dashboard | None | Professional 4-section Streamlit app |
| Deployment | None | Streamlit Community Cloud |

---

## 8. Baseline Metrics (Before Rebuild)

These will be compared against the new system's metrics.

| Metric | Original Value | Notes |
|--------|---------------|-------|
| Best regression model | LinearRegression | Selected by GridSearchCV on R² |
| Regression pipeline | MinMaxScaler → SelectFromModel → LR | |
| Classification model | LogisticRegression | No model comparison |
| Classification pipeline | MinMaxScaler → SelectFromModel → LogReg | |
| Risk threshold | 65 (hardcoded as Pass_Fail) | |
| Feature count (after encoding) | 27 | |
| Cross-validation | 5-fold (regression only) | Classification not cross-validated |
| Target modification | IQR capping on Exam_Score | Will be removed |

> **Exact metric values (R², MAE, RMSE, Accuracy, F1, etc.) are recorded in notebook outputs but cannot be reliably extracted from the .ipynb without re-running the notebook. These will be reproduced during the rebuild as the "original baseline" experiment.**

---

*This audit confirms the original project has a sound core methodology (Pipelines, stratified split, CV) but critical gaps in engineering, inference, model diversity, explainability, and product quality. The rebuild preserves the good ideas while fixing all identified issues.*
