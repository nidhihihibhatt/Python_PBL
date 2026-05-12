# ============================================================
# STUDENT PERFORMANCE PREDICTION — CORRECTED & IMPROVED
# ============================================================
# Key fixes vs. original:
#  1. Imputation  fitted on TRAIN only, applied to TEST (no leakage)
#  2. IQR outlier bounds computed on TRAIN only
#  3. SelectFromModel placed INSIDE the Pipeline (no leakage)
#  4. 5-fold Cross-Validation for model comparison
#  5. GridSearchCV actually called on best candidates
#  6. Best model stored and reused for residual analysis
#  7. Train vs. Test R2 reported (overfitting detection)
#  8. random_state set on all stochastic models
#  9. Baseline includes MAE + RMSE
# 10. All plots have axis labels and proper figure sizes
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("StudentPerformanceFactors.csv")
print("Shape:", df.shape)
df.info()

# ============================================================
# 2. TRAIN-TEST SPLIT (BEFORE any imputation / outlier removal)
#    This is the correct order to prevent leakage.
# ============================================================
X_raw = df.drop("Exam_Score", axis=1)
y     = df["Exam_Score"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nTrain size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")

# ============================================================
# 3. MISSING VALUE IMPUTATION (fit on TRAIN, apply to TEST)
# ============================================================
IMPUTE_COLS = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]

# Learn modes from training data only
train_modes = {col: X_train_raw[col].mode()[0] for col in IMPUTE_COLS}

print("\nImputation modes (learned from train):", train_modes)

X_train_raw = X_train_raw.copy()
X_test_raw  = X_test_raw.copy()

for col, mode_val in train_modes.items():
    X_train_raw[col] = X_train_raw[col].fillna(mode_val)
    X_test_raw[col]  = X_test_raw[col].fillna(mode_val)

print("\nMissing values in train after imputation:")
print(X_train_raw.isnull().sum()[X_train_raw.isnull().sum() > 0])

# ============================================================
# 4. OUTLIER REMOVAL (bounds computed on TRAIN only)
# ============================================================
train_y_series = y_train.copy()

Q1  = train_y_series.quantile(0.25)
Q3  = train_y_series.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nOutlier bounds (from train): [{lower_bound:.2f}, {upper_bound:.2f}]")

# Remove outliers from TRAINING set only
train_mask   = (train_y_series >= lower_bound) & (train_y_series <= upper_bound)
X_train_raw  = X_train_raw[train_mask].reset_index(drop=True)
y_train      = y_train[train_mask].reset_index(drop=True)

print(f"Outliers removed from train: {(~train_mask).sum()}")
print(f"Train size after removal: {len(X_train_raw)}")

# NOTE: Do NOT remove outliers from the test set; that would distort evaluation.

# ============================================================
# 5. TARGET DISTRIBUTION (training set only)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y_train, bins=30, color="steelblue", edgecolor="white")
ax.set_title("Exam Score Distribution (Train Set)", fontsize=13)
ax.set_xlabel("Exam Score")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()

print(f"\nTrain Exam_Score — Mean: {y_train.mean():.2f} | "
      f"Median: {y_train.median():.2f} | Std: {y_train.std():.2f}")

# ============================================================
# 6. ENCODING (fit dummies on TRAIN, align TEST)
# ============================================================
X_train_enc = pd.get_dummies(X_train_raw, drop_first=True)
X_test_enc  = pd.get_dummies(X_test_raw,  drop_first=True)

# Align columns — test gets any missing columns as 0, drops any extras
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Convert bool → int without SettingWithCopyWarning
X_train_enc = X_train_enc.copy()
X_test_enc  = X_test_enc.copy()
for df_part in [X_train_enc, X_test_enc]:
    bool_cols = df_part.select_dtypes(include=["bool"]).columns.tolist()
    df_part[bool_cols] = df_part[bool_cols].astype(int)

print(f"\nEncoded train shape: {X_train_enc.shape}")

# ============================================================
# 7. EDA — CORRELATION HEATMAP (train only, top features)
# ============================================================
full_train_df = X_train_enc.copy()
full_train_df["Exam_Score"] = y_train.values

# Show only top 15 correlated features for readability
corr_with_target = full_train_df.corr()["Exam_Score"].abs().sort_values(ascending=False)
top_features = corr_with_target.head(16).index.tolist()  # 15 features + target

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    full_train_df[top_features].corr(),
    annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax
)
ax.set_title("Correlation Heatmap — Top 15 Features vs Exam Score", fontsize=13)
plt.tight_layout()
plt.show()

# ============================================================
# 8. FEATURE IMPORTANCE (train only, for reference / EDA)
# ============================================================
fs_rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
fs_rf.fit(X_train_enc, y_train)

feat_imp = pd.Series(fs_rf.feature_importances_, index=X_train_enc.columns)
top10    = feat_imp.sort_values(ascending=False).head(10)

print("\nTop 10 Important Features (Random Forest):")
print(top10)

fig, ax = plt.subplots(figsize=(9, 5))
top10.plot(kind="bar", ax=ax, color="darkorange", edgecolor="white")
ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13)
ax.set_xlabel("Feature")
ax.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ============================================================
# 9. BASELINE MODEL
# ============================================================
baseline_pred = np.full(len(y_test), y_train.mean())

baseline_r2   = r2_score(y_test, baseline_pred)
baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print("\n--- Baseline (Mean Predictor) ---")
print(f"  R²  : {baseline_r2:.4f}")
print(f"  MAE : {baseline_mae:.4f}")
print(f"  RMSE: {baseline_rmse:.4f}")

# ============================================================
# 10. MODEL TRAINING WITH 5-FOLD CV + TRAIN/TEST SCORES
#
#  FIX: SelectFromModel is now INSIDE the Pipeline.
#       The selector is re-fitted per fold, preventing leakage.
# ============================================================
models = {
    "Linear Regression":   LinearRegression(),
    "KNN":                 KNeighborsRegressor(),
    "SVM":                 SVR(),
    "Random Forest":       RandomForestRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting":   GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = []

print("\n" + "="*70)
print("MODEL COMPARISON (5-Fold CV on Train | Hold-out Test Evaluation)")
print("="*70)

for name, model in models.items():
    # Feature selector uses a lightweight RF so it's fast inside CV
    inner_rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("scaler",   MinMaxScaler()),
        ("selector", SelectFromModel(inner_rf, threshold="median")),
        ("model",    model)
    ])

    # 5-fold CV for unbiased model comparison
    cv_scores = cross_val_score(
        pipe, X_train_enc, y_train,
        cv=5, scoring="r2", n_jobs=-1
    )

    # Fit on full train, evaluate on hold-out test
    pipe.fit(X_train_enc, y_train)
    y_pred_train = pipe.predict(X_train_enc)
    y_pred_test  = pipe.predict(X_test_enc)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test,  y_pred_test)
    mae      = mean_absolute_error(y_test, y_pred_test)
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n{name}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Train R²      : {train_r2:.4f}  |  Test R²: {test_r2:.4f}  "
          f"{'⚠ Overfit?' if (train_r2 - test_r2) > 0.10 else '✓ OK'}")
    print(f"  MAE           : {mae:.4f}")
    print(f"  RMSE          : {rmse:.4f}")

    results.append({
        "Model":            name,
        "CV_R2_mean":       cv_scores.mean(),
        "CV_R2_std":        cv_scores.std(),
        "Train_R2":         train_r2,
        "Test_R2":          test_r2,
        "MAE":              mae,
        "RMSE":             rmse,
        "fitted_pipeline":  pipe,
    })

results_df = pd.DataFrame(results).sort_values("CV_R2_mean", ascending=False)

print("\n--- Summary Table ---")
print(results_df[["Model","CV_R2_mean","CV_R2_std","Train_R2","Test_R2","MAE","RMSE"]]
      .to_string(index=False))

# ============================================================
# 11. HYPERPARAMETER TUNING ON TOP-2 MODELS (GridSearchCV)
# ============================================================
top2_names = results_df.head(2)["Model"].tolist()
print(f"\nRunning GridSearchCV on: {top2_names}")

param_grids = {
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth":    [None, 10, 20],
        "model__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators":  [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth":     [3, 5],
    },
    "Linear Regression": {},   # no hyperparameters to tune
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights":     ["uniform", "distance"],
    },
    "SVM": {
        "model__C":      [0.1, 1, 10],
        "model__kernel": ["rbf", "linear"],
    },
}

tuned_pipelines = {}

for name in top2_names:
    grid = param_grids.get(name, {})
    matching_row = results_df[results_df["Model"] == name].iloc[0]
    base_pipe    = matching_row["fitted_pipeline"]   # already configured pipeline

    if not grid:
        print(f"\n{name}: no hyperparameters to tune — using default.")
        tuned_pipelines[name] = base_pipe
        continue

    gs = GridSearchCV(
        base_pipe, grid,
        cv=5, scoring="r2",
        n_jobs=-1, refit=True
    )
    gs.fit(X_train_enc, y_train)

    print(f"\n{name} — Best params : {gs.best_params_}")
    print(f"{name} — CV R² (best): {gs.best_score_:.4f}")
    tuned_pipelines[name] = gs.best_estimator_

# ============================================================
# 12. SELECT OVERALL BEST MODEL
# ============================================================
best_name = results_df.iloc[0]["Model"]
best_pipe  = tuned_pipelines.get(best_name,
             results_df.iloc[0]["fitted_pipeline"])

print(f"\n🏆 Best Model: {best_name}")

# ============================================================
# 13. RESIDUAL ANALYSIS — using the BEST model (not last loop iter)
# ============================================================
best_pipe.fit(X_train_enc, y_train)
y_pred_best = best_pipe.predict(X_test_enc)

residuals = y_test - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f"Residual Analysis — {best_name}", fontsize=14)

# Residuals vs Fitted
axes[0].scatter(y_pred_best, residuals, alpha=0.4, color="steelblue", edgecolors="none")
axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
axes[0].set_title("Residuals vs Fitted")
axes[0].set_xlabel("Predicted Exam Score")
axes[0].set_ylabel("Residuals")

# Actual vs Predicted
axes[1].scatter(y_test, y_pred_best, alpha=0.4, color="seagreen", edgecolors="none")
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
axes[1].set_title("Actual vs Predicted")
axes[1].set_xlabel("Actual Exam Score")
axes[1].set_ylabel("Predicted Exam Score")

# Residual Distribution
axes[2].hist(residuals, bins=30, color="coral", edgecolor="white")
axes[2].set_title("Residual Distribution")
axes[2].set_xlabel("Residual")
axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Final metrics for best model
final_r2   = r2_score(y_test, y_pred_best)
final_mae  = mean_absolute_error(y_test, y_pred_best)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"\n--- Final Test Metrics [{best_name}] ---")
print(f"  R²  : {final_r2:.4f}")
print(f"  MAE : {final_mae:.4f}")
print(f"  RMSE: {final_rmse:.4f}")

print("\n✅ Analysis complete.")
