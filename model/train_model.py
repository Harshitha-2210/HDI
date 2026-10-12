"""
train_model.py — Preprocessing, model training, and serialization for the
HDI Predictor.

Steps:
    1. Load dataset (hdi_dataset.csv)
    2. Split into train / test BY COUNTRY (so no country appears on both sides)
    3. Impute missing values with the training-set median (no test leakage)
    4. Compare a scaled Linear Regression model against a Random Forest
    5. Evaluate with R-squared, MAE, RMSE on the held-out countries, plus
       5-fold country-grouped cross-validation
    6. Serialize the best model + scaler for reuse in Flask

Why split by country?
    The dataset holds ~23 yearly rows per country. A plain random split puts
    Norway-2014 in training and Norway-2015 in the test set, which lets the
    model "recognise" the country and inflates the score (R2 ~0.99). Holding
    out whole countries measures what the app actually does: predict HDI for
    a country the model has never seen (R2 ~0.92).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hdi_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "hdi_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

FEATURES = [
    "electricity_access",
    "internet_users",
    "water_access",
    "health_exp_gdp",
    "female_labor_force",
    "unemployment",
    "infant_mortality",
    "fertility_rate",
    "urban_pop",
    "co2_per_capita",
]
TARGET = "hdi"
GROUP = "country"
SEED = 42

# Random Forest size is kept moderate so hdi_model.pkl stays well under
# GitHub's 50 MB file warning.
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1)


def make_linear():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())


def make_forest():
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**RF_PARAMS))


# ---------- 1. Load ----------
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns, "
      f"{df[GROUP].nunique()} countries")

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

# ---------- 2. Train/test split, grouped by country ----------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
print(f"Train: {len(X_train)} rows / {groups.iloc[train_idx].nunique()} countries | "
      f"Test: {len(X_test)} rows / {groups.iloc[test_idx].nunique()} countries")

# ---------- 3/4. Compare Linear Regression and Random Forest ----------
linear = make_linear().fit(X_train, y_train)
linear_r2 = r2_score(y_test, linear.predict(X_test))

forest = make_forest().fit(X_train, y_train)
forest_r2 = r2_score(y_test, forest.predict(X_test))
print(f"Held-out-country R2  ->  Linear: {linear_r2:.4f} | Random Forest: {forest_r2:.4f}")

use_forest = forest_r2 >= linear_r2
model_name = "RandomForestRegressor" if use_forest else "LinearRegression"
pipe = forest if use_forest else linear

# Unpack the fitted pipeline so app.py can keep using model.pkl + scaler.pkl.
imputer = pipe.named_steps["simpleimputer"]
scaler = pipe.named_steps.get("standardscaler")
if use_forest:
    model = pipe.named_steps["randomforestregressor"]
    # Forest was trained on un-scaled data; save a harmless fitted scaler anyway
    # so the artifact set is always complete.
    scaler = StandardScaler().fit(imputer.transform(X_train))
    feature_importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
    uses_scaler = False
else:
    model = pipe.named_steps["linearregression"]
    feature_importances = dict(zip(FEATURES, np.abs(model.coef_).tolist()))
    uses_scaler = True

# ---------- 5. Evaluate ----------
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
train_r2 = r2_score(y_train, pipe.predict(X_train))

# 5-fold cross-validation, again grouped by country, for a steadier estimate.
cv_r2, cv_mae = [], []
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    fold = (make_forest() if use_forest else make_linear()).fit(X.iloc[tr], y.iloc[tr])
    p = fold.predict(X.iloc[te])
    cv_r2.append(r2_score(y.iloc[te], p))
    cv_mae.append(mean_absolute_error(y.iloc[te], p))

metrics = {
    "best_model": model_name,
    "evaluation": "Held-out countries (grouped split, no country in both train and test)",
    "r2_test": round(r2, 4),
    "r2_train": round(train_r2, 4),
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
    "cv_r2_mean": round(float(np.mean(cv_r2)), 4),
    "cv_r2_std": round(float(np.std(cv_r2)), 4),
    "cv_mae_mean": round(float(np.mean(cv_mae)), 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_train_countries": int(groups.iloc[train_idx].nunique()),
    "n_test_countries": int(groups.iloc[test_idx].nunique()),
    "intercept": round(float(model.intercept_), 4) if hasattr(model, "intercept_") else None,
    "feature_importances": {k: round(float(v), 4) for k, v in feature_importances.items()},
    "features": FEATURES,
    "target": TARGET,
    "uses_scaler": uses_scaler,
    # Training medians, in case you ever want to fill blanks in the app.
    "feature_medians": {k: round(float(v), 4) for k, v in zip(FEATURES, imputer.statistics_)},
}

print("Model performance:")
print(json.dumps(metrics, indent=2))

# ---------- 6. Serialize ----------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
print(f"Saved scaler to {SCALER_PATH}")
print(f"Saved metrics to {METRICS_PATH}")
