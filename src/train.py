"""
Train and compare models for Uber fare prediction.

Usage:
    python src/train.py

Requires xgboost + lightgbm (see requirements.txt) for the full comparison.
If they're not installed, falls back to sklearn's GradientBoostingRegressor /
RandomForestRegressor so the pipeline still runs end-to-end.

v3 changes (fixing the Linear Regression R2=-157006 blowup from v2):
- The linear baseline is now Ridge (L2-regularized) inside a
  StandardScaler pipeline, instead of bare LinearRegression. The old
  feature set had near-collinear columns (raw hour vs is_rush_hour/
  is_night, raw weekday vs is_weekend), which gave OLS wildly unstable
  coefficients. Ridge's penalty keeps coefficients bounded; scaling
  keeps every feature on a comparable footing so the penalty is fair.
- features.py now uses cyclical (sin/cos) encodings for hour/weekday/
  month instead of raw integers, plus log1p(distance_km), which removes
  that collinearity at the source and gives the linear model a feature
  that's closer to linear against log-fare.
- Every LogTargetRegressor now gets a clip_log_range computed from the
  training target's log range. This is the actual fix for the negative-
  157000 R2: without it, a single bad log-space prediction (from ANY
  model, not just linear) turns into an astronomical dollar value after
  expm1(), which is what was destroying RMSE/R2 before.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS
from model_wrapper import LogTargetRegressor, WeightedEnsembleRegressor

DATA_PATH_REAL = "data/uber.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# NYC rough bounding box. Rows outside this are almost always GPS errors
# (0,0 "null island", coordinates in the wrong hemisphere, etc) — on the
# real Kaggle dataset these are common enough to badly distort distance_km.
NYC_LAT_RANGE = (40.5, 41.5)
NYC_LON_RANGE = (-74.5, -72.5)


def load_data():
    df = pd.read_csv(DATA_PATH_REAL)
    df = df.dropna()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 300)]
    df = df[df["passenger_count"].between(1, 6)]

    df = df[df["pickup_latitude"].between(*NYC_LAT_RANGE)]
    df = df[df["dropoff_latitude"].between(*NYC_LAT_RANGE)]
    df = df[df["pickup_longitude"].between(*NYC_LON_RANGE)]
    df = df[df["dropoff_longitude"].between(*NYC_LON_RANGE)]

    return df


def evaluate(name, y_true, y_pred, results):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[name] = {"r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)}
    print(f"{name:28s}  R2={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows from {DATA_PATH_REAL}")

    X = build_features(df)
    y = df["fare_amount"].values

    keep = X["distance_km"] > 0.05
    X, y = X[keep], y[keep]
    print(f"After distance/coordinate cleaning: {len(X)} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train on log1p(fare) for every model — inverted back to dollars via
    # LogTargetRegressor so `results` and the saved model both read in
    # plain dollars.
    y_train_log = np.log1p(y_train)

    # Safety clip range for inverting log-space predictions back to
    # dollars. A little wider than the observed training range so we
    # don't clip legitimate high fares, but tight enough that a single
    # bad prediction can't turn into an absurd dollar value. This is the
    # actual fix for the R2=-157006 blowup seen before.
    clip_range = (float(y_train_log.min()) - 0.5, float(y_train_log.max()) + 0.5)
    print(f"Log-space clip range: {clip_range}")

    results = {}
    candidates = {}  # name -> fitted, dollar-unit-predicting model

    # --- Baseline: Ridge (regularized) + feature scaling ---
    # Plain LinearRegression on this feature set had near-collinear
    # columns, which blew up its coefficients and, combined with
    # expm1(), produced some absurd dollar predictions. Ridge's L2
    # penalty keeps coefficients bounded; StandardScaler keeps every
    # feature on a comparable scale so the penalty applies fairly.
    lin_inner = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    lin_inner.fit(X_train, y_train_log)
    lin = LogTargetRegressor(lin_inner, clip_log_range=clip_range)
    evaluate("Linear Regression (Ridge)", y_test, lin.predict(X_test), results)
    candidates["Linear Regression (Ridge)"] = lin

    # --- XGBoost (falls back to GradientBoosting if not installed) ---
    try:
        from xgboost import XGBRegressor
        xgb_inner = XGBRegressor(
            n_estimators=700, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1,
        )
        xgb_inner.fit(X_train, y_train_log)
        xgb = LogTargetRegressor(xgb_inner, clip_log_range=clip_range)
        evaluate("XGBoost", y_test, xgb.predict(X_test), results)
        candidates["XGBoost"] = xgb
        boost_a, boost_a_name = xgb, "XGBoost"
    except ImportError:
        print("[xgboost not installed - falling back to GradientBoostingRegressor]")
        from sklearn.ensemble import GradientBoostingRegressor
        gbr_inner = GradientBoostingRegressor(n_estimators=400, max_depth=4, random_state=42)
        gbr_inner.fit(X_train, y_train_log)
        gbr = LogTargetRegressor(gbr_inner, clip_log_range=clip_range)
        evaluate("GradientBoosting (fallback)", y_test, gbr.predict(X_test), results)
        candidates["GradientBoosting (fallback)"] = gbr
        boost_a, boost_a_name = gbr, "GradientBoosting (fallback)"

    # --- LightGBM (falls back to RandomForest if not installed) ---
    try:
        from lightgbm import LGBMRegressor
        lgbm_inner = LGBMRegressor(
            n_estimators=800, max_depth=-1, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        lgbm_inner.fit(X_train, y_train_log)
        lgbm = LogTargetRegressor(lgbm_inner, clip_log_range=clip_range)
        evaluate("LightGBM", y_test, lgbm.predict(X_test), results)
        candidates["LightGBM"] = lgbm
        boost_b, boost_b_name = lgbm, "LightGBM"
    except ImportError:
        print("[lightgbm not installed - falling back to RandomForestRegressor]")
        from sklearn.ensemble import RandomForestRegressor
        rf_inner = RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42, n_jobs=-1)
        rf_inner.fit(X_train, y_train_log)
        rf = LogTargetRegressor(rf_inner, clip_log_range=clip_range)
        evaluate("RandomForest (fallback)", y_test, rf.predict(X_test), results)
        candidates["RandomForest (fallback)"] = rf
        boost_b, boost_b_name = rf, "RandomForest (fallback)"

    # --- Weighted ensemble of the two boosted models ---
    ensemble = WeightedEnsembleRegressor([boost_a, boost_b], [0.5, 0.5])
    ensemble_name = f"Ensemble ({boost_a_name} + {boost_b_name})"
    evaluate(ensemble_name, y_test, ensemble.predict(X_test), results)
    candidates[ensemble_name] = ensemble

    # --- Pick whichever candidate actually scored best ---
    best_name = max(results, key=lambda n: results[n]["r2"])
    best_model = candidates[best_name]
    print(f"\nBest model: {best_name} (R2={results[best_name]['r2']})")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nSaved best model -> {MODELS_DIR}/best_model.joblib")
    print(f"Saved metrics -> {MODELS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
