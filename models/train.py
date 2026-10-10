"""
models/train.py
---------------
Module 3 — Forecasting Engine

Trains three models on the clean feature-engineered data:
  1. AR Model      — AutoRegressive linear model using lag features
  2. Random Forest — sklearn RandomForestRegressor
  3. Gradient Boosting — sklearn GradientBoostingRegressor (XGBoost-equivalent)

Then runs the A/B comparison loop (models/evaluate.py) to select the best.

Optional XGBoost
----------------
If ``xgboost`` is installed it will be used automatically in place of
GradientBoostingRegressor.  Install with:  pip install xgboost
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ML_FEATURES, AR_FEATURES, TEST_SPLIT, RANDOM_STATE,
    N_ESTIMATORS, PROCESSED_DIR, DB_PATH
)
from models.evaluate import compare_models, save_feature_importance
from utils.db import save_csv

# ── Optional XGBoost ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor as _XGB
    def _make_booster():
        return _XGB(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
                    verbosity=0)
    BOOSTER_NAME = "XGBoost"
except ImportError:
    def _make_booster():
        return GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
        )
    BOOSTER_NAME = "Gradient Boosting"


# ── Per-store time-based split helper ─────────────────────────────────────────

def _time_split(df: pd.DataFrame, test_split: float = TEST_SPLIT):
    """Split a sorted DataFrame into train / test preserving temporal order."""
    n = len(df)
    split = int(n * (1 - test_split))
    return df.iloc[:split], df.iloc[split:]


# ── Individual model trainers ─────────────────────────────────────────────────

def train_ar(X_train: pd.DataFrame, y_train: pd.Series):
    """
    AutoRegressive model — LinearRegression on lag + rolling features.
    Equivalent to AR(p) in the ARIMA family without explicit differencing.
    Differencing is implicitly handled through the lag feature construction.
    """
    model = LinearRegression()
    model.fit(X_train[AR_FEATURES], y_train)
    return model


def train_rf(X_train: pd.DataFrame, y_train: pd.Series):
    """Random Forest Regressor with time-aware hyperparameters."""
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=12,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train[ML_FEATURES], y_train)
    return model


def train_booster(X_train: pd.DataFrame, y_train: pd.Series):
    """Gradient Boosting / XGBoost Regressor."""
    model = _make_booster()
    model.fit(X_train[ML_FEATURES], y_train)
    return model


# ── Main entry point ──────────────────────────────────────────────────────────

def train_all(df_clean: pd.DataFrame) -> dict:
    """
    Full training pipeline:

    1. Aggregate all stores for a single comparable evaluation set.
    2. Train AR, Random Forest, and Gradient Boosting / XGBoost.
    3. Run A/B comparison → select best model.
    4. Save feature importances for the best tree model.
    5. Compute residuals for confidence-interval generation.
    6. Persist trained models to disk (pickle).

    Returns
    -------
    {
        "models":       {name: (model, feature_list)},
        "best_name":    str,
        "scores_df":    pd.DataFrame,
        "residuals":    np.ndarray,
        "X_test":       pd.DataFrame,
        "y_test":       pd.Series,
        "X_train":      pd.DataFrame,
        "y_train":      pd.Series,
        "feat_imp":     pd.DataFrame | None,
    }
    """
    print("\n── Module 3: Forecasting Engine ─────────────────────────────────")

    # ── use the aggregated (all-store) daily series for fair comparison ───
    daily = (
        df_clean.groupby("Date", as_index=False)
                .agg(Sales=("Sales", "sum"),
                     **{f: (f, "mean") for f in ML_FEATURES})
    ).sort_values("Date").reset_index(drop=True)

    train_df, test_df = _time_split(daily)

    X_train = train_df[ML_FEATURES]
    y_train = train_df["Sales"]
    X_test  = test_df[ML_FEATURES]
    y_test  = test_df["Sales"]

    print(f"   Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    # ── train ─────────────────────────────────────────────────────────────
    print("\n   Training AR Model …")
    ar_model = train_ar(X_train, y_train)

    print(f"   Training Random Forest ({N_ESTIMATORS} trees) …")
    rf_model = train_rf(X_train, y_train)

    print(f"   Training {BOOSTER_NAME} ({N_ESTIMATORS} trees) …")
    gb_model = train_booster(X_train, y_train)

    models = {
        "AR Model":    (ar_model,  AR_FEATURES),
        "Random Forest": (rf_model, ML_FEATURES),
        BOOSTER_NAME:  (gb_model,  ML_FEATURES),
    }

    # ── compare ───────────────────────────────────────────────────────────
    print("\n   A/B Model Comparison:")
    scores_df, best_name = compare_models(models, X_test, y_test)

    # ── feature importance ────────────────────────────────────────────────
    best_model, best_feats = models[best_name]
    feat_imp = None
    if hasattr(best_model, "feature_importances_"):
        print(f"\n   Feature Importance ({best_name}):")
        feat_imp = save_feature_importance(best_model, best_feats, best_name)

    # ── residuals for confidence intervals ────────────────────────────────
    best_train_preds = best_model.predict(X_train[best_feats])
    residuals = y_train.values - best_train_preds

    # ── persist models ────────────────────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    models_path = os.path.join(PROCESSED_DIR, "trained_models.pkl")
    with open(models_path, "wb") as f:
        pickle.dump({"models": models, "best_name": best_name}, f)
    print(f"\n   Models saved → {models_path}")

    return {
        "models":    models,
        "best_name": best_name,
        "scores_df": scores_df,
        "residuals": residuals,
        "X_test":    X_test,
        "y_test":    y_test,
        "X_train":   X_train,
        "y_train":   y_train,
        "feat_imp":  feat_imp,
    }
