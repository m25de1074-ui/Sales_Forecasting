"""
models/predict.py
-----------------
Module 3 (continued) — Forecast with Confidence Intervals  (NEW addition)

forecast_with_ci() uses the best-selected model to generate a future
forecast for the next FORECAST_HORIZON days and computes upper / lower
95 % confidence bounds from the model's training residuals.

Why residual-based CI?
----------------------
After fitting, the model's training residuals follow an approximately
normal distribution.  Scaling the standard deviation by z=1.96 gives
a 95 % prediction interval — simple, interpretable, and parameter-free.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ML_FEATURES, AR_FEATURES, FORECAST_HORIZON, CI_Z, PROCESSED_DIR
)
from utils.db import save_csv


def _build_future_features(
    df_clean: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Construct the feature matrix for the *horizon* days after the last
    observed date, using known values (date decomposition) and
    last-observed lag values.
    """
    last_date   = df_clean["Date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    # Use aggregate stats from the tail of the training data for lag values
    tail = df_clean.sort_values("Date").tail(30)
    lag_1  = tail["Sales"].iloc[-1]
    lag_7  = tail["Sales"].iloc[-7]  if len(tail) >= 7  else lag_1
    lag_14 = tail["Sales"].iloc[-14] if len(tail) >= 14 else lag_1
    lag_30 = tail["Sales"].iloc[-30] if len(tail) >= 30 else lag_1
    rolling_mean_7 = tail["Sales"].tail(7).mean()
    rolling_std_7  = tail["Sales"].tail(7).std(ddof=0)

    rows = []
    for i, date in enumerate(future_dates):
        rows.append({
            "Date":          date,
            "day":           date.dayofweek,
            "week":          date.isocalendar()[1],
            "month":         date.month,
            "quarter":       (date.month - 1) // 3 + 1,
            "is_weekend":    int(date.dayofweek >= 5),
            "lag_1":         lag_1,
            "lag_7":         lag_7,
            "lag_14":        lag_14,
            "lag_30":        lag_30,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7":  rolling_std_7,
        })

    return pd.DataFrame(rows)


def forecast_with_ci(
    model,
    feature_list: list,
    df_clean: pd.DataFrame,
    residuals: np.ndarray,
    horizon: int   = FORECAST_HORIZON,
    z: float       = CI_Z,
    model_name: str = "Best Model",
) -> pd.DataFrame:
    """
    Generate a *horizon*-day ahead forecast with 95 % confidence intervals.

    Parameters
    ----------
    model        : fitted sklearn-compatible model
    feature_list : list of feature column names the model expects
    df_clean     : the full clean DataFrame (used to build future features)
    residuals    : training residuals (y_train - y_hat_train)
    horizon      : number of future days to forecast
    z            : z-score for the interval (1.96 → 95 %)
    model_name   : label written to the output CSV

    Returns
    -------
    DataFrame with columns: Date, forecast, lower_ci, upper_ci, model_used
    """
    future_df = _build_future_features(df_clean, horizon)
    X_future  = future_df[feature_list]

    point_forecast = model.predict(X_future)
    std_residuals  = np.std(residuals)

    df_out = pd.DataFrame({
        "Date":       future_df["Date"].dt.strftime("%Y-%m-%d"),
        "forecast":   point_forecast.round(1),
        "lower_ci":   (point_forecast - z * std_residuals).clip(min=0).round(1),
        "upper_ci":   (point_forecast + z * std_residuals).round(1),
        "model_used": model_name,
    })

    save_csv(df_out, "predictions.csv", PROCESSED_DIR)

    print(f"\n   Forecast ({horizon} days) using {model_name}:")
    print(f"   Avg forecast:   {df_out['forecast'].mean():,.0f}")
    print(f"   CI width (±):   {(z * std_residuals):,.0f}  (95% CI)")
    print(f"   Date range:     {df_out['Date'].iloc[0]} → {df_out['Date'].iloc[-1]}")

    return df_out


# ── Evaluate on held-out test set and persist for dashboard ──────────────────

def evaluate_on_test(
    model,
    feature_list: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    residuals: np.ndarray,
    model_name: str = "Best Model",
    z: float = CI_Z,
) -> pd.DataFrame:
    """
    Run the best model on the test set and persist forecast-vs-actual CSV.
    Used by the dashboard's historical forecast tab.
    """
    preds = model.predict(X_test[feature_list])
    std   = np.std(residuals)

    df = pd.DataFrame({
        "Sales":      y_test.values,
        "forecast":   preds.round(1),
        "lower_ci":   (preds - z * std).clip(min=0).round(1),
        "upper_ci":   (preds + z * std).round(1),
        "model_used": model_name,
    })
    # Re-attach dates if available
    if "Date" in X_test.columns:
        df.insert(0, "Date", X_test["Date"].values)

    save_csv(df, "test_predictions.csv", PROCESSED_DIR)
    return df
