"""
etl/preprocess.py
-----------------
Module 1 — Data Ingestion  +  Module 2 — ETL Pipeline

Public API
----------
load_and_flag(filepath)   → raw DataFrame with is_anomaly column
transform(df)             → feature-engineered DataFrame ready for modelling
run_etl(filepath)         → convenience: runs both steps and saves outputs
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH, PROCESSED_DIR, ML_FEATURES
from utils.anomaly import flag_anomalies
from utils.db import save_df, save_csv


# ── MODULE 1: Data Ingestion ──────────────────────────────────────────────────

def load_and_flag(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV, validate schema, flag statistical anomalies.

    Steps
    -----
    1. Read CSV and parse the Date column.
    2. Assert required columns exist (fast-fail with a clear message).
    3. Apply rolling-window anomaly detection.
    4. Persist raw data (+ anomaly flag) to SQLite and CSV.
    """
    print("\n── Module 1: Data Ingestion ─────────────────────────────────────")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw data not found at '{filepath}'.\n"
            "Run:  python scripts/generate_sample_data.py  first."
        )

    df = pd.read_csv(filepath, parse_dates=["Date"])
    print(f"   Loaded {len(df):,} rows from {filepath}")

    # ── schema validation ─────────────────────────────────────────────────
    required = ["Date", "Store", "Sales", "Customers"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── coerce types ──────────────────────────────────────────────────────
    df["Sales"]     = pd.to_numeric(df["Sales"],     errors="coerce").fillna(0)
    df["Customers"] = pd.to_numeric(df["Customers"], errors="coerce").fillna(0)

    # ── anomaly detection ─────────────────────────────────────────────────
    df = flag_anomalies(df, column="Sales")

    # ── persist ───────────────────────────────────────────────────────────
    save_df(df, "raw_sales", DB_PATH)
    anomalies = df[df["is_anomaly"] == 1].copy()
    save_csv(anomalies, "anomalies.csv", PROCESSED_DIR)

    print(f"     Ingestion complete — {len(df):,} rows, "
          f"{df['Store'].nunique()} stores, "
          f"date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ── MODULE 2: ETL / Feature Engineering ──────────────────────────────────────

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from raw sales data.

    Transformations
    ---------------
    - Drop closed-store rows (Sales == 0 and Open == 0)
    - Forward-fill missing Sales values
    - Median-fill missing Customers values
    - Date decomposition: day-of-week, ISO week, month, quarter, is_weekend
    - Lag features: 1-day, 7-day, 14-day, 30-day previous sales
    - Rolling statistics: 7-day mean and std of Sales (lag-shifted to avoid leakage)
    - Drop rows with NaN (created by lag/rolling at the start of each store)

    Returns the cleaned, feature-rich DataFrame.
    """
    print("\n── Module 2: ETL / Feature Engineering ──────────────────────────")

    df = df.copy()

    # ── drop closed days ──────────────────────────────────────────────────
    if "Open" in df.columns:
        before = len(df)
        df = df[~((df["Open"] == 0) & (df["Sales"] == 0))]
        print(f"   Dropped {before - len(df):,} closed-store rows")

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # ── handle missing values ─────────────────────────────────────────────
    df["Sales"]     = df.groupby("Store")["Sales"].ffill()
    df["Customers"] = df["Customers"].fillna(df["Customers"].median())

    # ── date features ─────────────────────────────────────────────────────
    df["day"]        = df["Date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["week"]       = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["Date"].dt.month
    df["quarter"]    = df["Date"].dt.quarter
    df["is_weekend"] = (df["day"] >= 5).astype(int)

    # ── lag features (per store to prevent cross-store leakage) ───────────
    grp = df.groupby("Store")["Sales"]
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # ── rolling statistics (shifted by 1 to avoid look-ahead bias) ────────
    df["rolling_mean_7"] = (
        grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    df["rolling_std_7"] = (
        grp.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
    )

    # ── drop NaN rows introduced by lagging ──────────────────────────────
    before = len(df)
    df = df.dropna(subset=ML_FEATURES).reset_index(drop=True)
    print(f"   Dropped {before - len(df):,} NaN rows after feature engineering")
    print(f"   Feature set: {ML_FEATURES}")
    print(f"    ETL complete — {len(df):,} rows ready for modelling")

    # ── persist ───────────────────────────────────────────────────────────
    save_df(df, "clean_sales", DB_PATH)
    save_csv(df, "clean_sales.csv", PROCESSED_DIR)

    return df


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_etl(filepath: str) -> pd.DataFrame:
    """Load → flag anomalies → transform → return clean DataFrame."""
    raw_df   = load_and_flag(filepath)
    clean_df = transform(raw_df)
    return clean_df
