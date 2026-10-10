"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic retail sales CSV when you do not yet have
a real Kaggle dataset.  Run once before main.py:

    python scripts/generate_sample_data.py

To use a real dataset instead, download the Rossmann or Walmart Sales
dataset from Kaggle and place it at  data/raw/sales.csv  with columns:
    Date, Store, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DATA_PATH


def generate(
    n_stores: int = 5,
    start: str = "2022-01-01",
    end:   str = "2023-12-31",
    seed:  int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    dates  = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    records = []

    for store_id in range(1, n_stores + 1):
        # Each store has a slightly different base sales level
        base   = np.random.randint(4_000, 8_000)
        trend  = np.linspace(0, base * 0.15, n_days)           # gentle upward trend
        weekly = 600 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # weekly cycle
        yearly = 400 * np.sin(2 * np.pi * np.arange(n_days) / 365) # yearly cycle
        noise  = np.random.normal(0, base * 0.08, n_days)

        # Promotions: random 10% of days get a 20-40% boost
        promo  = np.zeros(n_days)
        promo_idx = np.random.choice(n_days, size=int(n_days * 0.10), replace=False)
        promo[promo_idx] = base * np.random.uniform(0.20, 0.40, len(promo_idx))

        # Closed days (Open=0): ~8% of days, mostly Sundays
        open_flag = np.ones(n_days, dtype=int)
        sunday_idx = [i for i, d in enumerate(dates) if d.dayofweek == 6]
        closed = np.random.choice(sunday_idx, size=int(len(sunday_idx) * 0.60), replace=False)
        open_flag[closed] = 0

        sales = (base + trend + weekly + yearly + noise + promo).clip(min=0)
        sales[open_flag == 0] = 0  # no sales on closed days
        sales = sales.round().astype(int)

        customers = (sales / np.random.uniform(8, 14)).clip(min=0).round().astype(int)

        df_store = pd.DataFrame({
            "Date":          dates,
            "Store":         store_id,
            "Sales":         sales,
            "Customers":     customers,
            "Open":          open_flag,
            "Promo":         (promo > 0).astype(int),
            "StateHoliday":  0,
            "SchoolHoliday": np.random.randint(0, 2, n_days),
        })
        records.append(df_store)

    df = pd.concat(records, ignore_index=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    print("⏳  Generating synthetic sales data …")
    df = generate()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f" Saved {len(df):,} rows ({df['Store'].nunique()} stores) → {RAW_DATA_PATH}")
    print(df.head())
