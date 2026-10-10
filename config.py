"""
Central configuration for the Sales Forecasting project.
Edit values here to change behaviour project-wide.
"""
import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH   = os.path.join(DATA_DIR, "raw",       "sales.csv")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
DB_PATH         = os.path.join(DATA_DIR, "sales.db")

# ── Forecast settings ────────────────────────────────────────────────────────
FORECAST_HORIZON = 30        # days to forecast into the future
CI_Z             = 1.96      # 1.96 → 95% confidence interval
TEST_SPLIT       = 0.20      # fraction of data held out for evaluation

# ── Model settings ───────────────────────────────────────────────────────────
RANDOM_STATE     = 42
N_ESTIMATORS     = 150       # trees for RF and GB

# ── Automation / drift ───────────────────────────────────────────────────────
DRIFT_THRESHOLD  = 15.0      # % MAPE increase that triggers a retraining alert

# ── Feature columns used by ML models ───────────────────────────────────────
ML_FEATURES = [
    "day", "week", "month", "quarter", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7",
]

AR_FEATURES = ["lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7"]
