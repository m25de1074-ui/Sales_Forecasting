"""Rolling-window anomaly detection for the ETL ingestion step."""
import pandas as pd
import numpy as np


def flag_anomalies(
    df: pd.DataFrame,
    column: str = "Sales",
    window: int = 30,
    n_std: float = 3.0,
) -> pd.DataFrame:
    """
    Adds an ``is_anomaly`` column (0/1) to *df*.

    A row is flagged when its value is more than *n_std* standard deviations
    away from the trailing *window*-day rolling mean — a standard
    statistical process control technique.

    Parameters
    ----------
    df      : DataFrame that contains *column*
    column  : column to inspect (default: 'Sales')
    window  : look-back window in rows (default: 30 days)
    n_std   : number of standard deviations for the outer fence (default: 3)
    """
    df = df.copy()
    rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
    rolling_std  = (
        df[column].rolling(window=window, min_periods=1).std().fillna(0)
    )
    upper = rolling_mean + n_std * rolling_std
    lower = rolling_mean - n_std * rolling_std

    df["is_anomaly"] = ((df[column] > upper) | (df[column] < lower)).astype(int)

    n_flagged = df["is_anomaly"].sum()
    pct = 100 * n_flagged / len(df)
    print(f"   Anomaly detection: {n_flagged} rows flagged ({pct:.1f}%)")
    return df
