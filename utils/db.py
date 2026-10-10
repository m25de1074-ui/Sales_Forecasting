"""
Thin wrapper around Python's built-in sqlite3 module.
No external dependencies required.
"""
import sqlite3
import os
import pandas as pd


def get_connection(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return sqlite3.connect(db_path)


def save_df(df: pd.DataFrame, table: str, db_path: str,
            if_exists: str = "replace") -> None:
    """Write a DataFrame to an SQLite table."""
    with get_connection(db_path) as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
    print(f"   DB: saved {len(df):,} rows → table '{table}'")


def load_df(table: str, db_path: str) -> pd.DataFrame:
    """Read an entire SQLite table into a DataFrame."""
    with get_connection(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    return df


def save_csv(df: pd.DataFrame, filename: str, processed_dir: str) -> str:
    """Save a DataFrame as a CSV in the processed data directory."""
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, filename)
    df.to_csv(path, index=False)
    print(f"   CSV: saved → {path}")
    return path
