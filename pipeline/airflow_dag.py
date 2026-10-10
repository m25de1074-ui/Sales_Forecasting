"""
pipeline/airflow_dag.py
-----------------------
Module 4 — Apache Airflow Orchestration DAG

This DAG runs the complete sales forecasting pipeline on a daily schedule.

Task graph
----------
ingest_data  →  run_etl  →  train_models  →  save_predictions  →  check_drift

Setup (one-time)
----------------
1.  pip install apache-airflow
2.  export AIRFLOW_HOME=~/airflow
3.  airflow db init
4.  Copy (or symlink) this file into  $AIRFLOW_HOME/dags/
5.  airflow webserver --port 8080  &  airflow scheduler

Then open http://localhost:8080 and enable the 'sales_forecast_pipeline' DAG.

Without Airflow
---------------
Run  python main.py  to execute the same steps sequentially.
"""
import os
import sys
from datetime import datetime, timedelta

# ── Guard: import airflow only when actually available ────────────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    print(" apache-airflow not installed. Run `python main.py` instead.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Callable task functions ───────────────────────────────────────────────────

def task_ingest(**context):
    from config   import RAW_DATA_PATH
    from etl.preprocess import load_and_flag
    load_and_flag(RAW_DATA_PATH)


def task_etl(**context):
    from config   import RAW_DATA_PATH
    from etl.preprocess import run_etl
    run_etl(RAW_DATA_PATH)


def task_train(**context):
    import pickle
    from config import PROCESSED_DIR, DB_PATH
    from utils.db import load_df
    from models.train import train_all

    df_clean = load_df("clean_sales", DB_PATH)
    result   = train_all(df_clean)
    # Push residuals & best model name via XCom for downstream tasks
    context["ti"].xcom_push(key="best_name",  value=result["best_name"])
    context["ti"].xcom_push(key="residuals",  value=result["residuals"].tolist())


def task_predict(**context):
    import pickle, numpy as np
    from config import PROCESSED_DIR, DB_PATH
    from utils.db import load_df
    from models.predict import forecast_with_ci

    best_name = context["ti"].xcom_pull(key="best_name",  task_ids="train_models")
    residuals = np.array(context["ti"].xcom_pull(key="residuals", task_ids="train_models"))

    models_path = os.path.join(PROCESSED_DIR, "trained_models.pkl")
    with open(models_path, "rb") as f:
        saved = pickle.load(f)

    model, feats = saved["models"][best_name]
    df_clean = load_df("clean_sales", DB_PATH)
    forecast_with_ci(model, feats, df_clean, residuals, model_name=best_name)


def task_drift(**context):
    import pandas as pd
    from config import PROCESSED_DIR, DRIFT_THRESHOLD
    from utils.metrics import check_drift

    scores_path = os.path.join(PROCESSED_DIR, "model_scores.csv")
    if not os.path.exists(scores_path):
        print("No previous scores found — skipping drift check.")
        return

    scores = pd.read_csv(scores_path)
    best   = scores[scores["Rank"] == 1].iloc[0]
    # In a real system you'd compare against a stored baseline MAPE.
    # Here we use a static reference value of 5.0% as a demo baseline.
    BASELINE_MAPE = 5.0
    check_drift(float(best["MAPE(%)"]), BASELINE_MAPE, DRIFT_THRESHOLD)


# ── DAG definition ────────────────────────────────────────────────────────────

if _AIRFLOW_AVAILABLE:
    default_args = {
        "owner":          "mtech_project",
        "depends_on_past": False,
        "retries":         1,
        "retry_delay":     timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="sales_forecast_pipeline",
        default_args=default_args,
        description="Daily automated sales forecasting pipeline",
        schedule_interval="@daily",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["sales", "forecasting", "etl", "mtech"],
    ) as dag:

        t1 = PythonOperator(
            task_id="ingest_data",
            python_callable=task_ingest,
        )
        t2 = PythonOperator(
            task_id="run_etl",
            python_callable=task_etl,
        )
        t3 = PythonOperator(
            task_id="train_models",
            python_callable=task_train,
        )
        t4 = PythonOperator(
            task_id="save_predictions",
            python_callable=task_predict,
        )
        t5 = PythonOperator(
            task_id="check_drift",
            python_callable=task_drift,
        )

        t1 >> t2 >> t3 >> t4 >> t5
