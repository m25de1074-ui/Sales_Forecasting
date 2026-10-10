"""
main.py
-------
Run the complete Sales Forecasting pipeline without Airflow.

Usage
-----
    # First time: generate sample data
    python scripts/generate_sample_data.py

    # Run the full pipeline
    python main.py

    # Launch the dashboard (separate terminal)
    streamlit run dashboard/app.py

What this does
--------------
Step 1  — Data Ingestion   : load CSV, validate schema, flag anomalies
Step 2  — ETL Pipeline     : clean, engineer features, save to SQLite
Step 3  — Model Training   : AR model, Random Forest, Gradient Boosting
Step 4  — A/B Comparison   : rank models by MAPE, pick winner    ← NEW
Step 5  — Feature Import.  : extract and save feature weights    ← NEW
Step 6  — Forecast + CI    : 30-day forecast with 95% CI bands   ← NEW
Step 7  — Drift Check      : compare current MAPE to baseline
"""
import os
import sys
import time

# ── Make project root importable regardless of cwd ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import RAW_DATA_PATH, PROCESSED_DIR, DRIFT_THRESHOLD

SEPARATOR = "═" * 60


def _banner(text: str):
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)


def run():
    start_total = time.time()
    _banner("Automated Sales Forecasting Pipeline  ")

    # ── Step 0: Generate sample data if none exists ───────────────────────
    if not os.path.exists(RAW_DATA_PATH):
        _banner("Step 0 — Generating sample data")
        from scripts.generate_sample_data import generate
        import pandas as pd
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df_gen = generate()
        df_gen.to_csv(RAW_DATA_PATH, index=False)
        print(f"     Sample data saved → {RAW_DATA_PATH}")
    else:
        print(f"\n      Raw data found at {RAW_DATA_PATH} — skipping generation.")

    # ── Step 1 + 2: Ingestion & ETL ───────────────────────────────────────
    _banner("Steps 1–2 — Ingestion & ETL")
    from etl.preprocess import run_etl
    t = time.time()
    df_clean = run_etl(RAW_DATA_PATH)
    print(f"\n     ETL completed in {time.time() - t:.1f}s")

    # ── Step 3–5: Training, A/B Comparison, Feature Importance ───────────
    _banner("Steps 3–5 — Model Training & Evaluation")
    from models.train import train_all
    t = time.time()
    result = train_all(df_clean)
    print(f"\n    Training completed in {time.time() - t:.1f}s")

    best_name  = result["best_name"]
    best_model, best_feats = result["models"][best_name]
    residuals  = result["residuals"]
    scores_df  = result["scores_df"]

    # ── Step 6: Forecast with Confidence Intervals ────────────────────────
    _banner("Step 6 — 30-Day Forecast with Confidence Intervals")
    from models.predict import forecast_with_ci, evaluate_on_test
    t = time.time()

    forecast_df = forecast_with_ci(
        model=best_model,
        feature_list=best_feats,
        df_clean=df_clean,
        residuals=residuals,
        model_name=best_name,
    )

    # Also generate test-set forecast vs actual (for dashboard tab 1)
    evaluate_on_test(
        model=best_model,
        feature_list=best_feats,
        X_test=result["X_test"],
        y_test=result["y_test"],
        residuals=residuals,
        model_name=best_name,
    )
    print(f"\n    Forecasting completed in {time.time() - t:.1f}s")

    # ── Step 7: Drift Check ───────────────────────────────────────────────
    _banner("Step 7 — Drift Detection")
    from utils.metrics import check_drift
    current_mape = float(scores_df.iloc[0]["MAPE(%)"])
    BASELINE_MAPE = 5.0   # replace with stored historical baseline in production
    drift_detected = check_drift(current_mape, BASELINE_MAPE, DRIFT_THRESHOLD)

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - start_total
    _banner("Pipeline Complete ")
    print(f"   Best model:       {best_name}")
    print(f"   Best MAPE:        {current_mape:.2f}%")
    print(f"   Forecast horizon: {len(forecast_df)} days")
    print(f"   Outputs saved to: {PROCESSED_DIR}/")
    print(f"   Total time:       {total_time:.1f}s")
    print(f"\n{'─'*60}")
    print("   To view the dashboard:")
    print("   1.  pip install streamlit plotly")
    print("   2.  streamlit run dashboard/app.py")
    print(f"{'─'*60}\n")

    return {
        "df_clean":    df_clean,
        "best_name":   best_name,
        "scores_df":   scores_df,
        "forecast_df": forecast_df,
    }


if __name__ == "__main__":
    run()
