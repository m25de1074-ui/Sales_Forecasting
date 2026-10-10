# 📈 Automated Sales Forecasting System with Lightweight ETL Pipeline

> **M.Tech Computer Science Project**  
> Modular · Automated · Production-Inspired  
> Implements: Confidence Intervals · A/B Model Comparison · Feature Importance

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Key Features](#3-key-features)
4. [Project Structure](#4-project-structure)
5. [Tech Stack](#5-tech-stack)
6. [Installation & Setup](#6-installation--setup)
7. [How to Run](#7-how-to-run)
8. [Module-by-Module Explanation](#8-module-by-module-explanation)
9. [Dashboard Guide](#9-dashboard-guide)
10. [Output Files](#10-output-files)
11. [Airflow Orchestration](#11-airflow-orchestration)
12. [Results & Evaluation](#12-results--evaluation)
13. [Extending the Project](#13-extending-the-project)

---

## 1. Project Overview

Traditional retail businesses struggle with **inaccurate sales forecasting** due to fragmented data pipelines, manual model execution, and no automated retraining. This project builds a complete, end-to-end automated sales forecasting system that:

- **Ingests** raw retail sales data (CSV / API)
- **Transforms** it through a robust ETL pipeline with anomaly detection
- **Trains** three forecasting models and automatically selects the best one
- **Forecasts** future sales with statistically-grounded confidence intervals
- **Visualises** all results through an interactive Streamlit dashboard
- **Automates** the entire cycle via Apache Airflow DAGs

The system is inspired by production MLOps architectures but optimised for lightweight, local deployment — no cloud required.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DATA SOURCES                               │
│          CSV  (Kaggle: Walmart / Rossmann / Store Sales)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODULE 1 — DATA INGESTION                          │
│  Load CSV → Schema validation → Anomaly flagging           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODULE 2 — ETL PIPELINE                            │
│  Missing values → Date features → Lag features             │
│  Rolling stats → SQLite persistence                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODULE 3 — FORECASTING ENGINE                      │
│                                                              │
│   AR Model  ──┐                                             │
│   Random Forest ──┤──► A/B Comparison ──► Best Model       │
│   Gradient Boosting ──┘   (MAPE rank)     + CI Generation  │
│                                           + Feature Import. │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODULE 4 — AUTOMATION (Apache Airflow)             │
│  Ingest → ETL → Train → Predict → Drift Check              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          MODULE 5 — STREAMLIT DASHBOARD                     │
│  Tab 1: Forecast + CI Band                                  │
│  Tab 2: A/B Table + Feature Importance                      │
│  Tab 3: Data Quality + Anomaly Report                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Key Features

### ✅ Confidence Intervals on Forecasts (NEW)
Instead of a single prediction line, the system generates a **95% confidence band** around every forecast. The band is derived from the model's training residuals (actual minus predicted), scaled by a z-score of 1.96. This gives stakeholders an honest picture of forecast uncertainty.

### ✅ A/B Model Comparison Table (NEW)
All three models are trained and evaluated on the **same held-out test set**. Their MAE, RMSE, and MAPE scores are ranked in a comparison table. The winning model is highlighted in green in the dashboard. This makes model selection transparent and reproducible.

### ✅ Feature Importance Visualisation (NEW)
After training, the winning tree-based model's `.feature_importances_` array is extracted, saved, and displayed as a ranked horizontal bar chart. This answers the key question: *"what is actually driving our forecast?"*

### Additional features
- **Anomaly Detection** — flags statistical outliers (>3σ) during ingestion
- **Drift Monitoring** — alerts when MAPE worsens beyond a configurable threshold
- **Modular architecture** — each module is independently testable and replaceable
- **Airflow DAG** — production-ready orchestration with scheduled retraining

---

## 4. Project Structure

```
sales-forecasting-project/
│
├── config.py                  # Central configuration (paths, thresholds)
├── main.py                    # Run the full pipeline (no Airflow needed)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                   # Place your sales.csv here
│   │   └── .gitkeep
│   └── processed/             # Auto-generated outputs
│       └── .gitkeep
│
├── scripts/
│   └── generate_sample_data.py   # Generates synthetic data for testing
│
├── etl/
│   ├── __init__.py
│   └── preprocess.py          # Modules 1 & 2: ingestion + transformation
│
├── models/
│   ├── __init__.py
│   ├── train.py               # Module 3: train all models
│   ├── evaluate.py            # A/B comparison + feature importance  ← NEW
│   └── predict.py             # Forecast with confidence intervals   ← NEW
│
├── pipeline/
│   ├── __init__.py
│   └── airflow_dag.py         # Module 4: Airflow DAG definition
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                 # Module 5: Streamlit dashboard
│
└── utils/
    ├── __init__.py
    ├── metrics.py             # MAE, RMSE, MAPE, drift check
    ├── anomaly.py             # Rolling-window anomaly flagging
    └── db.py                  # SQLite read/write helpers
```

---

## 5. Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| ETL & Data | Python + Pandas | Flexible, industry-standard |
| Storage | SQLite (built-in) | Zero-setup, file-based DB |
| ML Models | scikit-learn | RandomForest + GradientBoosting |
| Boosting (optional) | XGBoost | Drop-in upgrade, auto-detected |
| Time Series | statsmodels (optional) | ARIMA / SARIMA support |
| Dashboard | Streamlit + Plotly | Rapid, interactive UI |
| Orchestration | Apache Airflow (optional) | Production-grade DAG scheduling |
| Tracking | MLflow (optional) | Experiment logging |

> **No installation is required for the core pipeline.** Streamlit/Plotly are needed only for the dashboard. Airflow is needed only for automated scheduling.

---

## 6. Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/sales-forecasting-project.git
cd sales-forecasting-project
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
# Minimum (pipeline only):
pip install pandas numpy scikit-learn scipy

# Full install (pipeline + dashboard):
pip install -r requirements.txt

# Optional upgrades:
pip install xgboost statsmodels apache-airflow mlflow
```

### Step 4 — Add your data  *(skip if using sample data)*
Download the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data) or the [Walmart Sales dataset](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) from Kaggle.

Rename the main file to `sales.csv` and place it in `data/raw/`.

Required columns: `Date`, `Store`, `Sales`, `Customers`

---

## 7. How to Run

### Quick start (sample data + full pipeline)
```bash
python main.py
```
This will:
1. Generate synthetic sales data if `data/raw/sales.csv` is absent
2. Run ingestion with anomaly detection
3. Run the ETL feature engineering pipeline
4. Train all three models and compare them
5. Generate a 30-day forecast with confidence intervals
6. Save all outputs to `data/processed/`

### Launch the dashboard
```bash
# In a separate terminal (after running main.py):
streamlit run dashboard/app.py
```
Open your browser at **http://localhost:8501**

### Generate sample data only
```bash
python scripts/generate_sample_data.py
```

### Run with Airflow (optional)
```bash
pip install apache-airflow
export AIRFLOW_HOME=~/airflow
airflow db init
airflow dags unpause sales_forecast_pipeline
airflow scheduler &
airflow webserver --port 8080
```
Copy `pipeline/airflow_dag.py` to `~/airflow/dags/` and enable it in the UI.

---

## 8. Module-by-Module Explanation

### `config.py` — Central Configuration
All file paths, model hyperparameters, and thresholds live here. Edit this file to change the forecast horizon, CI z-score, drift threshold, or number of estimators.

```python
FORECAST_HORIZON = 30       # days ahead to forecast
CI_Z             = 1.96     # 1.96 → 95% confidence interval
DRIFT_THRESHOLD  = 15.0     # % MAPE increase that triggers a retraining alert
N_ESTIMATORS     = 150      # trees for RandomForest and GradientBoosting
```

---

### `etl/preprocess.py` — Data Ingestion & Feature Engineering

#### `load_and_flag(filepath)`
Loads the raw CSV, validates that required columns (`Date`, `Store`, `Sales`, `Customers`) exist, and runs anomaly detection on the `Sales` column using a 30-day rolling window with a 3-sigma threshold.

```
Input:  data/raw/sales.csv
Output: raw_sales table in SQLite + data/processed/anomalies.csv
```

#### `transform(df)`
Performs all feature engineering:

| Feature | How | Why |
|---------|-----|-----|
| `day`, `week`, `month`, `quarter` | `pd.DatetimeIndex` attributes | Capture seasonal patterns |
| `is_weekend` | `day >= 5` | Weekend vs weekday demand |
| `lag_1`, `lag_7`, `lag_14`, `lag_30` | `.groupby("Store").shift(n)` | Autoregressive signal |
| `rolling_mean_7`, `rolling_std_7` | Shifted rolling window | Local trend + volatility |

Lag features are computed **per store** and shifted by 1 before rolling to prevent **data leakage**.

```
Input:  raw DataFrame
Output: clean_sales table in SQLite + data/processed/clean_sales.csv
```

---

### `models/train.py` — Training Three Models

#### AR Model (`train_ar`)
A `LinearRegression` trained only on lag and rolling features (`AR_FEATURES`). This is equivalent to an AutoRegressive model — a linear combination of past observations — which is the core of ARIMA without explicit differencing. Interpretable, fast, and a good baseline.

#### Random Forest (`train_rf`)
`RandomForestRegressor` with 150 trees, `max_depth=12`, `min_samples_leaf=4`. Uses all engineered features. Handles non-linearity and feature interactions well.

#### Gradient Boosting / XGBoost (`train_booster`)
`GradientBoostingRegressor` by default. If `xgboost` is installed, `XGBRegressor` is used automatically — same interface, superior speed and performance. Uses all engineered features.

#### Time-based Train/Test Split
Data is split **chronologically** (80% train / 20% test) not randomly. Random splitting would cause data leakage in time series because future values would appear in training.

---

### `models/evaluate.py` — A/B Comparison & Feature Importance ← NEW

#### `compare_models(models, X_test, y_test)`
Evaluates every trained model on the **same test set** using three metrics:

- **MAE** (Mean Absolute Error) — average absolute difference in units
- **RMSE** (Root Mean Squared Error) — penalises large errors more
- **MAPE** (Mean Absolute Percentage Error) — scale-independent, best for comparison

Models are ranked by MAPE. The winner is saved to `data/processed/model_scores.csv`.

#### `save_feature_importance(model, feature_names, model_name)`
Calls `model.feature_importances_` on the winning tree model and saves a ranked DataFrame to `data/processed/feat_importance.csv`. Typical top features:
- `rolling_mean_7` — captures recent sales momentum
- `lag_1`, `lag_7` — direct previous-day and previous-week signal
- `month`, `week` — seasonal effects

---

### `models/predict.py` — Forecast with Confidence Intervals ← NEW

#### `forecast_with_ci(model, feature_list, df_clean, residuals)`
Generates a `FORECAST_HORIZON`-day forecast from the last observed date.

**Confidence Interval calculation:**
```
std_residuals = std(y_train - ŷ_train)
lower_ci = forecast - 1.96 × std_residuals
upper_ci = forecast + 1.96 × std_residuals
```

This is the standard 95% prediction interval under the assumption that residuals are normally distributed — a valid assumption for well-fitted regression models.

Output: `data/processed/predictions.csv` with columns `Date`, `forecast`, `lower_ci`, `upper_ci`, `model_used`.

---

### `dashboard/app.py` — Streamlit Dashboard

Three tabs:

**Tab 1 — Forecast View**
- Historical sales (last 90 days) as a solid blue line
- 30-day forecast as a dashed red line
- 95% CI shaded band around the forecast
- Expandable table with all forecast values

**Tab 2 — Model Performance**
- A/B comparison table with winner highlighted green
- MAPE bar chart for all three models
- Feature importance horizontal bar chart
- Full feature importance table (expandable)

**Tab 3 — Data Quality**
- Row count, missing value summary
- Monthly anomaly distribution bar chart
- Expandable table of flagged anomaly rows

---

### `utils/metrics.py` — Evaluation Metrics

| Function | Description |
|----------|-------------|
| `mae(y, ŷ)` | Mean Absolute Error |
| `rmse(y, ŷ)` | Root Mean Squared Error |
| `mape(y, ŷ)` | Mean Absolute Percentage Error (zero-safe) |
| `score_model(y, ŷ)` | Returns all three in one dict |
| `check_drift(current, baseline, threshold)` | Prints alert if MAPE has worsened |

---

### `utils/anomaly.py` — Anomaly Detection

Uses a **rolling Z-score** approach:
```
upper_fence = rolling_mean(30) + 3 × rolling_std(30)
lower_fence = rolling_mean(30) − 3 × rolling_std(30)
```
Rows outside these fences are flagged `is_anomaly = 1`. They are **not removed** — they are tagged for review. This preserves data integrity and provides an audit trail.

---

## 9. Dashboard Guide

After running `python main.py`, launch the dashboard:
```bash
streamlit run dashboard/app.py
```

| Tab | What you see | What to look for |
|-----|-------------|-----------------|
| 📈 Forecast View | Forecast line + CI shaded area | Width of CI shows model confidence |
| 🏆 Model Performance | Green-highlighted winner + feature bars | MAPE < 5% is good; lag_1 usually top feature |
| 🔍 Data Quality | Anomaly timeline + flagged rows | Spikes often correlate with promotions |

---

## 10. Output Files

All outputs are saved to `data/processed/`:

| File | Contents |
|------|----------|
| `clean_sales.csv` | Feature-engineered training data |
| `model_scores.csv` | MAE, RMSE, MAPE for all 3 models |
| `feat_importance.csv` | Feature weights from best tree model |
| `predictions.csv` | 30-day forecast + lower_ci + upper_ci |
| `test_predictions.csv` | Forecast vs actual on hold-out set |
| `anomalies.csv` | Rows flagged as statistical outliers |
| `trained_models.pkl` | Serialised trained models (pickle) |
| `../sales.db` | SQLite database with all tables |

---

## 11. Airflow Orchestration

The `pipeline/airflow_dag.py` defines a 5-task DAG:

```
ingest_data → run_etl → train_models → save_predictions → check_drift
```

Each task wraps a function from the pipeline. Tasks are connected with `>>` for strict sequential dependency. On failure, each task retries once after 5 minutes.

To schedule daily runs, set `schedule_interval="@daily"` (already configured).

For retraining on drift, the `check_drift` task compares the current MAPE against a stored baseline and logs an alert. In production, you'd trigger an additional retraining DAG run via the Airflow API.

---

## 12. Results & Evaluation

Sample results on 2-year synthetic dataset (5 stores, 730 days):

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| AR Model | ~620 | ~810 | ~8.5% |
| Random Forest | ~390 | ~510 | ~5.2% |
| Gradient Boosting | ~360 | ~480 | ~4.9% |

> Gradient Boosting typically wins due to its ability to capture non-linear interactions between lag features and calendar features.

**What MAPE means in context:**
- < 5% → Excellent (production-grade)
- 5–10% → Good (acceptable for most retail use cases)
- > 10% → Needs improvement

---

## 13. Extending the Project

| Enhancement | Effort | Where to add |
|-------------|--------|-------------|
| ARIMA / SARIMA | Medium | `models/train.py` — wrap `statsmodels.tsa.arima` |
| Per-store forecasting | Medium | Modify `train_all()` to loop over stores |
| MLflow experiment tracking | Low | Add `mlflow.log_metrics()` in `train.py` |
| REST API (Flask) | Medium | New `api/` module using Flask |
| Docker deployment | Medium | Add `Dockerfile` + `docker-compose.yml` |
| Real Kaggle dataset | Low | Download + rename to `data/raw/sales.csv` |
| Email alerts on drift | Low | Add `smtplib` call in `utils/metrics.py` |

---

## License

MIT — free to use and adapt for academic or commercial purposes.

---

*Built for M.Tech Computer Science — Project Proposal Submission*  
*"Modular forecasting with automated ETL and retraining capability,  
inspired by production MLOps systems but optimised for lightweight deployment."*
