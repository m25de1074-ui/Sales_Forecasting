"""
dashboard/app.py
----------------
Module 5 — Streamlit Interactive Dashboard

Three tabs
----------
Tab 1   Forecast View
       Historical sales chart + 30-day forecast line + shaded 95% CI band.
       Store / product dropdown filter.

Tab 2   Model Performance
       A/B comparison table (all models, green-highlighted winner).
       Feature importance horizontal bar chart.

Tab 3    Data Quality
       Anomaly-flagged rows table.  Missing-value summary.

Run
---
    streamlit run dashboard/app.py

Install dependencies first:
    pip install streamlit plotly
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, RAW_DATA_PATH

# ── Streamlit / Plotly (optional at import time so IDE doesn't break) ─────────
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    _DASH_AVAILABLE = True
except ImportError:
    print("Install Streamlit and Plotly:  pip install streamlit plotly")
    sys.exit(1)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helper: load CSVs with caching ────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_csv(filename: str) -> pd.DataFrame | None:
    path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_raw() -> pd.DataFrame | None:
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("Sales Forecasting")
st.sidebar.caption("M.Tech Project — Automated ETL + ML Pipeline")
st.sidebar.divider()

if st.sidebar.button("  Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ── Main header ───────────────────────────────────────────────────────────────
st.title("Automated Sales Forecasting Dashboard")
st.caption("Real-time forecast · A/B model comparison · Data quality monitoring")

# ── Check data availability ───────────────────────────────────────────────────
predictions  = load_csv("predictions.csv")
test_preds   = load_csv("test_predictions.csv")
model_scores = load_csv("model_scores.csv")
feat_imp     = load_csv("feat_importance.csv")
anomalies    = load_csv("anomalies.csv")
clean_sales  = load_csv("clean_sales.csv")

if predictions is None:
    st.warning(
        "  No forecast data found.  Run the pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )
    st.stop()

# ── KPI strip ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
best_model_name = model_scores.iloc[0]["Model"] if model_scores is not None else "—"
best_mape       = model_scores.iloc[0]["MAPE(%)"] if model_scores is not None else "—"
n_anomalies     = len(anomalies) if anomalies is not None else 0
avg_forecast    = predictions["forecast"].mean() if predictions is not None else 0

col1.metric("Best Model",        best_model_name)
col2.metric("Best MAPE",         f"{best_mape}%")
col3.metric("Anomalies Flagged", n_anomalies)
col4.metric("Avg Daily Forecast", f"{avg_forecast:,.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Forecast View", " Model Performance", " Data Quality"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST WITH CONFIDENCE INTERVAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Sales Forecast — Next 30 Days")

    # ── Historical sales chart ─────────────────────────────────────────────
    if clean_sales is not None:
        clean_sales["Date"] = pd.to_datetime(clean_sales["Date"])
        daily_hist = (
            clean_sales.groupby("Date", as_index=False)["Sales"].sum()
                       .sort_values("Date")
                       .tail(90)   # last 90 days of history
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_hist["Date"], y=daily_hist["Sales"],
            name="Historical Sales",
            line=dict(color="#0077A8", width=2),
        ))

    # ── Forecast + CI band ─────────────────────────────────────────────────
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    # Shaded confidence band (upper to lower, filled)
    fig.add_trace(go.Scatter(
        x=pd.concat([predictions["Date"], predictions["Date"][::-1]]),
        y=pd.concat([predictions["upper_ci"], predictions["lower_ci"][::-1]]),
        fill="toself",
        fillcolor="rgba(255, 107, 107, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=predictions["Date"], y=predictions["forecast"],
        name=f"Forecast ({predictions['model_used'].iloc[0]})",
        line=dict(color="#FF6B6B", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    fig.update_layout(
        title="Historical Sales + 30-Day Forecast with 95% Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Sales (units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ─────────────────────────────────────────────────────
    with st.expander(" View Forecast Table"):
        st.dataframe(
            predictions[["Date", "forecast", "lower_ci", "upper_ci", "model_used"]]
              .rename(columns={"forecast": "Point Forecast",
                               "lower_ci": "Lower CI (95%)",
                               "upper_ci": "Upper CI (95%)",
                               "model_used": "Model"}),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([1, 1])

    # ── A/B comparison table ───────────────────────────────────────────────
    with col_a:
        st.subheader("A/B Model Comparison")
        if model_scores is not None:

            def _highlight_winner(row):
                if row.get("Best") == " Winner":
                    return ["background-color: #d4f7dc; font-weight: bold"] * len(row)
                return [""] * len(row)

            display_scores = model_scores.copy()
            st.dataframe(
                display_scores.style.apply(_highlight_winner, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # Mini bar chart of MAPE comparison
            fig_mape = px.bar(
                model_scores,
                x="Model", y="MAPE(%)",
                color="Model",
                color_discrete_sequence=["#0077A8", "#00B4D8", "#FF6B6B"],
                title="MAPE by Model (lower is better)",
                text="MAPE(%)",
            )
            fig_mape.update_traces(textposition="outside")
            fig_mape.update_layout(
                showlegend=False, plot_bgcolor="white",
                paper_bgcolor="#f8fafc", height=320
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        else:
            st.info("Model scores not available. Run  `python main.py`  first.")

    # ── Feature importance ─────────────────────────────────────────────────
    with col_b:
        st.subheader("Feature Importance")
        if feat_imp is not None:
            model_label = feat_imp["Model"].iloc[0] if "Model" in feat_imp.columns else "Best Model"
            st.caption(f"Model: **{model_label}**")

            fig_fi = px.bar(
                feat_imp.head(11),
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#cce5f0", "#0077A8"],
                title="Top Feature Importances",
                text=feat_imp.head(11)["Importance"].round(4),
            )
            fig_fi.update_traces(textposition="outside")
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="#f8fafc",
                coloraxis_showscale=False,
                height=420,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            with st.expander("Full feature importance table"):
                st.dataframe(feat_imp, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance not available. Run  `python main.py`  first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Data Quality Report")

    if clean_sales is not None:
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Total Rows (clean)",  f"{len(clean_sales):,}")
        col_y.metric("Missing Values",      int(clean_sales.isnull().sum().sum()))
        col_z.metric("Stores",              clean_sales["Store"].nunique())

        st.markdown("**Missing values per column:**")
        mv = clean_sales.isnull().sum()
        mv = mv[mv > 0]
        if len(mv) == 0:
            st.success(" No missing values in clean dataset")
        else:
            st.dataframe(mv.rename("Missing Count"), use_container_width=True)

    st.divider()
    st.subheader(" Anomaly-Flagged Rows")
    if anomalies is not None and len(anomalies) > 0:
        st.warning(f"{len(anomalies)} rows flagged as anomalies (>3σ from rolling mean)")
        # Anomaly distribution over time
        if "Date" in anomalies.columns:
            anomalies["Date"] = pd.to_datetime(anomalies["Date"])
            monthly = anomalies.groupby(anomalies["Date"].dt.to_period("M")).size().reset_index()
            monthly.columns = ["Month", "Count"]
            monthly["Month"] = monthly["Month"].astype(str)
            fig_an = px.bar(monthly, x="Month", y="Count",
                            title="Anomalies per Month",
                            color_discrete_sequence=["#FF6B6B"])
            fig_an.update_layout(plot_bgcolor="white", paper_bgcolor="#f8fafc")
            st.plotly_chart(fig_an, use_container_width=True)

        with st.expander("View anomaly rows"):
            st.dataframe(anomalies.head(200), use_container_width=True, hide_index=True)
    else:
        st.success(" No anomalies detected in the current dataset")
