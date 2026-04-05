import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display

import streamlit as st
import plotly.graph_objects as go
from src.sarimax_model import sarimax_model_pipeline


MODEL_OVERVIEW_MD_SARIMAX_1 = """
# 📈 SARIMAX Model Overview

**SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)** is a statistical time series model that captures trends, seasonality, and external influences.

It works by:
- modeling relationships between past values and errors
- incorporating seasonal patterns explicitly
- using external variables (exogenous features) to improve predictions
- combining all components into a structured forecasting model
"""

MODEL_OVERVIEW_MD_SARIMAX_2 = """
### ✔ Why it works well here
- Captures **trend and seasonality** simultaneously
- Incorporates **external drivers** when included
- Strong for **time series forecasting problems**
- Provides **interpretable statistical structure**
- Well-suited for economic and housing data

### ⚠ Limitations
- Assumes mostly **linear relationships**
- Requires **parameter tuning** (p, d, q, P, D, Q)
- Can be slower on **large datasets**
- Sensitive to data quality and missing values
"""

MODEL_APROACHES_MD_1 = """
### 📍 Region Model
**Train:** Selected region only  
**Goal:** Capture local behavior  

✔ Pros:
- Highly localized  
- Captures regional trends  

⚠ Cons:
- Limited data volume
"""

MODEL_APROACHES_MD_2 = """
### 🏙️ State Aggregate Model
**Train:** Aggregated state-level data  
**Goal:** Understand state trends  

✔ Pros:
- More stable than region-level  
- Robust to single-county anomalies  

⚠ Cons:
- Loses granular regional detail
"""

MODEL_APROACHES_MD_3 = """
### 🇺🇸 US Aggregate Model
**Train:** Aggregated US-level data  
**Goal:** Capture national macro trends  

✔ Pros:
- Very stable  
- Shows broad market trajectory  

⚠ Cons:
- Too general for local real estate decisions
"""

model_vars = ["rg_sarimax", "state_sarimax", "aggr_sarimax"]

sarimax_tuning_features = ["p", "d", "q", "P", "D", "Q"]

sarimax_param_grid = {
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2],
    "P": [0, 1],
    "D": [0, 1],
    "Q": [0, 1]
}


def get_sarimax_params(prefix):
    return {
        "order": (
            st.session_state[f"selected_{prefix}_p"],
            st.session_state[f"selected_{prefix}_d"],
            st.session_state[f"selected_{prefix}_q"]
        ),
        "seasonal_order": (
            st.session_state[f"selected_{prefix}_P"],
            st.session_state[f"selected_{prefix}_D"],
            st.session_state[f"selected_{prefix}_Q"],
            12
        ),
        "enforce_stationarity": False,
        "enforce_invertibility": False
    }


def build_plot(result, plot_label):
    train = result["train"]
    test = result["test"]
    forecast = result["forecast"]
    future_forecast = result.get("future_forecast", None)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train.index,
        y=train.values,
        mode="lines",
        name="Train"
    ))

    fig.add_trace(go.Scatter(
        x=test.index,
        y=test.values,
        mode="lines+markers",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name="Predicted"
    ))

    if future_forecast is not None and not future_forecast.empty:
        fig.add_trace(go.Scatter(
            x=future_forecast.index,
            y=future_forecast.values,
            mode="lines+markers",
            line=dict(dash="dot"),
            name="18-Month Forward Forecast"
        ))

    fig.update_layout(
        title=f"Median Listing Price: Actual vs Predicted ({plot_label})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, width="stretch")


def models_cols(results, plot_label, model):
    col1, col2 = st.columns([3, 1])

    with col2:
        st.header("Model Tuning")

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            st.selectbox(
                "p value",
                options=sarimax_param_grid["p"],
                key=f"selected_{model}_p"
            )
        with row1_col2:
            st.selectbox(
                "d value",
                options=sarimax_param_grid["d"],
                key=f"selected_{model}_d"
            )
        with row1_col3:
            st.selectbox(
                "q value",
                options=sarimax_param_grid["q"],
                key=f"selected_{model}_q"
            )

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            st.selectbox(
                "P value",
                options=sarimax_param_grid["P"],
                key=f"selected_{model}_P"
            )
        with row2_col2:
            st.selectbox(
                "D value",
                options=sarimax_param_grid["D"],
                key=f"selected_{model}_D"
            )
        with row2_col3:
            st.selectbox(
                "Q value",
                options=sarimax_param_grid["Q"],
                key=f"selected_{model}_Q"
            )

        st.divider()
        metrics_display(results["eval_results"])

    with col1:
        build_plot(results, plot_label)


def sarimax_view(data, selected_region, selected_state):
    for model in model_vars:
        for param in sarimax_tuning_features:
            key = f"selected_{model}_{param}"
            if key not in st.session_state:
                st.session_state[key] = sarimax_param_grid[param][1]

    st.header("SARIMAX Forecasting Model")

    with st.expander("📘 Model Overview"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(MODEL_OVERVIEW_MD_SARIMAX_1)
        with col2:
            st.markdown(MODEL_OVERVIEW_MD_SARIMAX_2)

        st.markdown("""
        ---
        # 🧠 Modeling Approaches
        This dashboard compares three different geographical aggregation strategies for the time series:
        """)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown(MODEL_APROACHES_MD_1)
        with col2:
            st.markdown(MODEL_APROACHES_MD_2)
        with col3:
            st.markdown(MODEL_APROACHES_MD_3)

    tab1, tab2, tab3 = st.tabs([
        "Region Model",
        "State Aggregate",
        "US Aggregate"
    ])

    selected_features = ["median_listing_price_x"]

    with tab1:
        with st.spinner("Loading model...", show_time=True):
            result_region = sarimax_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                params=get_sarimax_params("rg_sarimax"),
                level="region",
                region=selected_region,
                test_periods=12
            )
        models_cols(result_region, selected_region, "rg_sarimax")

    with tab2:
        with st.spinner("Loading model...", show_time=True):
            result_state = sarimax_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                params=get_sarimax_params("state_sarimax"),
                level="state",
                state=selected_state,
                test_periods=12
            )
        models_cols(result_state, selected_state.upper(), "state_sarimax")

    with tab3:
        with st.spinner("Loading model...", show_time=True):
            result_us = sarimax_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                params=get_sarimax_params("aggr_sarimax"),
                level="us",
                test_periods=12
            )
        models_cols(result_us, "Entire US", "aggr_sarimax")
        