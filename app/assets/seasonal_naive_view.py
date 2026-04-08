import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from metrics_display import metrics_display

import streamlit as st
import plotly.graph_objects as go
from src.seasonal_naive_model import seasonal_naive_model

MODEL_OVERVIEW_MD_SNAIVE_1 = """
# 🔁 Seasonal Naive Model Overview

**Seasonal Naive (SNaive)** is a simple baseline model that assumes future values will repeat past seasonal patterns.

Instead of learning complex relationships, it:
- copies values from the previous season
- assumes patterns repeat over time (e.g., yearly seasonality)
- requires no training or parameter tuning
- serves as a benchmark for more advanced models
"""

MODEL_OVERVIEW_MD_SNAIVE_2 = """
### ✔ Why it works well here
- Captures **strong seasonal patterns** (common in housing data)
- Extremely **fast and lightweight**
- Provides a **reliable baseline** for comparison
- Easy to interpret and debug
- No risk of overfitting

### ⚠ Limitations
- Cannot capture **trends** (e.g., COVID price surge)
- Ignores **external factors** (rates, unemployment, etc.)
- Assumes past patterns repeat exactly
- Not suitable for complex or changing dynamics
"""

def build_snaive_plot(result, plot_label):
    train = result["train"]
    test = result["test"]
    forecast = result["forecast"]
    future_forecast = result["future_forecast"]
    future_forecast_dates = result["future_forecast_dates"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines", name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode="lines+markers", name="Predicted"))
    fig.add_trace(go.Scatter(
        x=future_forecast_dates,
        y=future_forecast,
        mode="lines+markers",
        line=dict(dash='dot'),
        name="18-Month Forward Forecast"
    ))

    fig.update_layout(
        title=f"Actual vs Predicted ({plot_label})",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width='stretch')

def snaive_cols(result, plot_label):
    col1, col2 = st.columns([3, 1])
    with col1:
        build_snaive_plot(result, plot_label)
    with col2:
        metrics_display(result["eval_results"])

def render_seasonal_naive(data, selected_regions, selected_state):
    with st.container():
        st.header("Seasonal Naive Model")
        
        with st.expander("📘 Model Overview"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(MODEL_OVERVIEW_MD_SNAIVE_1)
            with col2:
                st.markdown(MODEL_OVERVIEW_MD_SNAIVE_2)

        tab1, tab2, tab3 = st.tabs([
            "Region Model",
            "State Aggregate",
            "US Aggregate"
        ])

        target_col = "median_listing_price_x"

        with tab1:
            all_region_results = {}
            for region in selected_regions:
                with st.expander(f"📍 Region Model: {region}", expanded=True):
                    cache_key = f"cache_snaive_rg_{region}"
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = seasonal_naive_model(
                            data,
                            target_col,
                            "county_name_x",
                            region,
                            12
                        )
                    
                    result = st.session_state[cache_key]
                    all_region_results[region] = result["eval_results"]
                    snaive_cols(result, region)

        with tab2:
            cache_key_st = f"cache_snaive_st_{selected_state}"
            if cache_key_st not in st.session_state:
                state_data = data[data["state"] == selected_state].groupby("date")[target_col].mean().reset_index()
                state_data["level"] = "state"
                st.session_state[cache_key_st] = seasonal_naive_model(
                    state_data,
                    target_col,
                    "level",
                    "state",
                    12
                )
            result_state = st.session_state[cache_key_st]
            snaive_cols(result_state, selected_state.upper())

        with tab3:
            cache_key_us = "cache_snaive_us"
            if cache_key_us not in st.session_state:
                us_data = data.groupby("date")[target_col].mean().reset_index()
                us_data["level"] = "us"
                st.session_state[cache_key_us] = seasonal_naive_model(
                    us_data,
                    target_col,
                    "level",
                    "us",
                    12
                )
            result_us = st.session_state[cache_key_us]
            snaive_cols(result_us, "Entire US")

        return {
            "region": all_region_results,
            "state": result_state["eval_results"],
            "us": result_us["eval_results"]
        }
                    