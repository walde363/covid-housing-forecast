import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display

import streamlit as st
import plotly.graph_objects as go
from src.prophet_model import prophet_model_pipeline

MODEL_OVERVIEW_MD_1 = """
# 📈 Prophet Model Overview

**Prophet** is a forecasting procedure developed by Meta (formerly Facebook). It is specifically designed for time series forecasting based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

- Robust to missing data and shifts in the trend
- Excellent for univariate time series data
- Automatically handles outliers well
"""

MODEL_OVERVIEW_MD_2 = """
### ✔ Why it works well here
- Out-of-the-box support for **seasonality** (yearly, monthly patterns)
- Strong baseline performance without deep feature engineering
- Handles Covid-19 structural breaks (via changepoints) gracefully
- Explainable components (trend, seasonality)

### ⚠ Limitations
- Purely univariate (unless adding specific regressors)
- Doesn't extract cross-regional interactions (like panel data architectures)
- Weak at handling extremely granular local spikes without many data points
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

model_vars = ["rg_prophet", "state_prophet", "aggr_prophet"]

prophet_tuning_features = [
    "seasonality_mode",
    "changepoint_prior_scale",
    "seasonality_prior_scale",
    "yearly_seasonality"
]

prophet_param_grid = {
    "seasonality_mode": ["additive", "multiplicative"],
    "changepoint_prior_scale": [0.05, 0.1, 0.5],
    "seasonality_prior_scale": [10.0, 1.0, 0.1],
    "yearly_seasonality": [True, False]
}

for i in model_vars:
    for a in prophet_tuning_features:
        key = f"selected_{i}_{a}"
        if key not in st.session_state:
            st.session_state[key] = prophet_param_grid[a][0]

def get_prophet_params(prefix):
    return {
        "seasonality_mode": st.session_state[f"selected_{prefix}_seasonality_mode"],
        "changepoint_prior_scale": st.session_state[f"selected_{prefix}_changepoint_prior_scale"],
        "seasonality_prior_scale": st.session_state[f"selected_{prefix}_seasonality_prior_scale"],
        "yearly_seasonality": st.session_state[f"selected_{prefix}_yearly_seasonality"],
        "weekly_seasonality": False,
        "daily_seasonality": False
    }

def build_plot(result, plot_label):
    train = result["train"]
    test = result["test"]
    forecast = result["forecast"]

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

    fig.update_layout(
        title=f"Median Listing Price: Actual vs Predicted ({plot_label})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        height=700,
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')


def models_cols(results, plot_label, model):
    col1, col2 = st.columns([3, 1])

    with col2:
        st.header("Model Tuning")
        paramcol1, paramcol2 = st.columns(2)
        for param in prophet_tuning_features[:2]:
            paramcol1.selectbox(
                    f"{param}",
                    options=list(prophet_param_grid[param]),
                    key=f"selected_{model}_{param}"
                )
        for param in prophet_tuning_features[2:]:
            paramcol2.selectbox(
                    f"{param}",
                    options=list(prophet_param_grid[param]),
                    key=f"selected_{model}_{param}"
                )
        st.divider()
        metrics_display(results["eval_results"])
        
    with col1:
        build_plot(results, plot_label)


def prophet_view(data, selected_region, selected_state):
    st.header("Prophet Forecasting Model")
    
    with st.expander("📘 Model Overview"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(MODEL_OVERVIEW_MD_1)
        with col2:
            st.markdown(MODEL_OVERVIEW_MD_2)
        st.markdown("""
                    ---
                    # 🧠 Modeling Approaches
                    This dashboard compares three different geographical aggregation strategies for the univariate time-series:
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

    # Note: Prophet is purely univariate here, so selected_cols is largely unused except for the target, but we define it explicitly.
    selected_features = ["median_listing_price_x"]

    with tab1:
        with st.spinner("Loading model...", show_time=True):
            result_region = prophet_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="region",
                region=selected_region,
                test_periods=12,
                params=get_prophet_params("rg_prophet")
            )
        models_cols(result_region, selected_region, "rg_prophet")

    with tab2:
        with st.spinner("Loading model...", show_time=True):
            result_state = prophet_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="state",
                state=selected_state,
                test_periods=12,
                params=get_prophet_params("state_prophet")
            )
        models_cols(result_state, selected_state.upper(), "state_prophet")

    with tab3:
        with st.spinner("Loading model...", show_time=True):
            result_us = prophet_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="us",
                test_periods=12,
                params=get_prophet_params("aggr_prophet")
            )
        models_cols(result_us, "Entire US", "aggr_prophet")
