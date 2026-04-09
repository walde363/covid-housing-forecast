import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display
import pandas as pd

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
    "changepoint_prior_scale": [0.05, 0.5],
    "seasonality_prior_scale": [1.0, 10.0],
    "yearly_seasonality": [True]
}



def get_prophet_params(prefix, region=None):
    suffix = f"_{region}" if region else ""
    return {
        "seasonality_mode": str(st.session_state[f"selected_{prefix}{suffix}_seasonality_mode"]),
        "changepoint_prior_scale": float(st.session_state[f"selected_{prefix}{suffix}_changepoint_prior_scale"]),
        "seasonality_prior_scale": float(st.session_state[f"selected_{prefix}{suffix}_seasonality_prior_scale"]),
        "yearly_seasonality": bool(st.session_state[f"selected_{prefix}{suffix}_yearly_seasonality"]),
        "weekly_seasonality": False,
        "daily_seasonality": False
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
            line=dict(dash='dot'),
            name="18-Month Forward Forecast"
        ))

    fig.update_layout(
        title=f"Median Listing Price: Actual vs Predicted ({plot_label}) - Prophet",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        height=700,
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')

def run_tuning(model_prefix, data, target_col, selected_features, level, region=None, state=None, rerun=True):
    param_names = prophet_tuning_features
    combinations = list(itertools.product(*[prophet_param_grid[k] for k in param_names]))
    
    results_list = []
    progress_text = f"Tuning {model_prefix.replace('_', ' ')}..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, combo in enumerate(combinations):
        my_bar.progress((i + 1) / len(combinations), text=f"{progress_text} ({i+1}/{len(combinations)})")
        params = dict(zip(param_names, combo))
        params.update({"weekly_seasonality": False, "daily_seasonality": False})
        
        try:
            res = prophet_model_pipeline(
                target_col=target_col,
                dataset=data,
                selected_cols=selected_features,
                level=level,
                region=region,
                state=state,
                test_periods=12,
                params=params
            )
            rmse = next(m["Value"] for m in res["eval_results"] if m["Metric"] == "RMSE")
            results_list.append({**params, "RMSE": rmse})
        except Exception:
            continue
            
    if results_list:
        df = pd.DataFrame(results_list).sort_values("RMSE")
        res_key = f"{model_prefix}_{region}_tuning_results" if region else f"{model_prefix}_tuning_results"
        st.session_state[res_key] = df
        if rerun:
            st.rerun()


def render_tuning_ui(model, data, selected_features, level, region=None, state=None):
    st.subheader(f"⚙️ Tuning: {region if region else level}")
    suffix = f"_{region}" if region else ""
    paramcol1, paramcol2 = st.columns(2)
    for param in prophet_tuning_features[:2]:
        paramcol1.selectbox(
                f"{param}",
                options=list(prophet_param_grid[param]),
                key=f"selected_{model}{suffix}_{param}"
            )
    for param in prophet_tuning_features[2:]:
        paramcol2.selectbox(
                f"{param}",
                options=list(prophet_param_grid[param]),
                key=f"selected_{model}{suffix}_{param}"
            )
    
    if st.button(f"🚀 Auto-Tune based on {region if region else level}", key=f"tune_btn_{model}_{region}", width='stretch'):
        run_tuning(model, data, "median_listing_price_x", selected_features, level, region, state)

def models_cols(results, plot_label):
    col1, col2 = st.columns([3, 1])
    with col2:
        metrics_display(results["eval_results"])
    with col1:
        build_plot(results, plot_label)


def prophet_view(data, selected_regions, selected_state):
    for model_prefix in model_vars:
        if model_prefix == "rg_prophet":
            for region in selected_regions:
                res_key = f"{model_prefix}_{region}_tuning_results"
                for param in prophet_tuning_features:
                    key = f"selected_{model_prefix}_{region}_{param}"
                    if key not in st.session_state:
                        if res_key in st.session_state and not st.session_state[res_key].empty:
                            val = st.session_state[res_key].iloc[0][param]
                            if param == "yearly_seasonality":
                                val = bool(val)
                            st.session_state[key] = val
                        else:
                            st.session_state[key] = prophet_param_grid[param][0]
        else:
            tuning_results_key = f"{model_prefix}_tuning_results"
            if tuning_results_key in st.session_state and not st.session_state[tuning_results_key].empty:
                best_params = st.session_state[tuning_results_key].iloc[0]
                for param_name in prophet_tuning_features:
                    val = best_params[param_name]
                    if param_name == "yearly_seasonality":
                        val = bool(val)
                    st.session_state[f"selected_{model_prefix}_{param_name}"] = val
            else:
                for a in prophet_tuning_features:
                    key = f"selected_{model_prefix}_{a}"
                    if key not in st.session_state:
                        st.session_state[key] = prophet_param_grid[a][0]

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
        result_region_dict = {}
        for region in selected_regions:
            params_rg = get_prophet_params("rg_prophet", region=region)
            cache_key_rg = f"cache_prophet_rg_{region}_{str(params_rg)}"
            if cache_key_rg not in st.session_state:
                with st.spinner(f"Loading {region} model...", show_time=True):
                    st.session_state[cache_key_rg] = prophet_model_pipeline(
                        target_col="median_listing_price_x", dataset=data, selected_cols=selected_features,
                        level="region", region=region, test_periods=12, params=params_rg
                    )
            res_rg = st.session_state[cache_key_rg]
            result_region_dict[region] = res_rg["eval_results"]
            with st.expander(f"📍 Region Model: {region}", expanded=True):
                render_tuning_ui("rg_prophet", data, selected_features, "region", region=region)
                st.divider()
                models_cols(res_rg, region)

    with tab2:
        params_st = get_prophet_params("state_prophet")
        cache_key_st = f"cache_prophet_st_{selected_state}_{str(params_st)}"
        if cache_key_st not in st.session_state:
            with st.spinner("Loading model...", show_time=True):
                st.session_state[cache_key_st] = prophet_model_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=selected_features,
                    level="state",
                    state=selected_state,
                    test_periods=12,
                    params=params_st
                )
        result_state = st.session_state[cache_key_st]

        render_tuning_ui("state_prophet", data, selected_features, "state", state=selected_state)
        st.divider()
        models_cols(result_state, selected_state.upper())

    with tab3:
        params_us = get_prophet_params("aggr_prophet")
        cache_key_us = f"cache_prophet_us_{str(params_us)}"
        if cache_key_us not in st.session_state:
            with st.spinner("Loading model...", show_time=True):
                st.session_state[cache_key_us] = prophet_model_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=selected_features,
                    level="us",
                    test_periods=12,
                    params=params_us
                )
        result_us = st.session_state[cache_key_us]

        render_tuning_ui("aggr_prophet", data, selected_features, "us")
        st.divider()
        models_cols(result_us, "Entire US")

    st.divider()
    st.header("📊 Prophet Tuning Results")
    for m_var in model_vars:
        res_key = f"{m_var}_tuning_results"
        if res_key in st.session_state:
            with st.expander(f"Tuning History: {m_var.replace('_', ' ').title()}", expanded=True):
                st.dataframe(st.session_state[res_key], width='stretch')

    return {
        "region": result_region_dict,
        "state": result_state["eval_results"],
        "us": result_us["eval_results"]
    }
