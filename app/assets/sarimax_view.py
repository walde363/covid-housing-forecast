import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display
import pandas as pd

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
    "p": [0, 1],
    "d": [1],
    "q": [0, 1],
    "P": [0, 1],
    "D": [1],
    "Q": [0]
}


def get_sarimax_params(prefix, region=None):
    suffix = f"_{region}" if region else ""
    return {
        "order": (
            int(st.session_state[f"selected_{prefix}{suffix}_p"]),
            int(st.session_state[f"selected_{prefix}{suffix}_d"]),
            int(st.session_state[f"selected_{prefix}{suffix}_q"])
        ),
        "seasonal_order": (
            int(st.session_state[f"selected_{prefix}{suffix}_P"]),
            int(st.session_state[f"selected_{prefix}{suffix}_D"]),
            int(st.session_state[f"selected_{prefix}{suffix}_Q"]),
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
        title=f"Median Listing Price: Actual vs Predicted ({plot_label}) - SARIMAX",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
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

def run_tuning(model_prefix, data, target_col, selected_features, level, region=None, state=None, rerun=True):
    param_names = ["p", "d", "q", "P", "D", "Q"]
    combinations = list(itertools.product(*[sarimax_param_grid[k] for k in param_names]))
    
    results_list = []
    
    progress_text = f"Tuning {model_prefix.replace('_', ' ')}..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, combo in enumerate(combinations):
        my_bar.progress((i + 1) / len(combinations), text=f"{progress_text} ({i+1}/{len(combinations)})")
        params = {
            "order": combo[:3],
            "seasonal_order": (*combo[3:], 12),
            "enforce_stationarity": False,
            "enforce_invertibility": False
        }
        try:
            res = sarimax_model_pipeline(
                target_col=target_col,
                dataset=data,
                selected_cols=selected_features,
                params=params,
                level=level,
                region=region,
                state=state,
                test_periods=12
            )
            rmse = next(m["Value"] for m in res["eval_results"] if m["Metric"] == "RMSE")
            results_list.append({
                "p": combo[0], "d": combo[1], "q": combo[2],
                "P": combo[3], "D": combo[4], "Q": combo[5],
                "RMSE": rmse
            })
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

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.selectbox(
            "p value",
            options=sarimax_param_grid["p"],
            key=f"selected_{model}{suffix}_p"
        )
    with row1_col2:
        st.selectbox(
            "d value",
            options=sarimax_param_grid["d"],
            key=f"selected_{model}{suffix}_d"
        )
    with row1_col3:
        st.selectbox(
            "q value",
            options=sarimax_param_grid["q"],
            key=f"selected_{model}{suffix}_q"
        )

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.selectbox(
            "P value",
            options=sarimax_param_grid["P"],
            key=f"selected_{model}{suffix}_P"
        )
    with row2_col2:
        st.selectbox(
            "D value",
            options=sarimax_param_grid["D"],
            key=f"selected_{model}{suffix}_D"
        )
    with row2_col3:
        st.selectbox(
            "Q value",
            options=sarimax_param_grid["Q"],
            key=f"selected_{model}{suffix}_Q"
        )
    
    if st.button(f"🚀 Auto-Tune based on {region if region else level}", key=f"tune_btn_{model}{suffix}", width='stretch'):
        run_tuning(model, data, "median_listing_price_x", selected_features, level, region, state)

def models_cols(results, plot_label):
    col1, col2 = st.columns([3, 1])
    with col2:
        metrics_display(results["eval_results"])
    with col1:
        build_plot(results, plot_label)


def sarimax_view(data, selected_regions, selected_state):
    for model_prefix in model_vars:
        if model_prefix == "rg_sarimax":
            for region in selected_regions:
                res_key = f"{model_prefix}_{region}_tuning_results"
                for param in sarimax_tuning_features:
                    key = f"selected_{model_prefix}_{region}_{param}"
                    if key not in st.session_state:
                        if res_key in st.session_state and not st.session_state[res_key].empty:
                            st.session_state[key] = int(st.session_state[res_key].iloc[0][param])
                        else:
                            st.session_state[key] = sarimax_param_grid[param][0]
        else:
            tuning_results_key = f"{model_prefix}_tuning_results"
            if tuning_results_key in st.session_state and not st.session_state[tuning_results_key].empty:
                best_params = st.session_state[tuning_results_key].iloc[0]
                for param_name in sarimax_tuning_features:
                    st.session_state[f"selected_{model_prefix}_{param_name}"] = int(best_params[param_name])
            else:
                for param_name in sarimax_tuning_features:
                    key = f"selected_{model_prefix}_{param_name}"
                    if key not in st.session_state:
                        st.session_state[key] = sarimax_param_grid[param_name][0]

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
        result_region_dict = {}
        for region in selected_regions:
            params_rg = get_sarimax_params("rg_sarimax", region=region)
            cache_key_rg = f"cache_sarimax_rg_{region}_{str(params_rg)}"
            if cache_key_rg not in st.session_state:
                with st.spinner(f"Loading {region} model...", show_time=True):
                    st.session_state[cache_key_rg] = sarimax_model_pipeline(
                        target_col="median_listing_price_x", dataset=data, selected_cols=selected_features,
                        params=params_rg, level="region", region=region, test_periods=12
                    )
            res_rg = st.session_state[cache_key_rg]
            result_region_dict[region] = res_rg["eval_results"]
            with st.expander(f"📍 Region Model: {region}", expanded=True):
                render_tuning_ui("rg_sarimax", data, selected_features, "region", region=region)
                st.divider()
                models_cols(res_rg, region)

    with tab2:
        params_st = get_sarimax_params("state_sarimax")
        cache_key_st = f"cache_sarimax_st_{selected_state}_{str(params_st)}"
        if cache_key_st not in st.session_state:
            with st.spinner("Loading model...", show_time=True):
                st.session_state[cache_key_st] = sarimax_model_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=selected_features,
                    params=params_st,
                    level="state",
                    state=selected_state,
                    test_periods=12
                )
        result_state = st.session_state[cache_key_st]

        render_tuning_ui("state_sarimax", data, selected_features, "state", state=selected_state)
        st.divider()
        models_cols(result_state, selected_state.upper())

    with tab3:
        params_us = get_sarimax_params("aggr_sarimax")
        cache_key_us = f"cache_sarimax_us_{str(params_us)}"
        if cache_key_us not in st.session_state:
            with st.spinner("Loading model...", show_time=True):
                st.session_state[cache_key_us] = sarimax_model_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=selected_features,
                    params=params_us,
                    level="us",
                    test_periods=12
                )
        result_us = st.session_state[cache_key_us]

        render_tuning_ui("aggr_sarimax", data, selected_features, "us")
        st.divider()
        models_cols(result_us, "Entire US")

    st.divider()
    st.header("📊 SARIMAX Tuning Results")
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
        