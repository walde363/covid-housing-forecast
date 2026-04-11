import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display
import pandas as pd

import streamlit as st
import plotly.graph_objects as go
from src.rf_model import rf_model_pipeline
from src.rf_panel_pipeline import rf_panel_pipeline

MODEL_OVERVIEW_MD_1 = """
# 🌲 Random Forest Model Overview

A **Random Forest Regressor** is an ensemble model that combines multiple decision trees to make predictions.

Instead of relying on a single model, it:
- builds many trees on different subsets of the data
- averages their predictions for better accuracy
"""
MODEL_OVERVIEW_MD_2 = """
### ✔ Why it works well here
- Captures **nonlinear relationships**
- Handles **many features** (inventory, economic indicators, etc.)
- Learns **interactions between variables**
- Robust to noise

### ⚠ Limitations
- Does not inherently understand time → requires **lag features**
- Less interpretable than simple models
"""

MODEL_APROACHES_MD_1 = """
### 📍 Region Model
**Train:** Selected region only  
**Goal:** Capture local behavior  

✔ Pros:
- Highly localized
- Captures regional trends  

⚠ Cons:
- Limited data
- May miss broader patterns
"""
MODEL_APROACHES_MD_2 = """
### 🏙️ State Aggregate Model
**Train:** Aggregated state-level data  
**Goal:** Understand state trends  

✔ Pros:
- More stable than region-level  
- More data  

⚠ Cons:
- Loses regional detail  
- Represents averages, not specific regions  
"""
MODEL_APROACHES_MD_3 = """
### US Aggregate Model
**Train:** Aggregated US-level data  
**Goal:** Capture national trends  

✔ Pros:
- Very stable  
- Captures macro patterns  

⚠ Cons:
- Too general for local predictions  
"""
MODEL_APROACHES_MD_4 = """
### 🌐 US Train → Region Predict
**Train:** All US regions (panel data)  
**Predict:** Selected region only  

✔ Pros:
- Much larger dataset  
- Learns general housing dynamics  
- Often best for ML models  

⚠ Cons:
- More complex  
- Requires careful feature engineering
"""

model_vars = ["rg_rf", "state_rf", "aggr_rf", "ust_rf"]

rf_tuning_features = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap"
]

defaults = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "bootstrap": [True]
}



def get_rf_params(prefix, region=None):
    suffix = f"_{region}" if region else ""
    md = st.session_state[f"selected_{prefix}{suffix}_max_depth"]
    return {
        "n_estimators": int(st.session_state[f"selected_{prefix}{suffix}_n_estimators"]),
        "max_depth": int(md) if (md is not None and not pd.isna(md)) else None,
        "min_samples_split": int(st.session_state[f"selected_{prefix}{suffix}_min_samples_split"]),
        "min_samples_leaf": int(st.session_state[f"selected_{prefix}{suffix}_min_samples_leaf"]),
        "max_features": st.session_state[f"selected_{prefix}{suffix}_max_features"],
        "bootstrap": bool(st.session_state[f"selected_{prefix}{suffix}_bootstrap"]),
        "random_state": 42,
        "n_jobs": -1
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
        title=f"Median Listing Price: Actual vs Predicted ({plot_label}) - Random Forest",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark",
        height=700,
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width="stretch")


def run_tuning(model_prefix, data, target_col, selected_features, level, region=None, state=None, rerun=True):
    param_names = rf_tuning_features
    combinations = list(itertools.product(*[defaults[k] for k in param_names]))
    
    results_list = []
    progress_text = f"Tuning {model_prefix.replace('_', ' ')}..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, combo in enumerate(combinations):
        my_bar.progress((i + 1) / len(combinations), text=f"{progress_text} ({i+1}/{len(combinations)})")
        params = dict(zip(param_names, combo))
        params["random_state"] = 42
        params["n_jobs"] = -1
        
        try:
            if model_prefix == "ust_rf":
                res = rf_panel_pipeline(
                    target_col=target_col,
                    dataset=data,
                    selected_cols=selected_features,
                    selected_region=region,
                    test_periods=12,
                    params=params
                )
            else:
                res = rf_model_pipeline(
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
        
        # Update UI state with best parameters found
        best_params = df.iloc[0]
        suffix = f"_{region}" if region else ""
        for param in param_names:
            val = best_params[param]
            if param in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                if not pd.isna(val):
                    val = int(val)
            st.session_state[f"selected_{model_prefix}{suffix}_{param}"] = None if pd.isna(val) else val
            
        if rerun:
            st.rerun()

def render_tuning_ui(model, data, selected_features, level, region=None, state=None):
    st.subheader(f"⚙️ Tuning: {region if region else level}")
    suffix = f"_{region}" if region else ""

    # Handle tuning trigger before widgets are rendered
    trigger_key = f"trigger_tune_{model}{suffix}"
    if st.session_state.get(trigger_key, False):
        run_tuning(model, data, "median_listing_price_x", selected_features, level, region, state, rerun=False)
        st.session_state[trigger_key] = False
        st.rerun()

    paramcol1, paramcol2 = st.columns(2)
    for param in rf_tuning_features[:3]:
        paramcol1.selectbox(
                f"{param}",
                options=list(defaults[param]),
                key=f"selected_{model}{suffix}_{param}"
            )
    for param in rf_tuning_features[3:]:
        paramcol2.selectbox(
                f"{param}",
                options=list(defaults[param]),
                key=f"selected_{model}{suffix}_{param}"
            )
    
    if st.button(f"🚀 Auto-Tune based on {region if region else level}", key=f"tune_btn_{model}_{region}", width='stretch'):
        st.session_state[trigger_key] = True
        st.rerun()

def models_cols(results, plot_label):
    col1, col2 = st.columns([3, 1])
    with col2:
        metrics_display(results["eval_results"])
    with col1:
        build_plot(results, plot_label)


def rf_view(data, selected_regions, selected_state):
    for model_prefix in model_vars:
        if model_prefix in ["rg_rf", "ust_rf"]:
            for region in selected_regions:
                res_key = f"{model_prefix}_{region}_tuning_results"
                for param in rf_tuning_features:
                    key = f"selected_{model_prefix}_{region}_{param}"
                    if key not in st.session_state:
                        if res_key in st.session_state and not st.session_state[res_key].empty:
                            val = st.session_state[res_key].iloc[0][param]
                            if param in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                                if not pd.isna(val):
                                    val = int(val)
                            st.session_state[key] = None if pd.isna(val) else val
                        else:
                            st.session_state[key] = defaults[param][0]
        else:
            tuning_results_key = f"{model_prefix}_tuning_results"
            if tuning_results_key in st.session_state and not st.session_state[tuning_results_key].empty:
                best_params = st.session_state[tuning_results_key].iloc[0]
                for param_name in rf_tuning_features:
                    val = best_params[param_name]
                    if param_name in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                        if not pd.isna(val):
                            val = int(val)
                    st.session_state[f"selected_{model_prefix}_{param_name}"] = None if pd.isna(val) else val
            else:
                for a in rf_tuning_features:
                    key = f"selected_{model_prefix}_{a}"
                    if key not in st.session_state:
                        st.session_state[key] = defaults[a][0]

    st.header("Random Forest Regressor Model")
    
    result_region_dict = {}
    result_ust_dict = {}
    result_state = {"eval_results": []}
    result_us = {"eval_results": []}
    
    with st.expander("📘 Model Overview"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(MODEL_OVERVIEW_MD_1)
        with col2:
            st.markdown(MODEL_OVERVIEW_MD_2)
        st.markdown("""
                    ---
                    # 🧠 Modeling Approaches
                    This dashboard compares four different modeling strategies:
                    """)
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.markdown(MODEL_APROACHES_MD_1)
        with col2:
            st.markdown(MODEL_APROACHES_MD_2)
        with col3:
            st.markdown(MODEL_APROACHES_MD_3)
        with col4:
            st.markdown(MODEL_APROACHES_MD_4) 

    selected_features = [
        "median_listing_price_x",
        "active_listing_count",
        "new_listing_count",
        "pending_ratio",
        "price_reduced_share",
        "Unemployment_Rate",
        "Earnings"
    ]

    panel_features = [
        "active_listing_count",
        "new_listing_count",
        "pending_ratio",
        "price_reduced_share",
        "Unemployment_Rate",
        "Earnings",
        "Investor Purchases",
        "Investor Market Share"
    ]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Region Model",
        "State Aggregate",
        "US Aggregate",
        "US Train → Region Predict"
    ])

    with tab1:
        result_region_dict = {}
        for region in selected_regions:
            params_rg = get_rf_params("rg_rf", region=region)
            cache_key_rg = f"cache_rf_rg_{region}_{str(params_rg)}"
            if cache_key_rg not in st.session_state:
                with st.spinner(f"Loading {region} model...", show_time=True):
                    st.session_state[cache_key_rg] = rf_model_pipeline(
                        target_col="median_listing_price_x", dataset=data, selected_cols=selected_features,
                        level="region", region=region, test_periods=12, params=params_rg
                    )
            res_rg = st.session_state[cache_key_rg]
            result_region_dict[region] = res_rg["eval_results"]
            with st.expander(f"📍 Region Model: {region}", expanded=True):
                render_tuning_ui("rg_rf", data, selected_features, "region", region=region)
                st.divider()
                models_cols(res_rg, region)

    with tab2:
        params_st = get_rf_params("state_rf")
        cache_key_st = f"cache_rf_st_{selected_state}_{str(params_st)}"
        if cache_key_st not in st.session_state:
            with st.spinner("Loading model...", show_time=True):
                st.session_state[cache_key_st] = rf_model_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=selected_features,
                    level="state",
                    state=selected_state,
                    test_periods=12,
                    params=params_st
                )
        result_state = st.session_state[cache_key_st]

        render_tuning_ui("state_rf", data, selected_features, "state", state=selected_state)
        st.divider()
        models_cols(result_state, selected_state.upper())

    with tab3:
        params_us = get_rf_params("aggr_rf")
        cache_key_us = f"cache_rf_us_{str(params_us)}"
        if cache_key_us not in st.session_state:
            with st.spinner("Loading model...", show_time=True):
                st.session_state[cache_key_us] = rf_model_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=selected_features,
                    level="us",
                    test_periods=12,
                    params=params_us
                )
        result_us = st.session_state[cache_key_us]

        render_tuning_ui("aggr_rf", data, selected_features, "us")
        st.divider()
        models_cols(result_us, "Entire US")

    with tab4:
        result_ust_dict = {}
        st.warning('WARNING: Expected time to run is 3 minutes', icon="⚠️")
        for region in selected_regions:
            params_ust = get_rf_params("ust_rf", region=region)
            cache_key_ust = f"cache_rf_ust_{region}_{str(params_ust)}"
            if cache_key_ust not in st.session_state:
                with st.spinner(f"Loading {region} Panel model...", show_time=True):
                    st.session_state[cache_key_ust] = rf_panel_pipeline(
                        target_col="median_listing_price_x", dataset=data, selected_cols=panel_features,
                        selected_region=region, test_periods=12, params=params_ust
                    )
            res_ust = st.session_state[cache_key_ust]
            result_ust_dict[region] = res_ust["eval_results"]
            with st.expander(f"🌐 US Train → Region: {region}", expanded=True):
                render_tuning_ui("ust_rf", data, panel_features, "region", region=region)
                st.divider()
                models_cols(res_ust, f"US Train → {region}")

    st.divider()
    st.header("📊 RF Tuning Results")
    for m_var in model_vars:
        res_key = f"{m_var}_tuning_results"
        if res_key in st.session_state:
            with st.expander(f"Tuning History: {m_var.replace('_', ' ').title()}", expanded=True):
                st.dataframe(st.session_state[res_key], width='stretch')

    return {
        "region": {**result_region_dict, **{f"{k} (Panel)": v for k, v in result_ust_dict.items()}},
        "state": result_state["eval_results"],
        "us": result_us["eval_results"]
    }
        