import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display

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
    "n_estimators": [300, 100, 500],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [5, 2, 10],
    "min_samples_leaf": [2, 1, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}



def get_rf_params(prefix):
    return {
        "n_estimators": st.session_state[f"selected_{prefix}_n_estimators"],
        "max_depth": st.session_state[f"selected_{prefix}_max_depth"],
        "min_samples_split": st.session_state[f"selected_{prefix}_min_samples_split"],
        "min_samples_leaf": st.session_state[f"selected_{prefix}_min_samples_leaf"],
        "max_features": st.session_state[f"selected_{prefix}_max_features"],
        "bootstrap": st.session_state[f"selected_{prefix}_bootstrap"],
        "random_state": 42,
        "n_jobs": 1
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
        st.header("Model Tunning")
        paramcol1, paramcol2 = st.columns(2)
        for param in rf_tuning_features[:3]:
            paramcol1.selectbox(
                    f"{param}",
                    options=list(defaults[param]),
                    key=f"selected_{model}_{param}"
                )
        for param in rf_tuning_features[3:]:
            paramcol2.selectbox(
                    f"{param}",
                    options=list(defaults[param]),
                    key=f"selected_{model}_{param}"
                )
        st.divider()
        metrics_display(results["eval_results"])
    
    with col1:
        build_plot(results, plot_label)


def rf_view(data, selected_region, selected_state):
    for i in model_vars:
        for a in rf_tuning_features:
            key = f"selected_{i}_{a}"
            if key not in st.session_state:
                st.session_state[key] = defaults[a][0]
                
    st.header("Random Forest Regressor Model")
    
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
        result_region = rf_model_pipeline(
            target_col="median_listing_price_x",
            dataset=data,
            selected_cols=selected_features,
            level="region",
            region=selected_region,
            test_periods=12,
            params=get_rf_params("rg_rf")
        )
        models_cols(result_region, selected_region, "rg_rf")

    with tab2:
        result_state = rf_model_pipeline(
            target_col="median_listing_price_x",
            dataset=data,
            selected_cols=selected_features,
            level="state",
            state=selected_state,
            test_periods=12,
            params = get_rf_params("state_rf")
        )
        models_cols(result_state, selected_state.upper(), "state_rf")

    with tab3:
        result_us = rf_model_pipeline(
            target_col="median_listing_price_x",
            dataset=data,
            selected_cols=selected_features,
            level="us",
            test_periods=12,
            params = get_rf_params("aggr_rf")
        )
        models_cols(result_us, "Entire US", "aggr_rf")

    with tab4:
        st.warning('WARNING: Expected time to run is 3 minutes', icon="⚠️")
        if st.button("Run Model"):
            with st.spinner("Loading model...", show_time=True):
                rf_result = rf_panel_pipeline(
                    target_col="median_listing_price_x",
                    dataset=data,
                    selected_cols=panel_features,
                    selected_region=selected_region,
                    test_periods=12,
                    params=get_rf_params("ust_rf")
                )
            models_cols(rf_result, f"US Train → {selected_region}", "ust_rf")
        
        