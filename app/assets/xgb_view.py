import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

from metrics_display import metrics_display

import streamlit as st
import plotly.graph_objects as go
from src.xgb_model import xgb_model_pipeline
from src.xgb_panel_pipeline import xgb_panel_pipeline

MODEL_OVERVIEW_MD_1 = """
# 🚀 XGBoost Model Overview

**XGBoost (Extreme Gradient Boosting)** is an advanced ensemble model that builds decision trees sequentially to improve predictions.

Instead of building trees independently like Random Forest, it:
- builds trees one at a time
- each new tree focuses on correcting the errors of previous ones
- combines all trees into a strong predictive model
"""

MODEL_OVERVIEW_MD_2 = """
### ✔ Why it works well here
- Captures **complex nonlinear relationships**
- Excels at learning **small patterns and corrections**
- Often achieves **higher accuracy** than Random Forest
- Handles **feature interactions** very effectively
- Works well with structured/tabular data

### ⚠ Limitations
- More sensitive to **hyperparameters**
- Can **overfit** if not tuned properly
- Training is more complex than Random Forest
- Still requires **lag features** for time-based data
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

model_vars = ["rg_xgb", "state_xgb", "aggr_xgb", "ust_xgb"]

xgb_tuning_features = [
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree"
]

xgb_param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

for i in model_vars:
    for a in xgb_tuning_features:
        key = f"selected_{i}_{a}"
        if key not in st.session_state:
            st.session_state[key] = xgb_param_grid[a][0]

def get_xgb_params(prefix):
    return {
        "n_estimators": st.session_state[f"selected_{prefix}_n_estimators"],
        "learning_rate": st.session_state[f"selected_{prefix}_learning_rate"],
        "max_depth": st.session_state[f"selected_{prefix}_max_depth"],
        "min_child_weight": st.session_state[f"selected_{prefix}_min_child_weight"],
        "subsample": st.session_state[f"selected_{prefix}_subsample"],
        "colsample_bytree": st.session_state[f"selected_{prefix}_colsample_bytree"],
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

    with col1:
        build_plot(results, plot_label)

    with col2:
        st.header("Model Tunning")
        paramcol1, paramcol2 = st.columns(2)
        for param in xgb_tuning_features[:3]:
            paramcol1.selectbox(
                    f"{param}",
                    options=list(xgb_param_grid[param]),
                    key=f"selected_{model}_{param}"
                )
        for param in xgb_tuning_features[3:]:
            paramcol2.selectbox(
                    f"{param}",
                    options=list(xgb_param_grid[param]),
                    key=f"selected_{model}_{param}"
                )
        st.divider()
        metrics_display(results["eval_results"])


def xgb_view(data, selected_region, selected_state):
    st.header("XGBoost Regressor Model")
    
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
        with st.spinner("Loading model...", show_time=True):
            result_region = xgb_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="region",
                region=selected_region,
                test_periods=12,
                params=get_xgb_params("rg_xgb")
            )
        models_cols(result_region, selected_region, "rg_xgb")

    with tab2:
        with st.spinner("Loading model...", show_time=True):
            result_state = xgb_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="state",
                state=selected_state,
                test_periods=12,
                params=get_xgb_params("state_xgb")
            )
        models_cols(result_state, selected_state.upper(), "state_xgb")

    with tab3:
        with st.spinner("Loading model...", show_time=True):
            result_us = xgb_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="us",
                test_periods=12,
                params=get_xgb_params("aggr_xgb")
            )
        models_cols(result_us, "Entire US", "aggr_xgb")

    with tab4:
        with st.spinner("Loading model...", show_time=True):
            rf_result = xgb_panel_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=panel_features,
                selected_region=selected_region,
                test_periods=12,
                params=get_xgb_params("ust_xgb")
            )
        models_cols(rf_result, f"US Train → {selected_region}", "ust_xgb")
        