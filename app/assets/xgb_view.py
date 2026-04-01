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
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width='stretch')


def models_cols(results, plot_label):
    col1, col2 = st.columns([3, 1])

    with col1:
        build_plot(results, plot_label)

    with col2:
        st.subheader("Model Metrics")
        for key, value in results["eval_results"].items():
            if isinstance(value, (int, float)):
                st.write(f"{key}: {value:.4f}")
            else:
                st.write(f"{key}: {value}")


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
                test_periods=12
            )
        models_cols(result_region, selected_region)

    with tab2:
        with st.spinner("Loading model...", show_time=True):
            result_state = xgb_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="state",
                state=selected_state,
                test_periods=12
            )
        models_cols(result_state, selected_state.upper())

    with tab3:
        with st.spinner("Loading model...", show_time=True):
            result_us = xgb_model_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=selected_features,
                level="us",
                test_periods=12
            )
        models_cols(result_us, "Entire US")

    with tab4:
        with st.spinner("Loading model...", show_time=True):
            rf_result = xgb_panel_pipeline(
                target_col="median_listing_price_x",
                dataset=data,
                selected_cols=panel_features,
                selected_region=selected_region,
                test_periods=12
            )
        models_cols(rf_result, f"US Train → {selected_region}")
        