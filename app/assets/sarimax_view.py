import streamlit as st
import plotly.graph_objects as go
from src.sarimax_model import sarimax_model

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
- Incorporates **external drivers** (e.g., mortgage rates, unemployment)
- Strong for **time series forecasting problems**
- Provides **interpretable statistical structure**
- Well-suited for economic and housing data

### ⚠ Limitations
- Assumes mostly **linear relationships**
- Requires **parameter tuning** (p, d, q, P, D, Q)
- Can be slower on **large datasets**
- Sensitive to data quality and missing values
"""



def sarimax_view(filtered_data, selected_region):
    with st.container():
        st.header("SARIMAX Model")
        with st.expander("📘 Model Overview"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(MODEL_OVERVIEW_MD_SARIMAX_1)
            with col2:
                st.markdown(MODEL_OVERVIEW_MD_SARIMAX_2)
        
        result = sarimax_model(filtered_data, "county_name_x", selected_region,"median_listing_price_x", 12)
        
        train = result["train"]
        test = result["test"]
        forecast = result["forecast"]

        col1, col2 = st.columns([3, 1])

        with col1:
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
                x=test.index,
                y=forecast,
                mode="lines+markers",
                name="Predicted"
            ))

            fig.update_layout(
                title=f"Median Listing Price: Actual vs Predicted ({selected_region})",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                template="plotly_dark",
                paper_bgcolor="#1E293B",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
            )
            

            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Model Metrics")
            for key, value in result["eval_results"].items():
                if isinstance(value, (int, float)):
                    st.write(f"{key}: {value:.4f}")
                else:
                    st.write(f"{key}: {value}")