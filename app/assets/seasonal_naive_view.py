import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))

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

def render_seasonal_naive(filtered_data, selected_region):
    with st.container():
        st.header("Seasonal Naive Model")
        
        with st.expander("📘 Model Overview"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(MODEL_OVERVIEW_MD_SNAIVE_1)
            with col2:
                st.markdown(MODEL_OVERVIEW_MD_SNAIVE_2)

        result = seasonal_naive_model(
            filtered_data,
            "median_listing_price_x",
            "county_name_x",
            selected_region,
            12
        )

        train = result["train"]
        test = result["test"]
        forecast = result["forecast"]
        future_forecast = result["future_forecast"]
        future_forecast_dates = result["future_forecast_dates"]
    
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
            
            fig.add_trace(go.Scatter(
                x=future_forecast_dates,
                y=future_forecast,
                mode="lines+markers",
                line=dict(dash='dot'),
                name="18-Month Forward Forecast"
            ))

            fig.update_layout(
                title=f"Median Listing Price: Actual vs Predicted ({selected_region})",
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

        with col2:
            metrics_display(result["eval_results"])
                    