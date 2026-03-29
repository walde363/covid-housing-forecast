import streamlit as st
import plotly.graph_objects as go
from src.seasonal_naive_model import seasonal_naive_model


def render_seasonal_naive(filtered_data):
    with st.container():
        st.header("Seasonal Naive Model")

        regions = sorted(filtered_data["county_name_x"].dropna().unique().tolist())

        if not regions:
            st.warning("No regions found for the selected state.")
            return

        selected_region = st.selectbox(
            "Select Region",
            regions,
            key="seasonal_naive_region"
        )

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
            

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Model Metrics")
            for key, value in result["eval_results"].items():
                if isinstance(value, (int, float)):
                    st.write(f"{key}: {value:.4f}")
                else:
                    st.write(f"{key}: {value}")
                    