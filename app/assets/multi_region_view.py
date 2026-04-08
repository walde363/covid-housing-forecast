import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.xgb_model import xgb_model_pipeline
from src.rf_model import rf_model_pipeline
from assets.xgb_view import get_xgb_params
from assets.rf_view import get_rf_params

def render_multi_region(data, all_regions):
    st.header("🌎 Multi-Region Analysis")
    st.markdown("""
    Compare how a specific model configuration performs across multiple regions. 
    This helps identify if the model generalizes well across different local markets.
    """)

    col1, col2 = st.columns(2)
    with col1:
        selected_model_type = st.radio(
            "Select Model to Benchmark",
            ["XGBoost", "Random Forest"],
            horizontal=True
        )
    with col2:
        compare_regions = st.multiselect(
            "Select Regions to Compare",
            options=all_regions,
            default=all_regions[:3] if len(all_regions) >= 3 else all_regions
        )

    if not compare_regions:
        st.warning("Please select at least one region.")
        return

    # Standard features used in ML views
    ml_features = [
        "median_listing_price_x", "active_listing_count", "new_listing_count",
        "pending_ratio", "price_reduced_share", "Unemployment_Rate", "Earnings"
    ]

    results = []
    combined_forecasts = []

    for region in compare_regions:
        if selected_model_type == "XGBoost":
            params = get_xgb_params("rg_xgb")
            cache_key = f"cache_xgb_rg_{region}_{str(params)}"
            if cache_key not in st.session_state:
                with st.spinner(f"Running XGBoost for {region}..."):
                    st.session_state[cache_key] = xgb_model_pipeline(
                        target_col="median_listing_price_x",
                        dataset=data,
                        selected_cols=ml_features,
                        level="region",
                        region=region,
                        test_periods=12,
                        params=params
                    )
            res = st.session_state[cache_key]
        else:
            params = get_rf_params("rg_rf")
            cache_key = f"cache_rf_rg_{region}_{str(params)}"
            if cache_key not in st.session_state:
                with st.spinner(f"Running Random Forest for {region}..."):
                    st.session_state[cache_key] = rf_model_pipeline(
                        target_col="median_listing_price_x",
                        dataset=data,
                        selected_cols=ml_features,
                        level="region",
                        region=region,
                        test_periods=12,
                        params=params
                    )
            res = st.session_state[cache_key]
        
        # Extract metrics
        for m in res["eval_results"]:
            results.append({
                "Region": region,
                "Metric": m["Metric"],
                "Value": m["Value"]
            })
        
        # Collect forecast for plotting
        fc = res["forecast"].to_frame(name="Price")
        fc["Region"] = region
        fc["Type"] = "Forecast"
        combined_forecasts.append(fc)
        
        act = res["test"].to_frame(name="Price")
        act["Region"] = region
        act["Type"] = "Actual"
        combined_forecasts.append(act)

    # Metrics Table
    df_metrics = pd.DataFrame(results)
    pivot_metrics = df_metrics.pivot(index="Region", columns="Metric", values="Value")
    
    st.subheader("📊 Cross-Region Metrics")
    st.dataframe(pivot_metrics.style.background_gradient(cmap="YlOrRd", subset=["RMSE", "MAE", "MAPE"]), width="stretch")

    # Unified Plot
    st.subheader("📈 Overlaid Market Forecasts")
    df_plot = pd.concat(combined_forecasts).reset_index()
    
    fig = px.line(
        df_plot,
        x="date",
        y="Price",
        color="Region",
        line_dash="Type",
        title=f"{selected_model_type} Performance Across Selected Regions",
        template="plotly_dark"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")