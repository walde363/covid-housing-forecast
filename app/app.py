import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from components.slide_viewer import render_slide_viewer
from src.sarimax_model import sarimax_model
import pandas as pd

data = pd.read_csv("data/processed/processed_data_low.csv")
regions = data["RegionName"].unique()
print(regions)

st.set_page_config(
    page_title="COVID-era changes in Florida home prices compared to National trends",
    page_icon="🏠",
    layout="wide"
)

st.title("COVID-era changes in Florida home prices compared to National trends")
st.markdown(
    """
    Forecasting home price trends before, during, and after COVID to quantify structural breaks and shifts in market dynamics.
    """
)

tab1, tab2, tab3 = st.tabs(["Random Forest", "XGBoost", "SARIMAX"])

with tab1:
    st.header("Random Forest Model")
    st.write("Comming Soon")
with tab2:
    st.header("XGBoost Model")
    st.write("Commign Soon")
with tab3:
    st.header("SARIMAX Model")
    selected_region = st.selectbox("Select Region", regions)

    result = sarimax_model(data, selected_region, "MarketTemp")

    col1, col2 = st.columns([3, 1])

    with col1:
        plot_df = pd.DataFrame({
            "predicted": pd.Series(result["forecast"]),
            "train": pd.Series(result["train"]),
            "test": pd.Series(result["test"]),
        })

        st.line_chart(plot_df)

    with col2:
        st.subheader("Model Metrics")
        for key, value in list(result["eval_results"].items())[1:]:
            st.write(f"{key}: {value:4f}")

with st.expander("About"):
    render_slide_viewer(
    slides_dir="app/assets/pitch_presentation_resources",
    title="Presentation"
)
