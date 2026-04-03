import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from assets.map import state_map
from assets.seasonal_naive_view import render_seasonal_naive
from assets.sarimax_view import sarimax_view
from assets.rf_view import rf_view
from assets.xgb_view import xgb_view
from assets.choropleth_map import render_choropleth

data = pd.read_csv("data/processed/processed_data_pre_model.csv")
data["state"] = data["county_name_x"].str.split(", ").str[-1].str.lower()
data = data.drop(columns=[col for col in data.columns if "_mm" in col or "_yy" in col])

st.set_page_config(
    page_title="House Market Predictive Interactive Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("House Market Interactive Dashboard")

state_map()

selected_state = st.session_state.selected_state
filtered_data = data[data["state"] == selected_state]

st.write("Selected state:", selected_state)

render_choropleth(data, selected_state)

regions = sorted(filtered_data["county_name_x"].dropna().unique().tolist())

if not regions:
    st.warning("No regions found for the selected state.")
    
selected_region = st.selectbox(
            "Select Region",
            regions,
            key="region"
        )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Seasonal Naive", "SARIMAX", "Random Forest", "XGBoost", "TFT"]
)

with tab1:
    render_seasonal_naive(filtered_data, selected_region)

with tab2:
    sarimax_view(filtered_data, selected_region)

with tab3:
    rf_view(data, selected_region, selected_state)

with tab4:
    xgb_view(data, selected_region, selected_state)

with tab5:
    st.write("In progress")