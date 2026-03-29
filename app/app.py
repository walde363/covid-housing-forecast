import streamlit as st
import pandas as pd

from assets.map import state_map
from assets.seasonal_naive_view import render_seasonal_naive

data = pd.read_csv("data/processed/processed_data_pre_model.csv")
data["date"] = pd.to_datetime(data["month_date_yyyymm"])
data["state"] = data["county_name_x"].str.split(", ").str[-1].str.lower()

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
st.write("EDA for selected state data coming soon")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Seasonal Naive", "SARIMAX", "Random Forest", "XGBoost", "TFT"]
)

with tab1:
    render_seasonal_naive(filtered_data)

with tab2:
    st.write("In progress")

with tab3:
    st.write("In progress")

with tab4:
    st.write("In progress")

with tab5:
    st.write("In progress")