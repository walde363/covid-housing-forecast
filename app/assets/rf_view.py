import streamlit as st
import plotly.graph_objects as go
from src.rf_model import rf_model_pipeline

def rf_view(filtered_data, selected_region):
    with st.container():
        st.header("Random Forest Regressor Model")