import streamlit as st
import plotly.express as px
import pandas as pd
import urllib.request
import json

import os

GEOJSON_PATH = os.path.join(os.path.dirname(__file__), 'geojson-counties-fips.json')

@st.cache_data
def get_geojson():
    if not os.path.exists(GEOJSON_PATH):
        with urllib.request.urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            counties = json.load(response)
        with open(GEOJSON_PATH, 'w') as f:
            json.dump(counties, f)
        return counties
    else:
        with open(GEOJSON_PATH, 'r') as f:
            return json.load(f)

def render_choropleth(data, selected_state):
    st.subheader(f"Median Home Prices in {selected_state.upper()} by County")
    
    # Filter for the selected state
    state_data = data[data["state"] == selected_state].copy()
    
    if state_data.empty:
        st.warning(f"No data available for {selected_state.upper()}.")
        return

    # Ensure FIPS code is a 5-digit string for plotting
    state_data['fips_str'] = state_data['county_fips'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(5)
    
    # Process dates for the slider
    state_data['date_parsed'] = pd.to_datetime(state_data['date'])
    dates = state_data['date_parsed'].dt.date.unique()
    dates.sort()
    
    if len(dates) == 0:
        st.warning("No date information available.")
        return
        
    dates_str = [d.strftime('%Y-%m-%d') for d in dates]
    
    selected_date_str = st.select_slider(
        "Select Date", 
        options=dates_str, 
        value=dates_str[-1],
        key="choropleth_date_slider"
    )
    
    date_data = state_data[state_data['date_parsed'].dt.strftime('%Y-%m-%d') == selected_date_str]
    
    if date_data.empty:
        st.info(f"No price data available for {selected_date_str}.")
        return

    counties = get_geojson()
    
    # Get max price for scale consistency, or let it change per frame.
    # Usually it's better to dynamically change or set a fixed max based on data.
    max_price = state_data['median_listing_price_x'].max()
    
    fig = px.choropleth(
        date_data,
        geojson=counties,
        locations='fips_str',
        color='median_listing_price_x',
        color_continuous_scale="YlOrRd",
        range_color=(0, max_price),
        scope="usa",
        hover_name='county_name_x',
        labels={'median_listing_price_x': 'Median Price ($)'}
    )
    
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
