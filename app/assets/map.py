import streamlit as st
import pandas as pd

def state_map():
    state_lat_lon = {
        "al": {"lat": 32.8067, "lon": -86.7911},
        "ak": {"lat": 61.3707, "lon": -152.4044},
        "az": {"lat": 33.7298, "lon": -111.4312},
        "ar": {"lat": 34.9697, "lon": -92.3731},
        "ca": {"lat": 36.1162, "lon": -119.6816},
        "co": {"lat": 39.0598, "lon": -105.3111},
        "ct": {"lat": 41.5978, "lon": -72.7554},
        "de": {"lat": 39.3185, "lon": -75.5071},
        "fl": {"lat": 27.7663, "lon": -81.6868},
        "ga": {"lat": 33.0406, "lon": -83.6431},
        "hi": {"lat": 21.0943, "lon": -157.4983},
        "id": {"lat": 44.2405, "lon": -114.4788},
        "il": {"lat": 40.3495, "lon": -88.9861},
        "in": {"lat": 39.8494, "lon": -86.2583},
        "ia": {"lat": 42.0115, "lon": -93.2105},
        "ks": {"lat": 38.5266, "lon": -96.7265},
        "ky": {"lat": 37.6681, "lon": -84.6701},
        "la": {"lat": 31.1695, "lon": -91.8678},
        "me": {"lat": 44.6939, "lon": -69.3819},
        "md": {"lat": 39.0639, "lon": -76.8021},
        "ma": {"lat": 42.2302, "lon": -71.5301},
        "mi": {"lat": 43.3266, "lon": -84.5361},
        "mn": {"lat": 45.6945, "lon": -93.9002},
        "ms": {"lat": 32.7416, "lon": -89.6787},
        "mo": {"lat": 38.4561, "lon": -92.2884},
        "mt": {"lat": 46.9219, "lon": -110.4544},
        "ne": {"lat": 41.1254, "lon": -98.2681},
        "nv": {"lat": 38.3135, "lon": -117.0554},
        "nh": {"lat": 43.4525, "lon": -71.5639},
        "nj": {"lat": 40.2989, "lon": -74.5210},
        "nm": {"lat": 34.8405, "lon": -106.2485},
        "ny": {"lat": 42.1657, "lon": -74.9481},
        "nc": {"lat": 35.6301, "lon": -79.8064},
        "nd": {"lat": 47.5289, "lon": -99.7840},
        "oh": {"lat": 40.3888, "lon": -82.7649},
        "ok": {"lat": 35.5653, "lon": -96.9289},
        "or": {"lat": 44.5720, "lon": -122.0709},
        "pa": {"lat": 40.5908, "lon": -77.2098},
        "ri": {"lat": 41.6809, "lon": -71.5118},
        "sc": {"lat": 33.8569, "lon": -80.9450},
        "sd": {"lat": 44.2998, "lon": -99.4388},
        "tn": {"lat": 35.7478, "lon": -86.6923},
        "tx": {"lat": 31.0545, "lon": -97.5635},
        "ut": {"lat": 40.1500, "lon": -111.8624},
        "vt": {"lat": 44.0459, "lon": -72.7107},
        "va": {"lat": 37.7693, "lon": -78.1700},
        "wa": {"lat": 47.4009, "lon": -121.4905},
        "wv": {"lat": 38.4912, "lon": -80.9545},
        "wi": {"lat": 44.2685, "lon": -89.6165},
        "wy": {"lat": 42.7560, "lon": -107.3025}
    }

    if "map_selected_state" not in st.session_state:
        st.session_state.selected_state = "fl"

    st.selectbox(
        "Select a state",
        options=list(state_lat_lon.keys()),
        key="map_selected_state"
    )

    selected_state = st.session_state.selected_state
    lat = state_lat_lon[selected_state]["lat"]
    lon = state_lat_lon[selected_state]["lon"]

    map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
    st.map(map_df)