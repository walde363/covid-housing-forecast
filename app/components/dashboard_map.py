import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import streamlit as st

def dashboard_map(data):
    cols = ["county_name", "median_listing_price", "month_date_yyyymm"]

    map_data = data[cols].copy()
    map_data["date"] = pd.to_datetime(map_data["month_date_yyyymm"], format="%Y%m").dt.to_period("M")

    def extract_state(col):
        return col.split(", ")[-1].upper().strip()

    map_data["state"] = map_data["county_name"].apply(extract_state)

    # Filter to June 2023 and average by state
    for_the_map = (
        map_data[
            (map_data["date"].dt.year == 2023) &
            (map_data["date"].dt.month == 6)
        ]
        .groupby("state", as_index=False)["median_listing_price"]
        .mean()
    )

    states = gpd.read_file("data/processed/us_states.geojson")

    if states.crs is not None and states.crs.to_string() != "EPSG:4326":
        states = states.to_crs(epsg=4326)

    # GeoJSON state abbreviation column
    states = states.rename(columns={"id": "state"})
    states["state"] = states["state"].str.upper().str.strip()
    for_the_map["state"] = for_the_map["state"].str.upper().str.strip()

    gdf = states.merge(for_the_map, on="state", how="left")
    gdf["median_listing_price"] = gdf["median_listing_price"].fillna(0)

    # Optional: remove Alaska and Hawaii for a tighter US-only view
    gdf_plot = gdf[~gdf["state"].isin(["AK", "HI"])].copy()

    m = gdf_plot.explore(
        column="median_listing_price",
        cmap="Blues",
        legend=True,
        tooltip=["state", "median_listing_price"],
        popup=True,
        style_kwds={"weight": 1, "fillOpacity": 0.7},
        location=[39.5, -98.35],   # center of continental US
        zoom_start=4
    )

    # Force map to fit only the US geometries being plotted
    bounds = gdf_plot.total_bounds  # [minx, miny, maxx, maxy]
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    st_folium(m, width=1200, height=700)