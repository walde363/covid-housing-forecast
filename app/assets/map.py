import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def add_geojson_multipolygon_trace(fig, geojson_data, fillcolor="rgba(180,180,180,0.35)", linecolor="rgba(180,180,180,0.8)"):
    """
    Add a GeoJSON Polygon/MultiPolygon as Scattergeo traces.
    Designed for background silhouettes like Canada.
    """

    for feature in geojson_data["features"]:
        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        if geom_type == "Polygon":
            polygons = [coords]
        elif geom_type == "MultiPolygon":
            polygons = coords
        else:
            continue

        for polygon in polygons:
            # polygon[0] is the exterior ring
            exterior_ring = polygon[0]

            lons = [pt[0] for pt in exterior_ring]
            lats = [pt[1] for pt in exterior_ring]

            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                fill="toself",
                fillcolor=fillcolor,
                line=dict(color=linecolor, width=0.5),
                hoverinfo="skip",
                showlegend=False
            ))


def state_map(df):
    df = df.groupby(["state", "year"], as_index=False).mean(numeric_only=True)
    df = df[df["year"] == st.session_state.selected_year].copy()

    current_dir = Path(__file__).resolve().parent
    us_geojson_path = current_dir / "us_states.geojson"
    ca_geojson_path = current_dir / "ca.json"

    with open(us_geojson_path, "r") as f:
        us_geojson = json.load(f)

    with open(ca_geojson_path, "r") as f:
        ca_geojson = json.load(f)

    fig = go.Figure()

    # Canada silhouette first
    add_geojson_multipolygon_trace(
        fig,
        ca_geojson,
        fillcolor="rgba(190,190,190,0.30)",
        linecolor="rgba(160,160,160,0.55)"
    )

    # US choropleth on top
    us_fig = px.choropleth(
        df,
        geojson=us_geojson,
        locations="state",
        featureidkey="id",
        color="median_listing_price_x",
        hover_name="state",
        hover_data={
            "median_listing_price_x": ":,.0f",
            "year": False
        }
    )

    for trace in us_fig.data:
        fig.add_trace(trace)

    fig.update_geos(
        visible=False,
        center={"lat": 45, "lon": -115},
        projection_scale=3.2,
        fitbounds=False,
        bgcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        title=f"Median Listing Price by State ({st.session_state.selected_year})",
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": False,
            "doubleClick": False,
            "displayModeBar": False
        }
    )
    