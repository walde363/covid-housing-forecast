import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from assets.map import state_map
from assets.seasonal_naive_view import render_seasonal_naive
from assets.sarimax_view import sarimax_view, run_tuning as run_sarimax_tuning
from assets.rf_view import rf_view, run_tuning as run_rf_tuning
from assets.xgb_view import xgb_view, run_tuning as run_xgb_tuning
from assets.prophet_view import prophet_view, run_tuning as run_prophet_tuning
from assets.comparison_view import render_comparison
# from assets.choropleth_map import render_choropleth
import time

st.set_page_config(
    page_title="House Market Predictive Interactive Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
    <style>
        .custom-container {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv("data/processed/processed_data_pre_model.csv")
    data["state"] = data["county_name_x"].str.split(", ").str[-1].str.upper()
    data = data.drop(columns=[col for col in data.columns if "_mm" in col or "_yy" in col])
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    return data

def get_state_market_summary(data, selected_state, selected_year, selected_month):
    df = data[
        (data["state"] == selected_state) &
        (data["year"] == selected_year) & 
        (data["month"] == selected_month)
    ].copy()

    if df.empty:
        return None

    latest_date = df["date"].max()
    snapshot_df = df[df["date"] == latest_date].copy()

    summary = {
        "latest_date": latest_date,
        "median_price": snapshot_df["median_listing_price_x"].mean(),
        "inventory": snapshot_df["active_listing_count"].sum(),
        "new_listings": snapshot_df["new_listing_count"].sum(),
        "pending_ratio": snapshot_df["pending_ratio"].mean(),
        "days_on_market": snapshot_df["median_days_on_market_x"].mean(),
    }

    all_state_df = data[data["state"] == selected_state].copy()
    year_ago_date = latest_date - pd.DateOffset(years=1)

    latest_price = snapshot_df["median_listing_price_x"].mean()
    year_ago_df = all_state_df[all_state_df["date"] == year_ago_date].copy()

    if not year_ago_df.empty:
        year_ago_price = year_ago_df["median_listing_price_x"].mean()
        summary["price_yoy"] = (
            ((latest_price - year_ago_price) / year_ago_price) * 100
            if year_ago_price != 0 else None
        )
    else:
        summary["price_yoy"] = None

    latest_inventory = snapshot_df["active_listing_count"].sum()
    year_ago_inventory_df = all_state_df[all_state_df["date"] == year_ago_date].copy()

    if not year_ago_inventory_df.empty:
        year_ago_inventory = year_ago_inventory_df["active_listing_count"].sum()
        summary["inventory_yoy"] = (
            ((latest_inventory - year_ago_inventory) / year_ago_inventory) * 100
            if year_ago_inventory != 0 else None
        )
    else:
        summary["inventory_yoy"] = None

    return summary

def get_market_summary(df):
    if df.empty:
        return None

    latest_date = df["date"].max()
    snapshot_df = df[df["date"] == latest_date].copy()

    summary = {
        "latest_date": latest_date,
        "median_price": snapshot_df["median_listing_price_x"].mean(),
        "inventory": snapshot_df["active_listing_count"].sum(),
        "new_listings": snapshot_df["new_listing_count"].sum(),
        "pending_ratio": snapshot_df["pending_ratio"].mean(),
        "days_on_market": snapshot_df["median_days_on_market_x"].mean(),
    }

    year_ago_date = latest_date - pd.DateOffset(years=1)
    year_ago_df = df[df["date"] == year_ago_date].copy()

    if not year_ago_df.empty:
        year_ago_price = year_ago_df["median_listing_price_x"].mean()
        year_ago_inventory = year_ago_df["active_listing_count"].sum()

        summary["price_yoy"] = (
            ((summary["median_price"] - year_ago_price) / year_ago_price) * 100
            if year_ago_price != 0 else None
        )
        summary["inventory_yoy"] = (
            ((summary["inventory"] - year_ago_inventory) / year_ago_inventory) * 100
            if year_ago_inventory != 0 else None
        )
    else:
        summary["price_yoy"] = None
        summary["inventory_yoy"] = None

    return summary

data = load_data()

st.title("House Market Interactive Dashboard")

if "selected_state" not in st.session_state:
    st.session_state.selected_state = "FL"

if "selected_year" not in st.session_state:
    st.session_state.selected_year = 2026

if "selected_month" not in st.session_state:
    st.session_state.selected_month = 1
    
month_options = sorted(
    data.loc[data["year"] == st.session_state.selected_year, "month"]
    .dropna()
    .unique()
    .tolist()
)

if st.session_state.selected_month not in month_options:
    st.session_state.selected_month = month_options[0]

# create filtered_data BEFORE summaries
selected_state = st.session_state.selected_state
filtered_data = data[data["state"] == selected_state].copy()

summary_state = get_state_market_summary(
    data,
    st.session_state.selected_state,
    st.session_state.selected_year,
    st.session_state.selected_month
)

col1, col2 = st.columns([3, 1])

with col1:
    state_map(data[["state", "year", "median_listing_price_x"]])

with col2:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.selectbox(
            "Select a state",
            options=sorted(data["state"].dropna().unique().tolist()),
            key="selected_state"
        )
    with col_b:
        st.selectbox(
            "Select a year",
            options=sorted(data["year"].dropna().unique().tolist()),
            key="selected_year"
        )
    with col_c:
        st.selectbox(
            "Select a month",
            options=month_options,
            key="selected_month"
        )

    us_summary = get_market_summary(
        data[(data["year"] == st.session_state.selected_year) & 
            (data["month"] == st.session_state.selected_month)].copy()
    )

    if summary_state is not None:
        state_snapshot_container = st.container(border=True)
        state_snapshot_container.write(f"**{st.session_state.selected_state} Snapshot**")
        state_snapshot_container.caption(f"Snapshot month: {summary_state['latest_date'].strftime('%Y-%m')}")

        s1, s2 = state_snapshot_container.columns(2)

        with s1:
            st.metric("Median Listing Price", f"${summary_state['median_price']:,.0f}")
            st.metric("Inventory", f"{summary_state['inventory']:,.0f}")

        with s2:
            st.metric("Pending Ratio", f"{summary_state['pending_ratio']:.2f}")
            st.metric("Days on Market", f"{summary_state['days_on_market']:.1f}")

    if us_summary is not None:
        us_snapshot_container = st.container(border=True)
        us_snapshot_container.write("**U.S. Snapshot**")
        us_snapshot_container.caption(f"Snapshot month: {us_summary['latest_date'].strftime('%Y-%m')}")

        u1, u2 = us_snapshot_container.columns(2)
        with u1:
            st.metric("Median Listing Price", f"${us_summary['median_price']:,.0f}")
            st.metric("Inventory", f"{us_summary['inventory']:,.0f}")
        with u2:
            st.metric("Pending Ratio", f"{us_summary['pending_ratio']:.2f}")
            st.metric("Days on Market", f"{us_summary['days_on_market']:.1f}")

    st.divider()
    st.sidebar.header("⚙️ Global Actions")
    if st.sidebar.button("🚀 Auto-Tune All Models", width='stretch', help="Warning: This will take a long time to complete."):
        target = "median_listing_price_x"
        std_feats = ["median_listing_price_x"]
        ml_feats = ["median_listing_price_x", "active_listing_count", "new_listing_count", "pending_ratio", "price_reduced_share", "Unemployment_Rate", "Earnings"]
        pnl_feats = ["active_listing_count", "new_listing_count", "pending_ratio", "price_reduced_share", "Unemployment_Rate", "Earnings", "Investor Purchases", "Investor Market Share"]
        
        # Tune for the first selected region to keep it efficient
        if st.session_state.regions:
            primary_region = st.session_state.regions[0]
            
            # SARIMAX
            run_sarimax_tuning("rg_sarimax", data, target, std_feats, "region", region=primary_region, rerun=False)
            run_sarimax_tuning("state_sarimax", data, target, std_feats, "state", state=selected_state, rerun=False)
            run_sarimax_tuning("aggr_sarimax", data, target, std_feats, "us", rerun=False)
            
            # Prophet
            run_prophet_tuning("rg_prophet", data, target, std_feats, "region", region=primary_region, rerun=False)
            run_prophet_tuning("state_prophet", data, target, std_feats, "state", state=selected_state, rerun=False)
            run_prophet_tuning("aggr_prophet", data, target, std_feats, "us", rerun=False)
            
            # RF
            run_rf_tuning("rg_rf", data, target, ml_feats, "region", region=primary_region, rerun=False)
            run_rf_tuning("state_rf", data, target, ml_feats, "state", state=selected_state, rerun=False)
            run_rf_tuning("aggr_rf", data, target, ml_feats, "us", rerun=False)
            
            # XGB
            run_xgb_tuning("rg_xgb", data, target, ml_feats, "region", region=primary_region, rerun=False)
            run_xgb_tuning("state_xgb", data, target, ml_feats, "state", state=selected_state, rerun=False)
            run_xgb_tuning("aggr_xgb", data, target, ml_feats, "us", rerun=False)
        
        st.rerun()

# render_choropleth(data, selected_state)

regions = sorted(filtered_data["county_name_x"].dropna().unique().tolist())

if not regions:
    st.warning("No regions found for the selected state.")
    st.stop()

selected_regions = st.multiselect(
    "Select Region(s) to Analyze",
    regions,
    default=[regions[0]] if regions else [],
    key="regions"
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Seasonal Naive", "SARIMAX", "Prophet", "Random Forest", "XGBoost", "Model Comparison"]
)
with tab1:
    res_snaive = render_seasonal_naive(data, selected_regions, selected_state)

with tab2:
    res_sarimax = sarimax_view(data, selected_regions, selected_state)
    
with tab3:
    res_prophet = prophet_view(data, selected_regions, selected_state)

with tab4:
    res_rf = rf_view(data, selected_regions, selected_state)

with tab5:
    res_xgb = xgb_view(data, selected_regions, selected_state)

with tab6:
    render_comparison({
        "Seasonal Naive": res_snaive,
        "SARIMAX": res_sarimax,
        "Prophet": res_prophet,
        "Random Forest": {"region": res_rf["region"], "state": res_rf["state"], "us": res_rf["us"]},
        "Random Forest (Panel)": {"region": res_rf.get("panel", {})},
        "XGBoost": {"region": res_xgb["region"], "state": res_xgb["state"], "us": res_xgb["us"]},
        "XGBoost (Panel)": {"region": res_xgb.get("panel", {})}
    })
