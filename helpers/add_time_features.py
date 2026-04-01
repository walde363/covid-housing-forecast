import pandas as pd


def add_time_features(df, target_col, lags=(1, 2, 3, 6, 12), rolling_windows=(3, 6)):
    """
    Add lag and rolling features for time series tree models.
    Assumes one row per date.
    """
    df = df.copy()
    df = df.sort_values("date")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    for window in rolling_windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].shift(1).rolling(window).std()

    return df
