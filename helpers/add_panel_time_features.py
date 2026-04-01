import pandas as pd


def add_panel_time_features(
    df,
    target_col,
    region_col="county_name_x",
    lags=(1, 2, 3, 6, 12),
    rolling_windows=(3, 6)
):
    """
    Create grouped time features for panel data.
    Lags and rolling stats are computed within each region.

    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe.
    target_col : str
        Target column.
    region_col : str
        Region identifier.
    lags : tuple
        Lag periods.
    rolling_windows : tuple
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Dataframe with time features.
    """

    df = df.copy()
    df = df.sort_values([region_col, "date"])

    # calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # grouped lags
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = (
            df.groupby(region_col)[target_col].shift(lag)
        )

    # grouped rolling features
    for window in rolling_windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df.groupby(region_col)[target_col]
            .shift(1)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{target_col}_roll_std_{window}"] = (
            df.groupby(region_col)[target_col]
            .shift(1)
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df
