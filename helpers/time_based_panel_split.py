import pandas as pd

from helpers.add_panel_time_features import add_panel_time_features


def time_based_panel_split(
    df_model,
    target_col,
    selected_region,
    region_col="county_name_x",
    state_col="state",
    test_periods=12
):
    """
    Time-based split for panel data.
    Trains on all rows, tests only on selected region.

    Parameters
    ----------
    df_model : pd.DataFrame
        Panel dataframe.
    target_col : str
        Target variable.
    selected_region : str
        Region to evaluate/predict.
    region_col : str
        Region column.
    state_col : str
        State column.
    test_periods : int
        Number of time periods in test set.

    Returns
    -------
    train_df, test_df, X_train, y_train, X_test, y_test
    """

    df_model = df_model.copy()
    df_model = add_panel_time_features(
        df_model,
        target_col=target_col,
        region_col=region_col
    )

    df_model = df_model.dropna().reset_index(drop=True)
    df_model = df_model.sort_values([region_col, "date"])

    # encode categories after feature creation
    df_model["region_code"] = df_model[region_col].astype("category").cat.codes
    df_model["state_code"] = df_model[state_col].astype("category").cat.codes

    sorted_dates = sorted(df_model["date"].unique())
    test_dates = sorted_dates[-test_periods:]
    cutoff_date = min(test_dates)

    # train on all US rows before cutoff
    train_df = df_model[df_model["date"] < cutoff_date].copy()

    # test only on selected region in test dates
    selected_region = selected_region.lower().strip()
    test_df = df_model[
        (df_model["date"].isin(test_dates)) &
        (df_model[region_col] == selected_region)
    ].copy()

    if test_df.empty:
        raise ValueError(
            f"No test rows found for selected_region='{selected_region}'."
        )

    drop_cols = ["date", target_col, region_col, state_col]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col]

    return train_df, test_df, X_train, y_train, X_test, y_test
