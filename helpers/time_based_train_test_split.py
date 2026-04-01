import pandas as pd

from helpers.add_time_features import add_time_features


def time_based_train_test_split(
    df_model,
    target_col,
    test_periods=12,
    add_features=True
):
    """
    Time-based split for single-series datasets (one row per date).
    """

    df_model = df_model.copy()
    df_model["date"] = pd.to_datetime(df_model["date"])
    df_model = df_model.sort_values("date")

    if add_features:
        df_model = add_time_features(df_model, target_col)

    df_model = df_model.dropna().reset_index(drop=True)

    sorted_dates = sorted(df_model["date"].unique())
    test_dates = sorted_dates[-test_periods:]

    train_df = df_model[df_model["date"] < min(test_dates)].copy()
    test_df = df_model[df_model["date"].isin(test_dates)].copy()

    X_train = train_df.drop(columns=["date", target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=["date", target_col])
    y_test = test_df[target_col]

    return train_df, test_df, X_train, y_train, X_test, y_test