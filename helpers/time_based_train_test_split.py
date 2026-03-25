def time_based_train_test_split(df_model, target_col, feature_cols, test_periods=12):
    sorted_dates = sorted(df_model["date"].unique())
    test_dates = sorted_dates[-test_periods:]

    train_df = df_model[df_model["date"] < min(test_dates)].copy()
    test_df = df_model[df_model["date"].isin(test_dates)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return train_df, test_df, X_train, y_train, X_test, y_test