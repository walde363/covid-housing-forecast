import pandas as pd
from xgboost import XGBRegressor

from helpers.prepare_tree_model_data import prepare_tree_model_data
from helpers.time_based_train_test_split import time_based_train_test_split
from helpers.model_evaluator import evaluate_model
from helpers.add_time_features import add_time_features


def train_xgboost(X_train, y_train, X_test, params):
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def fit_xgboost_full(df_model, target_col, params):
    X_full = df_model.drop(columns=[target_col, "date"], errors="ignore").copy()
    y_full = df_model[target_col].copy()

    model = XGBRegressor(**params)
    model.fit(X_full, y_full)

    return model, X_full, y_full


def recursive_xgb_forecast(
    model,
    df_model,
    target_col,
    feature_cols,
    feature_dtypes,
    steps=12
):
    """
    Recursively forecast future values using an XGBoost model
    trained on tree-based time series features.
    """
    history = df_model.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    base_cols = history.columns.tolist()
    carry_forward_cols = [col for col in base_cols if col not in ["date", target_col]]

    future_predictions = []

    for _ in range(steps):
        last_date = history["date"].max()
        next_date = last_date + pd.offsets.MonthBegin(1)

        new_row = {col: None for col in base_cols}
        new_row["date"] = next_date

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        # carry forward known exogenous/static columns
        for col in carry_forward_cols:
            history.loc[history.index[-1], col] = history.loc[history.index[-2], col]

        history = add_time_features(history, target_col=target_col)

        next_row_features = history.iloc[[-1]][feature_cols].copy()

        # force same dtypes as training data
        for col in feature_cols:
            if str(feature_dtypes[col]) in ["int64", "int32", "float64", "float32", "bool"]:
                next_row_features[col] = pd.to_numeric(next_row_features[col], errors="coerce")

        next_row_features = next_row_features.astype(feature_dtypes)

        pred = model.predict(next_row_features)[0]

        history.loc[history.index[-1], target_col] = pred
        future_predictions.append((next_date, pred))

    return pd.Series(
        [pred for _, pred in future_predictions],
        index=[date for date, _ in future_predictions],
        name="future_forecast"
    )


def xgb_model_pipeline(
    target_col,
    dataset,
    selected_cols,
    params,
    level="region",
    region=None,
    state=None,
    test_periods=12,
    forecast_periods=12
):
    """
    XGBoost pipeline supporting region/state/us modeling.
    """

    df_model = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols,
        level=level,
        region=region,
        state=state,
    )

    train_df, test_df, X_train, y_train, X_test, y_test = time_based_train_test_split(
        df_model=df_model,
        target_col=target_col,
        test_periods=test_periods,
        add_features=True
    )

    model, pred = train_xgboost(X_train, y_train, X_test, params)

    evaluation_result = evaluate_model(
        y_true=y_test,
        y_pred=pred,
        model_name=f"XGBoost ({level})",
        metrics=["rmse", "mae", "mase", "mape"],
        train=y_train
    )

    train = train_df.sort_values("date").set_index("date")[target_col]
    test = test_df.sort_values("date").set_index("date")[target_col]

    forecast = pd.Series(pred, index=test.index, name="forecast")

    future_source = df_model[df_model["date"] <= test.index.max()].copy()

    final_model, X_full, y_full = fit_xgboost_full(
        df_model=future_source,
        target_col=target_col,
        params=params
    )

    future_forecast = recursive_xgb_forecast(
        model=final_model,
        df_model=future_source,
        target_col=target_col,
        feature_cols=X_full.columns.tolist(),
        feature_dtypes=X_full.dtypes.to_dict(),
        steps=forecast_periods
    )

    return {
        "model": model,
        "final_model": final_model,
        "forecast": forecast,
        "future_forecast": future_forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_full": X_full,
        "y_full": y_full,
        "df_model": df_model
    }
    