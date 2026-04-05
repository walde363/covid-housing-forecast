import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from helpers.prepare_panel_model_data import prepare_panel_model_data
from helpers.time_based_panel_split import time_based_panel_split
from helpers.model_evaluator import evaluate_model
from helpers.add_time_features import add_time_features


def train_random_forest(X_train, y_train, X_test, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def recursive_panel_rf_forecast(
    model,
    df_model,
    target_col,
    selected_region,
    feature_cols,
    region_col="county_name_x",
    steps=18
):
    """
    Recursively forecast future values for one selected region
    using a Random Forest trained on panel data.
    """
    region_key = selected_region.lower().strip()

    history = df_model[df_model[region_col] == region_key].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    if history.empty:
        raise ValueError(f"No history found for selected_region='{selected_region}'")

    base_cols = history.columns.tolist()
    carry_forward_cols = [col for col in base_cols if col not in ["date", target_col]]

    future_predictions = []

    for _ in range(steps):
        last_date = history["date"].max()
        next_date = last_date + pd.offsets.MonthBegin(1)

        new_row = {col: pd.NA for col in base_cols}
        new_row["date"] = next_date
        new_row[region_col] = region_key

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        # carry forward known exogenous/static columns
        for col in carry_forward_cols:
            if col != region_col:
                history.loc[history.index[-1], col] = history.loc[history.index[-2], col]

        history = add_time_features(history, target_col=target_col)

        next_row_features = history.iloc[[-1]][feature_cols].copy()
        pred = model.predict(next_row_features)[0]

        history.loc[history.index[-1], target_col] = pred
        future_predictions.append((next_date, pred))

    return pd.Series(
        [pred for _, pred in future_predictions],
        index=[date for date, _ in future_predictions],
        name="future_forecast"
    )


def rf_panel_pipeline(
    target_col,
    dataset,
    selected_cols,
    selected_region,
    params,
    region_col="county_name_x",
    state_col="state",
    test_periods=12,
):
    """
    Train RF on all US rows, predict only selected region,
    and generate recursive future forecast for that region.
    """

    df_model = prepare_panel_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols,
        region_col=region_col,
        state_col=state_col
    )

    train_df, test_df, X_train, y_train, X_test, y_test = time_based_panel_split(
        df_model=df_model,
        target_col=target_col,
        selected_region=selected_region,
        region_col=region_col,
        state_col=state_col,
        test_periods=test_periods
    )

    model, pred = train_random_forest(X_train, y_train, X_test, params)

    evaluation_result = evaluate_model(
        y_true=y_test,
        y_pred=pred,
        model_name="Random Forest (US train, region predict)",
        metrics=["rmse", "mae", "mase", "mape"],
        train=y_train
    )

    test = test_df.sort_values("date").set_index("date")[target_col]
    forecast = pd.Series(pred, index=test.index, name="forecast")

    region_key = selected_region.lower().strip()
    region_history = df_model[df_model[region_col] == region_key].copy()
    region_history = region_history.sort_values("date")

    train_cutoff = test.index.min()
    train_region = region_history[region_history["date"] < train_cutoff].copy()
    train = train_region.set_index("date")[target_col]
    
    future_source = df_model[df_model["date"] <= test.index.max()].copy()
    
    future_forecast = recursive_panel_rf_forecast(
        model=model,
        df_model=future_source,
        target_col=target_col,
        selected_region=selected_region,
        feature_cols=X_train.columns.tolist(),
        region_col=region_col,
        steps=18
    )

    return {
        "model": model,
        "forecast": forecast,
        "future_forecast": future_forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "df_model": df_model
    }
    