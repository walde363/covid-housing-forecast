import pandas as pd

from helpers.prepare_panel_model_data import prepare_panel_model_data
from helpers.time_based_panel_split import time_based_panel_split
from helpers.model_evaluator import evaluate_model
from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, X_test, params):
    model = XGBRegressor(
        **params
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, predictions


def xgb_panel_pipeline(
    target_col,
    dataset,
    selected_cols,
    selected_region,
    params,
    region_col="county_name_x",
    state_col="state",
    test_periods=12
):
    """
    Train XGBoost on all US rows, predict only selected region.
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

    model, pred = train_xgboost(X_train, y_train, X_test, params)

    evaluation_result = evaluate_model(
        y_true=y_test,
        y_pred=pred,
        model_name="XGBoost (US train, region predict)",
        metrics=["rmse", "mae", "mase", "mape"],
        train=y_train
    )

    test = test_df.sort_values("date").set_index("date")[target_col]
    forecast = pd.Series(pred, index=test.index, name="forecast")

    region_history = df_model[df_model[region_col] == selected_region.lower().strip()].copy()
    region_history = region_history.sort_values("date")

    train_cutoff = test.index.min()
    train_region = region_history[region_history["date"] < train_cutoff].copy()
    train = train_region.set_index("date")[target_col]

    return {
        "model": model,
        "forecast": forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df
    }
    