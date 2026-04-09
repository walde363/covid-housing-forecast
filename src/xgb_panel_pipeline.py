import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from helpers.prepare_panel_model_data import prepare_panel_model_data
from helpers.time_based_panel_split import time_based_panel_split
from helpers.model_evaluator import evaluate_model
from helpers.add_time_features import add_time_features

_GPU_AVAILABLE = None

def _check_gpu_available():
    """Checks if a GPU is available for XGBoost and caches the result."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            # Attempt a tiny fit on GPU to verify CUDA support
            XGBRegressor(device="cuda").fit(np.array([[0]]), np.array([0]))
            _GPU_AVAILABLE = True
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE

def get_forecast_feature_cols(target_col):
    return [
        "year", "month", "quarter",
        f"{target_col}_lag_1",
        f"{target_col}_lag_2",
        f"{target_col}_lag_3",
        f"{target_col}_lag_6",
        f"{target_col}_lag_12",
        f"{target_col}_roll_mean_3",
        f"{target_col}_roll_std_3",
        f"{target_col}_roll_mean_6",
        f"{target_col}_roll_std_6",
    ]

def train_xgboost(X_train, y_train, X_test, params):
    xgb_params = params.copy()
    xgb_params["device"] = "cuda" if _check_gpu_available() else "cpu"
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def recursive_panel_xgb_forecast(
    model,
    df_model,
    target_col,
    selected_region,
    feature_cols,
    feature_dtypes,
    region_col="county_name_x",
    steps=18
):
    """
    Recursively forecast future values for one selected region
    using an XGBoost model trained on forecast-safe features.
    """
    region_key = selected_region.lower().strip()

    history = df_model[df_model[region_col] == region_key][["date", target_col]].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    if history.empty:
        raise ValueError(f"No history found for selected_region='{selected_region}'")

    future_predictions = []

    for _ in range(steps):
        last_date = history["date"].max()
        next_date = last_date + pd.offsets.MonthBegin(1)

        new_row = pd.DataFrame({
            "date": [next_date],
            target_col: [np.nan]
        })

        history = pd.concat([history, new_row], ignore_index=True)
        history = add_time_features(history, target_col=target_col)

        next_row_features = history.iloc[[-1]][feature_cols].copy()
        next_row_features = next_row_features.astype(feature_dtypes)

        pred = model.predict(next_row_features)[0]

        history.loc[history.index[-1], target_col] = pred
        future_predictions.append((next_date, pred))

    return pd.Series(
        [pred for _, pred in future_predictions],
        index=[date for date, _ in future_predictions],
        name="future_forecast"
    )


def xgb_panel_pipeline(
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
    Train XGBoost on all US rows, predict only selected region,
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

    region_key = selected_region.lower().strip()
    region_history = df_model[df_model[region_col] == region_key].copy()
    region_history = region_history.sort_values("date")

    train_cutoff = test.index.min()
    train_region = region_history[region_history["date"] < train_cutoff].copy()
    train = train_region.set_index("date")[target_col]

    # Forecast-only model
    future_source = df_model[df_model["date"] <= test_df["date"].max()].copy()

    future_source_region = future_source[future_source[region_col] == region_key].copy()
    future_source_region["date"] = pd.to_datetime(future_source_region["date"])
    future_source_region = future_source_region.sort_values("date")

    future_source_region = add_time_features(future_source_region, target_col=target_col)

    forecast_feature_cols = get_forecast_feature_cols(target_col)

    future_source_region = future_source_region.dropna(
        subset=forecast_feature_cols + [target_col]
    ).copy()

    X_full = future_source_region[forecast_feature_cols].copy()
    y_full = future_source_region[target_col].copy()

    forecast_params = params.copy()
    forecast_params["device"] = "cuda" if _check_gpu_available() else "cpu"

    forecast_model = XGBRegressor(**forecast_params)
    forecast_model.fit(X_full, y_full)

    future_forecast = recursive_panel_xgb_forecast(
        model=forecast_model,
        df_model=future_source,
        target_col=target_col,
        selected_region=selected_region,
        feature_cols=forecast_feature_cols,
        feature_dtypes=X_full.dtypes.to_dict(),
        region_col=region_col,
        steps=18
    )

    return {
        "model": model,
        "forecast_model": forecast_model,
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
    