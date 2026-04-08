import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from helpers.prepare_tree_model_data import prepare_tree_model_data
from helpers.time_based_train_test_split import time_based_train_test_split
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

def train_xgboost(X_train, y_train, X_test, params):
    xgb_params = params.copy()
    xgb_params["device"] = "cuda" if _check_gpu_available() else "cpu"
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def fit_xgboost_full(df_model, target_col, params):
    X_full = df_model.drop(columns=[target_col, "date"], errors="ignore").copy()
    y_full = df_model[target_col].copy()

    xgb_params = params.copy()
    xgb_params["device"] = "cuda" if _check_gpu_available() else "cpu"
    model = XGBRegressor(**xgb_params)
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

    future_predictions = []

    for _ in range(steps):
        last_date = history["date"].max()
        next_date = last_date + pd.offsets.MonthBegin(1)

        # Copy the last row to maintain dtypes and exogenous values, then update date
        new_row = history.iloc[[-1]].copy()
        new_row["date"] = next_date
        new_row[target_col] = np.nan

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
    