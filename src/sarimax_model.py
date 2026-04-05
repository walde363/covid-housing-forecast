import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from helpers.prepare_tree_model_data import prepare_tree_model_data
from helpers.model_evaluator import evaluate_model

def train_sarimax(train_series, params):
    model = SARIMAX(
        train_series,
        order=params.get("order", (1, 1, 1)),
        seasonal_order=params.get("seasonal_order", (1, 1, 1, 12)),
        enforce_stationarity=params.get("enforce_stationarity", False),
        enforce_invertibility=params.get("enforce_invertibility", False)
    )
    model_fit = model.fit(disp=False)
    return model_fit


def sarimax_model_pipeline(
    target_col,
    dataset,
    selected_cols,
    params,
    level="region",
    region=None,
    state=None,
    test_periods=12
):
    """
    SARIMAX pipeline supporting region/state/us modeling.

    Parameters
    ----------
    target_col : str
        Target column to forecast.
    dataset : pandas.DataFrame
        Full dataset.
    selected_cols : list
        Columns to keep/use when preparing the modeling dataframe.
    params : dict
        SARIMAX parameters:
            - order
            - seasonal_order
            - enforce_stationarity
            - enforce_invertibility
    level : str, optional
        Aggregation/modeling level: "region", "state", or "us".
    region : str, optional
        Region name if level="region".
    state : str, optional
        State name if level="state".
    test_periods : int, optional
        Number of observations for the test set.

    Returns
    -------
    dict
        Dictionary containing fitted model, forecast, future forecast,
        train/test splits, evaluation results, and prepared dataframe.
    """

    df_model = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols,
        level=level,
        region=region,
        state=state
    )

    ts = df_model[["date", target_col]].copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.dropna(subset=[target_col]).sort_values("date")
    ts = ts.set_index("date")
    ts = ts.asfreq("MS")
    ts = ts.dropna(subset=[target_col])

    if len(ts) <= test_periods:
        raise ValueError(
            f"Not enough data points ({len(ts)}) for test_periods={test_periods}"
        )

    selected_target = ts[target_col]

    train = selected_target.iloc[:-test_periods]
    test = selected_target.iloc[-test_periods:]

    # Fit on training data only
    model_fit = train_sarimax(train, params)

    pred = model_fit.forecast(steps=len(test))
    forecast = pd.Series(pred, index=test.index, name="forecast")

    evaluation_result = evaluate_model(
        y_true=test.values,
        y_pred=forecast.values,
        model_name=f"SARIMAX ({level})",
        metrics=["rmse", "mae", "mase", "mape"],
        train=train.values
    )

    # Refit on full series for future forecasting
    full_model_fit = train_sarimax(selected_target, params)

    last_test_date = test.index.max()
    future_dates = pd.date_range(
        start=last_test_date,
        periods=18 + 1,
        freq="MS"
    )[1:]

    future_pred = full_model_fit.forecast(steps=18)
    future_forecast = pd.Series(
        future_pred,
        index=future_dates,
        name="future_forecast"
    )

    return {
        "model": model_fit,
        "forecast": forecast,
        "future_forecast": future_forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result,
        "df_model": df_model
    }
    