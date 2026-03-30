from statsmodels.tsa.statespace.sarimax import SARIMAX
from helpers.model_evaluator import evaluate_model
import pandas as pd

def sarimax_model(
    data,
    filter_col, 
    filter_by,
    target,
    periods=16,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
):
    """
    Fit a SARIMA or SARIMAX model for a single region and evaluate forecast performance.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing region, date, target, and optional feature columns.
    regionName : str
        Region name to model.
    target : str
        Target column to forecast.
    feature_cols : list, optional
        Exogenous feature columns. If not provided, the function fits SARIMA.
        If provided, the function fits SARIMAX.
    periods : int, optional
        Number of observations used for the test set.
    order : tuple, optional
        Non-seasonal order (p, d, q).
    seasonal_order : tuple, optional
        Seasonal order (P, D, Q, s).
    enforce_stationarity : bool, optional
        Whether to enforce stationarity.
    enforce_invertibility : bool, optional
        Whether to enforce invertibility.

    Returns
    -------
    dict
        Model object, forecast, train/test splits, optional exogenous splits,
        and evaluation results.
    """
    region = data[data[filter_col] == filter_by].copy()

    region["date"] = pd.to_datetime(region["date"])
    region = region.sort_values("date")
    region = region[["date", target]].dropna()
    region = region.set_index("date")
    region = region.asfreq("MS")
    region = region.dropna(subset=[target])

    if len(region) <= periods:
        raise ValueError(
            f"Not enough observations for region '{filter_by}'. Need more than {periods} rows."
        )

    selected_target = region[target]

    train = selected_target.iloc[:-periods]
    test = selected_target.iloc[-periods:]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    )

    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))

    evaluation_result = evaluate_model(
        y_true=test,
        y_pred=forecast,
        model_name="SARIMAX",
        metrics=["rmse", "mae", "mase", "mape"],
        train=train
    )

    return {
        "model": model_fit,
        "forecast": forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result
    }