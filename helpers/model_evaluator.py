import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


def evaluate_model(y_true, y_pred, model_name, metrics=None):
    """
    Evaluate a model using a configurable list of regression metrics.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Name of the model.
    metrics : list, optional
        List of metric names to calculate.
        Supported: "r2", "rmse", "mse", "mae", "mape"

    Returns
    -------
    dict
        Dictionary with model name and requested metric results.
    """

    if metrics is None:
        metrics = ["r2", "rmse"]

    metric_functions = {
        "r2": lambda yt, yp: r2_score(yt, yp),
        "mse": lambda yt, yp: mean_squared_error(yt, yp),
        "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        "mae": lambda yt, yp: mean_absolute_error(yt, yp),
        "mape": lambda yt, yp: mean_absolute_percentage_error(yt, yp)
    }

    metric_labels = {
        "r2": "R2",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "mape": "MAPE"
    }

    results = {"Model": model_name}

    for metric in metrics:
        metric = metric.lower()
        if metric not in metric_functions:
            raise ValueError(
                f"Unsupported metric: '{metric}'. "
                f"Supported metrics are: {list(metric_functions.keys())}"
            )
        results[metric_labels[metric]] = metric_functions[metric](y_true, y_pred)

    return results