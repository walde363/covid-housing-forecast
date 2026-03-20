import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

def mase(y_train, y_test, y_pred):
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Naive forecast errors (on training set)
    naive_errors = np.abs(y_train[1:] - y_train[:-1])
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        return np.nan

    # Model errors
    model_mae = np.mean(np.abs(y_test - y_pred))

    return model_mae / naive_mae


def evaluate_model(y_true, y_pred, model_name, metrics=None, train=None):
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
        Supported: "r2", "rmse", "mse", "mae", "mape", "mase"
    train : array-like, optional
        Training target values, required for MASE.

    Returns
    -------
    dict
        Dictionary with model name and requested metric results.
    """

    if metrics is None:
        metrics = ["r2", "rmse"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metric_functions = {
        "r2": lambda yt, yp: r2_score(yt, yp),
        "mse": lambda yt, yp: mean_squared_error(yt, yp),
        "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        "mae": lambda yt, yp: mean_absolute_error(yt, yp),
        "mape": lambda yt, yp: mean_absolute_percentage_error(yt, yp),
        "mase": lambda tr, yt, yp: mase(tr, yt, yp),
    }

    metric_labels = {
        "r2": "R2",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "mape": "MAPE",
        "mase": "MASE",
    }

    results = {"Model": model_name}

    for metric in metrics:
        metric = metric.lower()

        if metric not in metric_functions:
            raise ValueError(
                f"Unsupported metric: '{metric}'. "
                f"Supported metrics are: {list(metric_functions.keys())}"
            )

        if metric == "mase":
            if train is None:
                raise ValueError("train must be provided when using 'mase'.")
            results[metric_labels[metric]] = metric_functions[metric](train, y_true, y_pred)
        else:
            results[metric_labels[metric]] = metric_functions[metric](y_true, y_pred)

    return results