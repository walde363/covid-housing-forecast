import matplotlib.pyplot as plt
import pandas as pd


def plot_model_results(
    test_df,
    actual_values,
    model_results,
    labels,
    title,
    target_col,
    train_df=None,
    y_train=None,
    saveFig=None,
    aggregate=True
):
    """
    Generic plotting function for time series / panel forecasting results.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing at least a 'date' column.
    actual_values : array-like
        Actual y values for the test set.
    model_results : array-like or list of array-like
        One or more prediction arrays.
    labels : list
        Labels in this order:
        - if train_df and y_train are provided:
            ["Train", "Actual", "Model 1", "Model 2", ...]
        - otherwise:
            ["Actual", "Model 1", "Model 2", ...]
    title : str
        Plot title.
    target_col : str
        Y-axis label.
    train_df : pd.DataFrame, optional
        Training dataframe containing at least a 'date' column.
    y_train : array-like, optional
        Actual y values for the training set.
    saveFig : str, optional
        File path to save the figure. If None, show the figure.
    aggregate : bool, default=True
        If True, aggregate panel data by date using mean before plotting.
    """

    # Make predictions always a list
    if not isinstance(model_results, list):
        model_results = [model_results]

    if not isinstance(labels, list):
        labels = [labels]

    # Validate label count
    expected_labels = len(model_results) + 1   # Actual + models
    if train_df is not None and y_train is not None:
        expected_labels += 1  # add Train

    if len(labels) != expected_labels:
        raise ValueError(
            f"Expected {expected_labels} labels, but got {len(labels)}. "
            f"Labels should include Actual and one per model prediction"
            f"{' plus Train' if train_df is not None and y_train is not None else ''}."
        )

    # Build test plotting dataframe
    plot_test_df = test_df[["date"]].copy()
    plot_test_df["Actual"] = actual_values

    for i, preds in enumerate(model_results):
        plot_test_df[labels[i + (2 if train_df is not None and y_train is not None else 1)]] = preds

    # Aggregate if needed
    if aggregate:
        plot_test_df = plot_test_df.groupby("date", as_index=False).mean()

    plt.figure(figsize=(12, 6))

    label_idx = 0

    # Plot training line if provided
    if train_df is not None and y_train is not None:
        plot_train_df = train_df[["date"]].copy()
        plot_train_df["Train"] = y_train

        if aggregate:
            plot_train_df = plot_train_df.groupby("date", as_index=False).mean()

        plt.plot(plot_train_df["date"], plot_train_df["Train"], label=labels[label_idx])
        label_idx += 1

    # Plot actual test values
    plt.plot(plot_test_df["date"], plot_test_df["Actual"], label=labels[label_idx])
    label_idx += 1

    # Plot model predictions
    for col in plot_test_df.columns:
        if col not in ["date", "Actual"]:
            plt.plot(plot_test_df["date"], plot_test_df[col], label=col)

    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if saveFig is None:
        plt.show()
    else:
        plt.savefig(saveFig, bbox_inches="tight")
        plt.close()