import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from helpers.prepare_tree_model_data import prepare_tree_model_data
from helpers.time_based_train_test_split import time_based_train_test_split
from helpers.model_evaluator import evaluate_model

def train_random_forest(X_train, y_train, X_test, params):
    model = RandomForestRegressor(
        **params
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

def rf_model_pipeline(
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
    Random Forest pipeline supporting region/state/us modeling.
    """

    df_model = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols,
        level=level,
        region=region,
        state=state
    )

    train_df, test_df, X_train, y_train, X_test, y_test = time_based_train_test_split(
        df_model=df_model,
        target_col=target_col,
        test_periods=test_periods,
        add_features=True
    )

    model, pred = train_random_forest(X_train, y_train, X_test, params)

    evaluation_result = evaluate_model(
        y_true=y_test,
        y_pred=pred,
        model_name=f"Random Forest ({level})",
        metrics=["rmse", "mae", "mase", "mape"],
        train=y_train
    )

    train = train_df.sort_values("date").set_index("date")[target_col]
    test = test_df.sort_values("date").set_index("date")[target_col]

    forecast = pd.Series(
        pred,
        index=test.index,
        name="forecast"
    )

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
        "df_model": df_model
    }
