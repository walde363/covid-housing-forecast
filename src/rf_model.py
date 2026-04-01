import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from helpers.prepare_tree_model_data import prepare_tree_model_data
from helpers.time_based_train_test_split import time_based_train_test_split
from helpers.model_evaluator import evaluate_model


def train_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def rf_model_pipeline(target_col, dataset, selected_cols, region):
    
    df_model = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols,
        region=region
    )

    train_df, test_df, X_train, y_train, X_test, y_test = time_based_train_test_split(
        df_model, target_col, selected_cols
    )

    model, pred = train_random_forest(X_train, y_train, X_test)

    evaluation_result = evaluate_model(
        y_true=y_test,
        y_pred=pred,
        model_name="Random Forest",
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
        "eval_results": evaluation_result
    }