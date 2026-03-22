import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from helpers.feature_engineering import prepare_tree_model_data
from helpers.data_split import time_based_train_test_split
from helpers.model_evaluator import evaluate_model
from helpers.plotting import plot_model_results


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


def rf_model_pipeline(target_col, dataset, selected_cols):
    df_model, feature_cols = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols
    )

    train_df, test_df, X_train, y_train, X_test, y_test = time_based_train_test_split(
        df_model, target_col, feature_cols
    )

    model, predictions = train_random_forest(X_train, y_train, X_test)

    metrics_table = pd.DataFrame([
        evaluate_model(y_test, predictions, "Random Forest")
    ]).round(4)

    plot_model_results(
        test_df=test_df,
        actual_values=y_test.values,
        model_results=[predictions],
        labels=["Train", "Actual", "RF Forecast"],
        title="Random Forest Forecast Comparison",
        target_col=target_col,
        train_df=train_df,
        y_train=y_train.values,
        aggregate=True
    )

    return metrics_table, model