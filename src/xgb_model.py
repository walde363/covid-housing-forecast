import pandas as pd
from xgboost import XGBRegressor

from helpers.feature_engineering import prepare_tree_model_data
from helpers.data_split import time_based_train_test_split
from helpers.model_evaluator import evaluate_model
from helpers.plotting import plot_model_results


def train_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def xgb_model_pipeline(target_col, dataset, selected_cols):
    df_model, feature_cols = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols
    )

    train_df, test_df, X_train, y_train, X_test, y_test = time_based_train_test_split(
        df_model, target_col, feature_cols
    )

    model, predictions = train_xgboost(X_train, y_train, X_test)

    metrics_table = pd.DataFrame([
        evaluate_model(y_test, predictions, "XGBoost")
    ]).round(4)

    plot_model_results(
        test_df=test_df,
        actual_values=y_test.values,
        model_results=[predictions],
        labels=["Train", "Actual", "XGB Forecast"],
        title="XGBoost Forecast Comparison",
        target_col=target_col,
        train_df=train_df,
        y_train=y_train.values,
        aggregate=True
    )

    return metrics_table, model