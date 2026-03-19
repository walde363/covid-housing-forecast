import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from helpers.plotting import plot_model_results
from helpers.model_evaluator import evaluate_model

def rfr_model(rf_X_train, rf_y_train, rf_X_test):
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )

    rf_model.fit(rf_X_train, rf_y_train)
    rf_pred = rf_model.predict(rf_X_test)

    return rf_model, rf_pred


def xgb_model(xgb_X_train, xgb_y_train, xgb_X_test):
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_model.fit(xgb_X_train, xgb_y_train)
    xgb_pred = xgb_model.predict(xgb_X_test)

    return xgb_model, xgb_pred


def train_test_split(df_model, target_col, feature_cols):
    sorted_dates = sorted(df_model["date"].unique())
    test_dates = sorted_dates[-12:]

    train_df = df_model[df_model["date"] < min(test_dates)].copy()
    test_df = df_model[df_model["date"].isin(test_dates)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return train_df, test_df, X_train, y_train, X_test, y_test

def rf_xgb_models(target_col, dataset, selected_cols):
    # Select columns
    df_model = dataset[selected_cols].copy()

    # Date handling
    df_model["date"] = pd.to_datetime(df_model["date"])

    # Sort before time-based features
    df_model = df_model.sort_values(["RegionName", "date"]).copy()

    # Calendar features
    df_model["year"] = df_model["date"].dt.year
    df_model["month"] = df_model["date"].dt.month
    df_model["quarter"] = df_model["date"].dt.quarter

    # One-hot encode RegionType only if present
    if "RegionType" in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=["RegionType"], drop_first=True)

    # Encode RegionName
    le_region = LabelEncoder()
    df_model["RegionName_enc"] = le_region.fit_transform(df_model["RegionName"])

    # Lag features by region
    for lag in [1, 2, 3, 6, 12]:
        df_model[f"lag_{lag}"] = df_model.groupby("RegionName")[target_col].shift(lag)

    # Rolling features by region
    grouped_target = df_model.groupby("RegionName")[target_col]

    df_model["roll_mean_3"] = (
        grouped_target.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )
    df_model["roll_mean_6"] = (
        grouped_target.shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    )
    df_model["roll_std_3"] = (
        grouped_target.shift(1).rolling(3).std().reset_index(level=0, drop=True)
    )

    # Feature selection
    cols_to_not_include = [target_col, "date", "RegionName"]
    feature_cols = [col for col in df_model.columns if col not in cols_to_not_include]

    # Drop rows with missing values in required columns
    df_model = df_model.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    # Train/test split
    train_df, test_df, X_train, y_train, X_test, y_test = train_test_split(
        df_model, target_col, feature_cols
    )

    # Random Forest
    rf_model_result, rf_pred_results = rfr_model(X_train, y_train, X_test)
    rf_results = evaluate_model(y_test, rf_pred_results, "Random Forest")

    # XGBoost
    xgb_model_result, xgb_pred_results = xgb_model(X_train, y_train, X_test)
    xgb_results = evaluate_model(y_test, xgb_pred_results, "XGBoost")

    # Metrics table
    metrics_table = pd.DataFrame([rf_results, xgb_results]).round(4)

    # Plot
    plot_model_results(
        test_df=test_df,
        actual_values=y_test.values,
        model_results=[rf_pred_results, xgb_pred_results],
        labels=["Train", "Actual", "RF Forecast", "XGB Forecast"],
        title="RF and XGBoost Forecast Comparison",
        target_col=target_col,
        train_df=train_df,
        y_train=y_train.values,
        aggregate=True
    )

    return metrics_table