import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_tree_model_data(dataset, target_col, selected_cols):
    df_model = dataset[selected_cols].copy()

    df_model["date"] = pd.to_datetime(df_model["date"])
    df_model = df_model.sort_values(["RegionName", "date"]).copy()

    df_model["year"] = df_model["date"].dt.year
    df_model["month"] = df_model["date"].dt.month
    df_model["quarter"] = df_model["date"].dt.quarter

    if "RegionType" in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=["RegionType"], drop_first=True)

    le_region = LabelEncoder()
    df_model["RegionName_enc"] = le_region.fit_transform(df_model["RegionName"])

    for lag in [1, 2, 3, 6, 12]:
        df_model[f"lag_{lag}"] = df_model.groupby("RegionName")[target_col].shift(lag)

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

    cols_to_not_include = [target_col, "date", "RegionName"]
    feature_cols = [col for col in df_model.columns if col not in cols_to_not_include]

    df_model = df_model.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    return df_model, feature_cols