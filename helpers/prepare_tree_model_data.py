import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_tree_model_data(dataset, target_col, selected_cols, region):
    df_model = dataset[selected_cols].copy()

    df_model["date"] = pd.to_datetime(df_model["date"])
    df_model = df_model.sort_values([region, "date"]).copy()

    df_model["year"] = df_model["date"].dt.year
    df_model["month"] = df_model["date"].dt.month
    df_model["quarter"] = df_model["date"].dt.quarter

    le_region = LabelEncoder()
    df_model[f"{region}_enc"] = le_region.fit_transform(df_model[region])
    
    le_city = LabelEncoder()
    df_model["city_enc"] = le_city.fit_transform(df_model["city"])

    for lag in [1, 2, 3, 6, 12]:
        df_model[f"lag_{lag}"] = df_model.groupby(region)[target_col].shift(lag)

    grouped_target = df_model.groupby(region)[target_col]

    df_model["roll_mean_3"] = (
        grouped_target.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    )
    df_model["roll_mean_6"] = (
        grouped_target.shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    )
    df_model["roll_std_3"] = (
        grouped_target.shift(1).rolling(3).std().reset_index(level=0, drop=True)
    )
    
    df_model = df_model.drop(columns=[region, "city", "state"])
    
    non_numeric_columns = df_model.select_dtypes(exclude=['number']).columns
    print("Non-numeric columns:", non_numeric_columns)

    return df_model