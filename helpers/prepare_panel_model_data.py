import pandas as pd


def prepare_panel_model_data(
    dataset,
    target_col,
    selected_cols,
    region_col="county_name_x",
    state_col="state"
):
    """
    Prepare panel data for tree models.
    Keeps all US rows, one row per region-date.
    """

    required_cols = ["date", region_col, state_col, target_col]
    cols_to_use = list(dict.fromkeys(required_cols + selected_cols))

    missing_cols = [col for col in cols_to_use if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = dataset[cols_to_use].copy()

    df["date"] = pd.to_datetime(df["date"])
    df[region_col] = df[region_col].astype(str).str.lower().str.strip()
    df[state_col] = df[state_col].astype(str).str.lower().str.strip()

    df = df.sort_values([region_col, "date"]).reset_index(drop=True)

    # add encoded columns here
    df["region_code"] = df[region_col].astype("category").cat.codes
    df["state_code"] = df[state_col].astype("category").cat.codes

    return df
