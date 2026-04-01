import pandas as pd

from helpers.build_aggregation_rules import build_aggregation_rules


def prepare_tree_model_data(
    dataset,
    target_col,
    selected_cols,
    level="region",
    region=None,
    state=None
):
    """
    Prepare data for tree-based models at different geographic levels.

    Parameters
    ----------
    dataset : pd.DataFrame
        Full dataset.
    target_col : str
        Target variable.
    selected_cols : list
        Columns to keep.
    level : str
        One of: "region", "state", "us"
    region : str, optional
        Region name for region-level modeling.
    state : str, optional
        State abbreviation for state-level modeling.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with one row per date for the chosen level.
    """

    required_cols = ["date", target_col]
    cols_to_use = list(dict.fromkeys(required_cols + selected_cols))

    missing_cols = [col for col in cols_to_use if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = dataset[cols_to_use].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # -------------------------
    # Level-specific filtering
    # -------------------------
    if level == "region":
        if region is None:
            raise ValueError("region must be provided when level='region'")

        if "county_name_x" not in dataset.columns and "region" not in dataset.columns:
            raise ValueError("No region column found. Expected 'county_name_x' or 'region'.")

        region_col = "county_name_x" if "county_name_x" in dataset.columns else "region"
        df[region_col] = dataset[region_col]

        df = df[df[region_col].str.lower() == region.lower()].copy()
        df = df.drop(columns=[region_col], errors="ignore")

    elif level == "state":
        if state is None:
            raise ValueError("state must be provided when level='state'")

        if "state" not in dataset.columns:
            raise ValueError("Column 'state' is required for state-level modeling.")

        df["state"] = dataset["state"].astype(str).str.lower()
        df = df[df["state"] == state.lower()].copy()
        df = df.drop(columns=["state"], errors="ignore")

        agg_rules = build_aggregation_rules(df.columns, target_col)
        df = df.groupby("date", as_index=False).agg(agg_rules)

    elif level == "us":
        agg_rules = build_aggregation_rules(df.columns, target_col)
        df = df.groupby("date", as_index=False).agg(agg_rules)

    else:
        raise ValueError("level must be one of: 'region', 'state', 'us'")

    # -------------------------
    # Final cleanup
    # -------------------------
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    return df
