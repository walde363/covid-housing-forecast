def build_aggregation_rules(columns, target_col):
    """
    Build aggregation rules for aggregate modeling.

    Rules:
    - count-like columns -> sum
    - share/rate/ratio/price-like columns -> mean
    - target_col -> mean
    - date is excluded
    """

    agg_rules = {}

    sum_keywords = [
        "count", "inventory", "listings", "permits", "sales", "volume"
    ]

    mean_keywords = [
        "price", "share", "ratio", "rate", "days", "score", "median", "average"
    ]

    for col in columns:
        if col == "date":
            continue

        col_lower = col.lower()

        if col == target_col:
            agg_rules[col] = "mean"
        elif any(word in col_lower for word in sum_keywords):
            agg_rules[col] = "sum"
        elif any(word in col_lower for word in mean_keywords):
            agg_rules[col] = "mean"
        else:
            # safe default
            agg_rules[col] = "mean"

    return agg_rules
