import pandas as pd
import numpy as np
from helpers.model_evaluator import evaluate_model

def seasonal_naive_model(data, target, filter_col, filter_by, periods):
    
    region = data[data[filter_col] == filter_by]
    region["date"] = pd.to_datetime(region["date"])
    region = region.sort_values("date")
    region = region[["date", target]].dropna()
    region = region.set_index("date")
    region = region.asfreq("MS")
    region = region.dropna(subset=[target])
    
    if len(region) <= periods:
        raise ValueError(
            f"Not enough observations for region '{filter_by}'. Need more than {periods} rows."
        )
    
    
    selected_target = region[target]
    
    train = selected_target.iloc[:-periods]
    test = selected_target.iloc[-periods:]
    
    forecast = []
    train_values = train.values
    
    for i in range(len(test)):
        forecast.append(train_values[-periods + (i % periods)])
    
    evaluation_result = evaluate_model(
        y_true=test,
        y_pred=forecast,
        model_name="Seasonal Naive",
        metrics=["rmse", "mae", "mase", "mape"],
        train=train
    )
    
    return {
        "forecast": forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result
    }
    