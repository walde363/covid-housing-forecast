import pandas as pd
from prophet import Prophet

from helpers.prepare_tree_model_data import prepare_tree_model_data
from helpers.model_evaluator import evaluate_model

def train_prophet(train_df, test_df, params):
    model = Prophet(**params)
    model.fit(train_df)
    
    future = pd.DataFrame({'ds': test_df['ds']})
    forecast = model.predict(future)
    
    predictions = forecast['yhat'].values
    return model, predictions, forecast


def prophet_model_pipeline(
    target_col,
    dataset,
    selected_cols,
    params,
    level="region",
    region=None,
    state=None,
    test_periods=12
):
    """
    Prophet pipeline supporting region/state/us modeling.
    """

    df_model = prepare_tree_model_data(
        dataset=dataset,
        target_col=target_col,
        selected_cols=selected_cols,
        level=level,
        region=region,
        state=state
    )

    # Prophet requires 'ds' and 'y' columns
    pdf = df_model[["date", target_col]].copy()
    pdf.columns = ['ds', 'y']
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    pdf = pdf.dropna().sort_values("ds").reset_index(drop=True)
    
    # Manual time-based split
    if len(pdf) <= test_periods:
        raise ValueError(f"Not enough data points ({len(pdf)}) for test_periods={test_periods}")
        
    train_df = pdf.iloc[:-test_periods].copy()
    test_df = pdf.iloc[-test_periods:].copy()

    # Fit and Predict
    model, pred, _ = train_prophet(train_df, test_df, params)

    # Evaluate using the standard evaluate_model routine
    evaluation_result = evaluate_model(
        y_true=test_df['y'].values,
        y_pred=pred,
        model_name=f"Prophet ({level})",
        metrics=["rmse", "mae", "mase", "mape"],
        train=train_df['y'].values
    )

    # 18-month forward projection from the end of the test set
    last_test_date = test_df['ds'].max()
    
    # Generate exactly 18 months starting the month AFTER the last test date
    # M defaults to month-end, MS is month-start. We use MS since real estate data is usually YYYY-MM-01.
    future_dates = pd.date_range(start=last_test_date, periods=18 + 1, freq='MS')[1:]
    future_df_18m = pd.DataFrame({'ds': future_dates})
    
    forecast_18m = model.predict(future_df_18m)
    
    future_forecast = pd.Series(
        forecast_18m['yhat'].values,
        index=forecast_18m['ds'],
        name="future_forecast"
    )

    train = train_df.set_index("ds")['y']
    test = test_df.set_index("ds")['y']

    forecast = pd.Series(
        pred,
        index=test.index,
        name="forecast"
    )

    return {
        "model": model,
        "forecast": forecast,
        "future_forecast": future_forecast,
        "train": train,
        "test": test,
        "eval_results": evaluation_result,
        "df_model": df_model
    }
