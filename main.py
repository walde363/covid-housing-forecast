from src.rf_xgb_models import rf_xgb_models
from src.sarimax_model import sarimax_model
from src.seasonal_naive_model import seasonal_naive_model
import pandas as pd

def main():
    data = pd.read_csv("data/processed/processed_data_low.csv")
    
    # RF and XGB:
    # selected_cols = ["SizeRank", "RegionName", "RegionType", "MarketTemp", "date", "ZHVI_Tier", "ZORDI_All"]
    # result = rf_xgb_models("ZHVI_Tier", data, selected_cols)
    # print(result)
    #--------------------------------------------------------------------------
    # SARIMAX
    # result = sarimax_model(data, "Abilene, TX", "MarketTemp")
    # print(result)
    # Seasonal Naive
    result = seasonal_naive_model(data, "MarketTemp", "RegionName", "Abilene, TX", 12)
    print(result)


if __name__ == "__main__":
    main()