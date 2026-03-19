from src.rf_xgb_models import rf_xgb_models
import pandas as pd

def main():
    data = pd.read_csv("data/processed/processed_data_low.csv")
    
    selected_cols = ["SizeRank", "RegionName", "RegionType", "MarketTemp", "date", "ZHVI_Tier", "ZORDI_All"]
    result = rf_xgb_models("ZHVI_Tier", data, selected_cols)
    
    print(result)

if __name__ == "__main__":
    main()