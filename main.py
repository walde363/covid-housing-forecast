from src.rf_xgb_models import rf_xgb_models
import pandas as pd

def main():
    data = pd.read_csv("data/processed/processed_data_low.csv")
    
    cols_to_drop = [
        "RegionID",
        "Month",
        "StateName",
        "NewConSales",
        "ZORDI_Condo",
        "ZORDI_MFR",
        "ZORDI_SFR",
        "DaysToClose",
    ]
    result = rf_xgb_models("MarketTemp", data, cols_to_drop)
    
    print(result)

if __name__ == "__main__":
    main()