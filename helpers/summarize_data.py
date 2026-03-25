import pandas as pd
import os

def summarize_data(file_path):
    """Loads a CSV file and prints basic statistics for its features."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please ensure you are running this script from the root of the project directory.")
        return
    
    # Ensure pandas doesn't truncate output when saving to file
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Loading data from {file_path}...")
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    print(f"Total Rows: {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]:,}")
    
    print("\n" + "="*50)
    print("COLUMN INFORMATION & DATA TYPES")
    print("="*50)
    df.info()
    
    print("\n" + "="*50)
    print("BASIC STATISTICS (NUMERICAL FEATURES)")
    print("="*50)
    print(df.describe().T)
    
    print("\n" + "="*50)
    print("BASIC STATISTICS (CATEGORICAL FEATURES)")
    print("="*50)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe().T)
    else:
        print("No categorical features found.")
        
    print("\n" + "="*50)
    print("MISSING VALUES")
    print("="*50)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage (%)': (missing_data / len(df) * 100).round(2)
        })
        print(missing_df)
    else:
        print("No missing values found.")

if __name__ == "__main__":
    import contextlib

    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "data_summary.txt")

    # Using a relative file reference pointing to the processed data directory
    relative_path = os.path.join("./", "data", "processed", "RDC_Inventory_Core_Metrics_Zip_History.csv")
    
    print(f"Writing summary to {report_path}...")
    with open(report_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            summarize_data(relative_path)
    print("Done!")
