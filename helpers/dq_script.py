import pandas as pd
import io

files = [
    'c:/Users/sbout/Documents/school/URI/capstone/covid-housing-forecast/data/processed/processed_data.csv',
    'c:/Users/sbout/Documents/school/URI/capstone/covid-housing-forecast/data/processed/processed_data_low.csv'
]

report_lines = ["# Data Quality Report\n\n"]

for file in files:
    try:
        df = pd.read_csv(file)
        report_lines.append(f"## File: `{file.split('/')[-1]}`\n\n")
        report_lines.append(f"- **Rows**: {df.shape[0]}\n")
        report_lines.append(f"- **Columns**: {df.shape[1]}\n")
        report_lines.append(f"- **Duplicates**: {df.duplicated().sum()}\n\n")
        
        report_lines.append("### Missing Values\n\n")
        missing_count = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_count, 'Percentage (%)': missing_percent})
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if len(missing_df) == 0:
            report_lines.append("No missing values.\n\n")
        else:
            report_lines.append(missing_df.to_markdown() + "\n\n")
            
        report_lines.append("### Data Types\n\n")
        dtypes_df = df.dtypes.to_frame('dtype').reset_index().rename(columns={'index': 'Column', 'dtype': 'Data Type'})
        dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)
        report_lines.append(dtypes_df.to_markdown(index=False) + "\n\n")

        report_lines.append("### Summary Statistics (Numerical)\n\n")
        num_df = df.select_dtypes(include=['number'])
        if not num_df.empty:
            report_lines.append(num_df.describe().T.to_markdown() + "\n\n")
        else:
            report_lines.append("No numerical columns.\n\n")
            
        report_lines.append("### Summary Statistics (Categorical)\n\n")
        cat_df = df.select_dtypes(exclude=['number'])
        if not cat_df.empty:
            report_lines.append(cat_df.describe().T.to_markdown() + "\n\n")
        else:
            report_lines.append("No categorical columns.\n\n")
            
    except Exception as e:
        report_lines.append(f"**Error processing {file}**: {e}\n\n")

with open('C:/Users/sbout/.gemini/antigravity/brain/c3c65a8c-bfad-4d7f-96b1-356465dc59e2/analysis_results.md', 'w') as f:
    f.writelines(report_lines)
