import pandas as pd
import glob
import argparse
import os
from great_tables import GT, md

def analyze_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return None
            
        rows, cols = df.shape
        duplicates = df.duplicated().sum()
        total_cells = rows * cols
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) if total_cells > 0 else 0
        
        return {
            "File Name": os.path.basename(file_path),
            "Rows": rows,
            "Columns": cols,
            "Duplicates": duplicates,
            "Missing %": missing_pct,
            "Status": "Success"
        }
    except Exception as e:
        return {
            "File Name": os.path.basename(file_path),
            "Rows": None,
            "Columns": None,
            "Duplicates": None,
            "Missing %": None,
            "Status": f"Error: {e}"
        }

def main():
    parser = argparse.ArgumentParser(description="Generate a generic Data Quality report using great_tables.")
    parser.add_argument("--path", type=str, default="data/raw", 
                        help="Path to a directory or a specific data file to analyze.")
    parser.add_argument("--limit", type=int, default=10, 
                        help="Limit the number of files analyzed (for performance in large dirs). Set to 0 for unlimited.")
    parser.add_argument("--output", type=str, default="reports/generic_dq_report.html", 
                        help="Path to save the generated HTML table.")
    
    args = parser.parse_args()
    
    files_to_process = []
    if os.path.isfile(args.path):
        files_to_process.append(args.path)
    elif os.path.isdir(args.path):
        for ext in ["*.csv", "*.xls", "*.xlsx"]:
            files_to_process.extend(glob.glob(os.path.join(args.path, ext)))
    else:
        print(f"Path not found: {args.path}")
        return

    # Sort files by size to potentially process smaller ones first or just sort alphabetically
    files_to_process.sort()

    if args.limit > 0 and len(files_to_process) > args.limit:
        print(f"Found {len(files_to_process)} files. Limiting to first {args.limit} files. Use --limit 0 to process all.")
        files_to_process = files_to_process[:args.limit]
        
    results = []
    for file in files_to_process:
        print(f"Analyzing {file}...")
        res = analyze_file(file)
        if res:
            results.append(res)
            
    if not results:
        print("No valid files processed.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Create the great_tables chart
    from great_tables import loc, style
    gt = (
        GT(df_results)
        .tab_header(
            title="Generic Data Quality Overview",
            subtitle=f"Analysis of files from: {args.path}"
        )
        .fmt_number(columns=["Rows", "Columns", "Duplicates"], decimals=0)
        .fmt_percent(columns=["Missing %"], decimals=2)
        .sub_missing(missing_text="-")
        .tab_style(
            style=style.text(color="red", weight="bold"),
            locations=loc.body(rows=lambda df: df["Status"] != "Success")
        )
    )
    
    # Ensure reports directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(gt.as_raw_html())
        
    print(f"Great Tables data quality report saved to {args.output}")

if __name__ == "__main__":
    main()
