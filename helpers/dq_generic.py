import pandas as pd
import numpy as np
import glob
import argparse
import os
from great_tables import GT, md, loc, style

## produces data quality report for a given file as an html file
## use python helpers/dq_generic.py --path data/processed/{FILENAME}.csv --output reports/{FILENAME}_dq_report.html

def analyze_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return None, ""
            
        rows, cols = df.shape
        duplicates = df.duplicated().sum()
        total_cells = rows * cols
        
        stats_data = []
        top5_data = []
        
        for c in df.columns:
            total_vals = len(df)
            missing = df[c].isnull().sum()
            unique = df[c].nunique()
            valid_count = df[c].count()
            col_duplicates = valid_count - unique if valid_count > 0 else 0
            
            missing_pct = (missing / total_vals) if total_vals > 0 else 0
            dup_pct = (col_duplicates / total_vals) if total_vals > 0 else 0
            
            dtype = str(df[c].dtype)
            
            is_numeric = pd.api.types.is_numeric_dtype(df[c])
            mean_val = df[c].mean() if is_numeric else np.nan
            median_val = df[c].median() if is_numeric else np.nan
            q25 = df[c].quantile(0.25) if is_numeric else np.nan
            q75 = df[c].quantile(0.75) if is_numeric else np.nan
            std_val = df[c].std() if is_numeric else np.nan
            
            zeros = (df[c] == 0).sum() if is_numeric else 0 # if not numeric, no zeros
            zeros_pct = (zeros / total_vals) if total_vals > 0 else 0
            
            stats_data.append({
                "Column": c,
                "Type": dtype,
                "Missing": missing,
                "Missing %": missing_pct,
                "Unique": unique,
                "Duplicate": col_duplicates,
                "Duplicate %": dup_pct,
                "Zeros": zeros,
                "Zeros %": zeros_pct,
                "Mean": mean_val if pd.notnull(mean_val) else np.nan,
                "Median": median_val if pd.notnull(median_val) else np.nan,
                "25%": q25 if pd.notnull(q25) else np.nan,
                "75%": q75 if pd.notnull(q75) else np.nan,
                "Std Dev": std_val if pd.notnull(std_val) else np.nan
            })
            
            top5 = df[c].value_counts().head(5)
            top5_strs = [f"{idx} ({val})" for idx, val in top5.items()]
            top5_strs += ["-"] * (5 - len(top5_strs))
            
            top5_data.append({
                "Column": c,
                "Top 1": top5_strs[0],
                "Top 2": top5_strs[1],
                "Top 3": top5_strs[2],
                "Top 4": top5_strs[3],
                "Top 5": top5_strs[4]
            })

        df_stats = pd.DataFrame(stats_data)
        df_top5 = pd.DataFrame(top5_data)

        # Build great_tables for this specific file's components
        gt_stats = (
            GT(df_stats)
            .tab_header(title=f"Column-Level Statistics: {os.path.basename(file_path)}")
            .fmt_percent(columns=["Missing %", "Duplicate %", "Zeros %"], decimals=2)
            .fmt_number(columns=["Mean", "Median", "25%", "75%", "Std Dev"], decimals=2)
            .sub_missing(missing_text="-")
        )

        gt_top5 = (
            GT(df_top5)
            .tab_header(title=f"Top 5 Most Common Values: {os.path.basename(file_path)}")
            .sub_missing(missing_text="-")
        )

        file_html = f"<div style='margin-top: 40px;'><h2 style='font-family: sans-serif; color: #333;'>File: {os.path.basename(file_path)}</h2>\n"
        file_html += gt_stats.as_raw_html() + "<br><br>\n"
        file_html += gt_top5.as_raw_html() + "<br></div><hr>\n"
        
        missing_cells = df.isnull().sum().sum()
        missing_pct_overall = (missing_cells / total_cells) if total_cells > 0 else 0
        
        return {
            "File Name": os.path.basename(file_path),
            "Rows": rows,
            "Columns": cols,
            "Duplicates": duplicates,
            "Missing %": missing_pct_overall,
            "Status": "Success"
        }, file_html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "File Name": os.path.basename(file_path),
            "Rows": None,
            "Columns": None,
            "Duplicates": None,
            "Missing %": None,
            "Status": f"Error: {e}"
        }, f"<div style='margin-top:40px; color:red;'><h2>Error processing {os.path.basename(file_path)}:</h2><p>{e}</p></div><hr>"

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

    files_to_process.sort()

    if args.limit > 0 and len(files_to_process) > args.limit:
        print(f"Found {len(files_to_process)} files. Limiting to first {args.limit} files. Use --limit 0 to process all.")
        files_to_process = files_to_process[:args.limit]
        
    results = []
    html_parts = []
    
    for file in files_to_process:
        print(f"Analyzing {file}...")
        res, file_html = analyze_file(file)
        if res:
            results.append(res)
            if file_html:
                html_parts.append(file_html)
            
    if not results:
        print("No valid files processed.")
        return
        
    df_results = pd.DataFrame(results)
    
    gt_overview = (
        GT(df_results)
        .tab_header(
            title="Generic Data Quality Overview",
            subtitle=f"Overall summary of {len(results)} files processed from: {args.path}"
        )
        .fmt_number(columns=["Rows", "Columns", "Duplicates"], decimals=0)
        .fmt_percent(columns=["Missing %"], decimals=2)
        .sub_missing(missing_text="-")
        .tab_style(
            style=style.text(color="red", weight="bold"),
            locations=loc.body(rows=lambda df: df["Status"] != "Success")
        )
    )
    
    full_html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>Data Quality Report</title>\n"
        "<style>body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 30px; background-color: #fcfcfc; }</style>\n"
        "</head>\n<body>\n"
    )
    
    full_html += gt_overview.as_raw_html()
    full_html += "<br><hr style='border:1px solid #ddd;'><br>\n"
    full_html += "".join(html_parts)
    full_html += "</body>\n</html>"
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"\nSuccess! Great Tables data quality report complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
