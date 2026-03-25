import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def plot_missingness(df, output_dir):
    """Generates a bar plot showing the percentage of missing values per column."""
    plt.figure(figsize=(10, 6))
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        sns.barplot(x=missing.values, y=missing.index, hue=missing.index, legend=False, palette='viridis')
        plt.title('Percentage of Missing Values per Column')
        plt.xlabel('Percentage (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missingness.png'))
    else:
        print("No missing values found to plot.")
    plt.close()

def plot_outliers(df, output_dir):
    """Generates horizontal box plots for the top 10 numerical columns to visualize outliers."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        # Limit to 10 for readability in a single plot
        num_cols = num_cols[:10]
        plt.figure(figsize=(12, 6))
        
        # Standardize the data quickly for better visualization on a single axes, 
        # normally you would plot separately, but a normalized view works nicely.
        df_scaled = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
        
        sns.boxplot(data=df_scaled, orient='h', palette='Set2')
        plt.title('Outlier Visualization (Standardized Top 10 Numerical Columns)')
        plt.xlabel('Z-Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'outliers.png'))
    plt.close()

def plot_distributions(df, output_dir):
    """Plots histograms with KDE overlays for the first 6 numerical columns to view shifts/shapes."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        num_cols = num_cols[:6] # Limit to 6 for a 2-column grid
        fig, axes = plt.subplots((len(num_cols) + 1) // 2, 2, figsize=(14, 4 * ((len(num_cols) + 1) // 2)))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='royalblue')
            axes[i].set_title(f'Distribution of {col}')
            
        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distributions.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize missingness, outliers, and distributions.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default="reports/visualizations", help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    
    print(f"Generating visualizations for {os.path.basename(args.input)}...")
    plot_missingness(df, args.output_dir)
    plot_outliers(df, args.output_dir)
    plot_distributions(df, args.output_dir)
    
    print(f"Success: Visualizations successfully exported as PNGs to: {args.output_dir}")

if __name__ == "__main__":
    main()
