
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(base_dir):
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    dfs = []
    
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            model_name = os.path.basename(filepath).replace(".csv", "")
            
            # Clean data: drop rows with NaN norm or cleaned_rating
            df = df.dropna(subset=['norm', 'cleaned_rating'])
            
            # Ensure cleaned_rating is numeric
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            
            # Keep only necessary columns
            df = df[['norm', 'word', 'cleaned_rating']].copy()
            df['model'] = model_name
            
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

def analyze_correlations(full_df):
    # Pivot to wide format: Index=[norm, word], Columns=[model]
    pivot_df = full_df.pivot_table(index=['norm', 'word'], columns='model', values='cleaned_rating')
    
    # 1. Global Correlation
    global_corr = pivot_df.corr(method='pearson')
    
    # 2. Per-Norm Correlation
    norm_corrs = {}
    norms = full_df['norm'].unique()
    
    for norm in norms:
        norm_data = pivot_df.xs(norm, level='norm')
        # Only compute if we have enough data
        if len(norm_data) > 10:
            norm_corrs[norm] = norm_data.corr(method='pearson')
            
    return global_corr, norm_corrs, pivot_df

def plot_correlations(global_corr, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(global_corr, annot=True, cmap='RdBu_r', vmin=0, vmax=1, fmt=".3f")
    plt.title("Global Model Correlation (All Norms)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_descriptive_stats(full_df, output_path):
    # Calculate stats for the plot
    # We want a plot that shows the distribution or mean/sd for each norm per model.
    # Since there are 15 norms, a faceted plot is best.
    
    g = sns.FacetGrid(full_df, col="norm", col_wrap=5, sharex=False, sharey=False, height=3, aspect=1.2)
    
    # Using boxplot to show distribution
    # g.map_dataframe(sns.boxplot, x="model", y="cleaned_rating", showfliers=False) 
    
    # Alternatively pointplot for Mean +/- CI (or SD)
    # Let's do a pointplot to see mean differences clearly
    g.map_dataframe(sns.pointplot, x="model", y="cleaned_rating", 
                    errorbar='sd', # Show standard deviation
                    linestyle="none", capsize=.2)
    
    # Adjust layout
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            
    g.fig.suptitle("Model Norm Ratings: Mean ± SD", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    base_dir = "outputs/raw_behavior/model_norms"
    plots_dir = "outputs/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading data...")
    full_df = load_data(base_dir)
    if full_df is None:
        print("No data found.")
        return

    print("Analyzing correlations...")
    global_corr, norm_corrs, pivot_df = analyze_correlations(full_df)
    
    print("\n=== Global Correlation Matrix ===")
    print(global_corr)
    
    print("\n=== Per-Norm Correlations (Mean Correlation per Norm) ===")
    # Summarize per-norm correlation by taking the average off-diagonal correlation for each norm
    norm_summary = []
    for norm, corr_mat in norm_corrs.items():
        # Get off-diagonal elements
        mask = np.ones(corr_mat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        avg_corr = corr_mat.values[mask].mean()
        norm_summary.append({'norm': norm, 'avg_inter_model_corr': avg_corr})
    
    norm_summary_df = pd.DataFrame(norm_summary).sort_values('avg_inter_model_corr', ascending=False)
    print(norm_summary_df.to_string(index=False))

    print("\nGenerating Plots...")
    plot_correlations(global_corr, os.path.join(plots_dir, "model_norm_correlations.png"))
    plot_descriptive_stats(full_df, os.path.join(plots_dir, "model_norm_descriptive_stats.png"))
    
    print("\n=== Descriptive Statistics (Aggregated) ===")
    stats = full_df.groupby(['model', 'norm'])['cleaned_rating'].agg(['mean', 'std', 'count']).reset_index()
    # Print a sample or summary
    # Let's print the mean rating per model per norm in a wide format for readability
    mean_pivot = stats.pivot(index='norm', columns='model', values='mean')
    print("\nMean Ratings per Norm:")
    print(mean_pivot)
    
    print(f"\nAnalysis complete. Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
