
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import scipy.stats as stats
import numpy as np

def find_file(base_dir, pattern):
    files = glob.glob(os.path.join(base_dir, pattern))
    if not files:
        # Try looser pattern
        files = glob.glob(os.path.join(base_dir, "*mistral*.csv"))
        if not files:
             raise FileNotFoundError(f"No file matching {pattern} or 'mistral' in {base_dir}")
    if len(files) > 1:
        # Prefer the one with 'small' and '24b' if possible
        better_files = [f for f in files if 'small' in f and '24b' in f]
        if better_files:
            return better_files[0]
        print(f"Warning: Multiple files found in {base_dir}: {files}. Using {files[0]}")
    return files[0]

def analyze():
    # Paths
    dir_orig = "outputs/raw_behavior/model_norms/"
    dir_rep = "outputs/raw_behavior/model_norms_replic/"
    plot_dir = "outputs/plots/comparison_mistral_runs/"
    os.makedirs(plot_dir, exist_ok=True)

    print("Searching for files...")
    # Find files
    try:
        file_orig = find_file(dir_orig, "*mistral*small*24b*.csv")
        file_rep = find_file(dir_rep, "*mistral*small*24b*.csv")
    except Exception as e:
        print(f"Error finding files: {e}")
        return

    print(f"Loading Original: {file_orig}")
    print(f"Loading Replication: {file_rep}")

    # Load
    try:
        df_orig = pd.read_csv(file_orig)
        df_rep = pd.read_csv(file_rep)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # Check columns
    req_cols = ['norm', 'word', 'cleaned_rating']
    if not all(col in df_orig.columns for col in req_cols):
        print(f"Original file missing columns: {df_orig.columns}")
        return
    if not all(col in df_rep.columns for col in req_cols):
        print(f"Replication file missing columns: {df_rep.columns}")
        return

    # Rename value columns
    df_orig = df_orig.rename(columns={'cleaned_rating': 'rating_orig'})
    df_rep = df_rep.rename(columns={'cleaned_rating': 'rating_rep'})

    # Force numeric and handle missing
    df_orig['rating_orig'] = pd.to_numeric(df_orig['rating_orig'], errors='coerce')
    df_rep['rating_rep'] = pd.to_numeric(df_rep['rating_rep'], errors='coerce')

    print(f"NaNs in Orig after coercion: {df_orig['rating_orig'].isna().sum()}")
    print(f"NaNs in Rep after coercion: {df_rep['rating_rep'].isna().sum()}")

    df_orig = df_orig.dropna(subset=['rating_orig'])
    df_rep = df_rep.dropna(subset=['rating_rep'])

    # Merge
    # We strip whitespace and lowercase just in case for robust matching
    df_orig['word'] = df_orig['word'].astype(str).str.strip().str.lower()
    df_rep['word'] = df_rep['word'].astype(str).str.strip().str.lower()
    
    # Remove duplicates
    orig_dupes = df_orig.duplicated(subset=['norm', 'word']).sum()
    rep_dupes = df_rep.duplicated(subset=['norm', 'word']).sum()
    print(f"Dropping duplicates - Orig: {orig_dupes}, Rep: {rep_dupes}")
    
    df_orig = df_orig.drop_duplicates(subset=['norm', 'word'])
    df_rep = df_rep.drop_duplicates(subset=['norm', 'word'])
    
    print(f"Original Unique Count: {len(df_orig)}")
    print(f"Replication Unique Count: {len(df_rep)}")

    merged = pd.merge(df_orig, df_rep, on=['norm', 'word'], suffixes=('_orig', '_rep'), how='inner')
    print(f"Merged Row Count: {len(merged)}")
    
    if len(merged) == 0:
        print("CRITICAL ERROR: Merge resulted in 0 rows. Check 'norm' and 'word' columns.")
        return

    # Metrics
    norms = merged['norm'].unique()
    print(f"\nAnalysis per Norm ({len(norms)} norms found):")
    print(f"{'Norm':<40} | {'Pearson':<8} | {'Spearman':<8} | {'MAE':<6} | {'N':<5}")
    print("-" * 80)

    success = True
    
    for norm in norms:
        sub = merged[merged['norm'] == norm]
        if len(sub) < 5:
            print(f"{norm:<40} | Too few samples ({len(sub)})")
            continue
            
        x = sub['rating_orig']
        y = sub['rating_rep']
        
        # Correlation
        if x.std() == 0 or y.std() == 0:
             r, p = np.nan, np.nan
             rho, sp = np.nan, np.nan
             print(f"{norm:<40} | {'NaN':<8} | {'NaN':<8} | {np.mean(np.abs(x - y)):.4f} | {len(sub):<5} (Zero variance)")
        else:
             r, p = stats.pearsonr(x, y)
             rho, sp = stats.spearmanr(x, y)
             mae = np.mean(np.abs(x - y))
             n = len(sub)
             print(f"{norm:<40} | {r:.4f}   | {rho:.4f}   | {mae:.4f} | {n:<5}")
             
             if not np.isnan(r) and r < 0.9:
                 success = False

        # Plot
        plt.figure(figsize=(10, 6))
        
        # Melt for seaborn
        plot_df = pd.DataFrame({
            'Rating': pd.concat([x, y]),
            'Run': ['Original'] * len(x) + ['Replication'] * len(y)
        })
        
        # Hist + KDE
        try:
            sns.histplot(data=plot_df, x='Rating', hue='Run', kde=True, element="step", stat="density", common_norm=False)
            plt.title(f"Distribution Comparison: {norm}\nr={r:.4f}, mae={mae:.4f}")
            plt.savefig(f"{plot_dir}/{norm}_dist.png")
            plt.close()
            
            # Scatter
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=x, y=y, alpha=0.5)
            
            # Identity line
            min_val = min(x.min(), y.min())
            max_val = max(x.max(), y.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
            
            plt.xlabel("Original Rating")
            plt.ylabel("Replication Rating")
            plt.title(f"Scatter: {norm}")
            plt.savefig(f"{plot_dir}/{norm}_scatter.png")
            plt.close()
        except Exception as plot_e:
            print(f"Error plotting {norm}: {plot_e}")

    print("-" * 80)
    
    # Overall
    overall_r = merged['rating_orig'].corr(merged['rating_rep'])
    print(f"\nOverall Pearson Correlation: {overall_r:.4f}")
    
    if success:
        print("\nCONCLUSION: Replication SUCCESSFUL (High correlations across norms).")
    else:
        print("\nCONCLUSION: Replication SHOWS DEVIATIONS (Some correlations < 0.9).")
        
    print(f"\nPlots saved to {plot_dir}")

if __name__ == "__main__":
    analyze()
