"""
src/utils/check_model_consistency.py

Consistency Check:
Calculates how strongly different models correlate with EACH OTHER on specific norms.
If models disagree, the prompt might be ambiguous or the models unstable.

Outputs:
- CLI Summary of Mean Inter-Model Correlation.
- Heatmaps in 'outputs/consistency_plots/'
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
NORMS_DIR = BASE_DIR / "outputs" / "raw_behavior" / "model_norms"
OUTPUT_DIR = BASE_DIR / "outputs" / "consistency_plots"

# Filter to keep graphs readable (optional, set None to use all)
# SELECTED_MODELS = [
#     "llama-3.1-8b-instruct", "llama-3.3-70b-instruct", 
#     "mistral-small-24b-instruct", "qwen-3-32b-instruct"
# ]
SELECTED_MODELS = None # Use all found

def load_all_norms():
    print(f"Loading data from {NORMS_DIR}...")
    all_data = []
    
    files = sorted(list(NORMS_DIR.glob("*.csv")))
    
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'cleaned_rating' not in df.columns: continue
            
            # Standardize columns
            norm_col = 'norm' if 'norm' in df.columns else 'norm_name'
            
            # Clean
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            
            # Keep minimal columns
            sub = df[['word', norm_col, 'cleaned_rating']].copy()
            sub.rename(columns={norm_col: 'norm'}, inplace=True)
            sub['model'] = f.stem # e.g. 'llama-3.1-8b-instruct'
            
            if SELECTED_MODELS and f.stem not in SELECTED_MODELS:
                continue
                
            all_data.append(sub)
            
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
            
    return pd.concat(all_data, ignore_index=True)

def analyze_consistency(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of all norms
    norms = df['norm'].unique()
    print(f"Analyzing consistency for {len(norms)} norms...")
    
    summary_stats = []
    
    for norm in tqdm(norms):
        # Filter for this norm
        subset = df[df['norm'] == norm]
        
        # Pivot: Row=Word, Col=Model, Val=Rating
        pivot = subset.pivot_table(index='word', columns='model', values='cleaned_rating')
        
        # We need at least 2 models and some overlapping words
        if pivot.shape[1] < 2 or len(pivot) < 10:
            continue
            
        # Calculate Correlation Matrix (Pearson)
        corr_mat = pivot.corr(method='pearson')
        
        # Calculate Mean Correlation (excluding diagonal 1.0s)
        # We use a mask to ignore the upper triangle to avoid duplicates and self-corr
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        lower_tri = corr_mat.mask(mask)
        mean_corr = lower_tri.stack().mean()
        
        summary_stats.append({
            "norm": norm,
            "mean_inter_model_corr": mean_corr,
            "n_words_overlap": len(pivot)
        })
        
        # Plot Heatmap for major norms (optional condition to avoid 100s of plots)
        # Plot if mean_corr is suspiciously low (< 0.5) or for key norms
        if mean_corr < 0.6 or any(k in norm for k in ["valence", "concreteness", "arousal"]):
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_mat, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f")
            plt.title(f"Model Agreement: {norm}\nMean r = {mean_corr:.2f} (N={len(pivot)})")
            plt.tight_layout()
            
            safe_name = norm.replace("/", "_")
            plt.savefig(OUTPUT_DIR / f"consistency_{safe_name}.png")
            plt.close()

    # Summary Report
    summary_df = pd.DataFrame(summary_stats).sort_values("mean_inter_model_corr", ascending=False)
    print("\n" + "="*60)
    print("CONSISTENCY LEADERBOARD (Which norms do models agree on?)")
    print("="*60)
    print(summary_df.head(10).to_string(index=False))
    print("\n" + "-"*60)
    print("LEAST CONSISTENT (Models disagree on these):")
    print("-"*60)
    print(summary_df.tail(10).to_string(index=False))
    
    summary_df.to_csv(OUTPUT_DIR / "consistency_summary.csv", index=False)
    print(f"\nFull report saved to {OUTPUT_DIR}/consistency_summary.csv")

if __name__ == "__main__":
    df = load_all_norms()
    if not df.empty:
        analyze_consistency(df)
    else:
        print("No data loaded.")