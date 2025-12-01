"""
src/utils/check_model_comprehension_strict.py

"The Strict Comprehension Check"
Calculates the 'Leave-One-Out Consensus Correlation' for each model,
BUT ONLY on the subset of words that EVERY model successfully rated.

This removes "Coverage Bias." We are testing:
"On the concepts we ALL know, do you agree with the group?"
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
OUTPUT_DIR = BASE_DIR / "outputs" / "comprehension_check_strict"

# Set True to exclude specific models that might kill the intersection size
DROP_BAD_MODELS = False 
BAD_MODELS = ["gpt-oss-20b-instruct"] 

def load_all_norms():
    print(f"Loading data from {NORMS_DIR}...")
    all_data = []
    files = sorted(list(NORMS_DIR.glob("*.csv")))
    
    for f in files:
        if DROP_BAD_MODELS and f.stem in BAD_MODELS:
            continue
            
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'cleaned_rating' not in df.columns: continue
            
            # Standardize
            norm_col = 'norm' if 'norm' in df.columns else 'norm_name'
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            
            sub = df[['word', norm_col, 'cleaned_rating']].copy()
            sub.rename(columns={norm_col: 'norm'}, inplace=True)
            sub['model'] = f.stem
            all_data.append(sub)
        except:
            pass
            
    return pd.concat(all_data, ignore_index=True)

def analyze_strict_comprehension(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    norms = df['norm'].unique()
    
    results = []
    
    print(f"Analyzing STRICT comprehension across {len(norms)} norms...")
    
    for norm in tqdm(norms):
        subset = df[df['norm'] == norm]
        pivot = subset.pivot_table(index='word', columns='model', values='cleaned_rating')
        
        # --- STRICT FILTER ---
        # Keep only words where ALL models have a value
        pivot_strict = pivot.dropna()
        
        # Need at least 3 models and some shared words
        if pivot_strict.shape[1] < 3 or len(pivot_strict) < 10:
            continue
            
        # Leave-One-Out Correlation on the STRICT set
        for model in pivot_strict.columns:
            # 1. Consensus = Mean of everyone else
            other_models = [c for c in pivot_strict.columns if c != model]
            consensus = pivot_strict[other_models].mean(axis=1)
            
            # 2. Correlate Model vs Consensus
            corr = np.corrcoef(pivot_strict[model], consensus)[0, 1]
            
            results.append({
                "model": model,
                "norm": norm,
                "strict_consensus_r": corr,
                "n_shared_words": len(pivot_strict)
            })

    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No valid intersections found. Try enabling DROP_BAD_MODELS.")
        return

    # --- AGGREGATION & PLOTTING ---
    
    # 1. Leaderboard
    avg_scores = results_df.groupby("model").agg(
        mean_r=("strict_consensus_r", "mean"),
        avg_n=("n_shared_words", "mean")
    ).reset_index().sort_values("mean_r", ascending=False)
    
    print("\n" + "="*70)
    print("STRICT COMPREHENSION RANKING (Shared Vocabulary Only)")
    print("="*70)
    print(avg_scores.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_scores, x="mean_r", y="model", palette="viridis")
    plt.title("Strict Comprehension Check\n(Correlation with Group Consensus on Shared Words)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Correlation ($r$)")
    plt.ylabel("")
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strict_comprehension_scores.png")
    
    results_df.to_csv(OUTPUT_DIR / "strict_comprehension_full.csv", index=False)
    print(f"\nSaved results to {OUTPUT_DIR}")

if __name__ == "__main__":
    df = load_all_norms()
    if not df.empty:
        analyze_strict_comprehension(df)