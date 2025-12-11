"""
src/debug_anisotropy.py

Calculates Mean Cosine Similarity for all activation matrices to compare anisotropy levels.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import ast

RAW_DIR = Path("outputs/raw_activations")

def main():
    print(f"Scanning {RAW_DIR}...")
    files = list(RAW_DIR.glob("*.csv"))
    
    if not files:
        print("No CSV files found.")
        return
            
    results = []
    
    print("\n--- Anisotropy Report (Mean Cosine Similarity) ---")
    print(f"{'Model':<40} | {'Mean Sim':<10} | {'Rows':<10}")
    print("-" * 70)
    
    for fp in files:
        model_name = fp.stem
        print(f"Processing {model_name}...", end="\r")
        
        try:
            df = pd.read_csv(fp)
            
            # Extract Vectors
            # Check for 'vector' column (string) or dim columns
            if 'vector' in df.columns:
                # Parse string "[0.1, ...]"
                # Sample first to check format
                first = df['vector'].iloc[0]
                if isinstance(first, str) and first.startswith('['):
                     matrix = np.stack(df['vector'].apply(lambda x: np.array(ast.literal_eval(x))).values)
                else:
                     # Maybe already list?
                     matrix = np.stack(df['vector'].values)
            else:
                # Assume wide format
                cols = [c for c in df.columns if c.startswith('dim_') or c.isdigit()]
                if not cols:
                    continue
                matrix = df[cols].values
                
            # Filter Zero Vectors (though raw CSVs usually don't have them unless extraction failed)
            norms = np.linalg.norm(matrix, axis=1)
            valid_mask = norms > 0
            X_valid = matrix[valid_mask]
            
            if X_valid.shape[0] < 2:
                continue
                
            # Sample
            n_samples = min(100, X_valid.shape[0])
            indices = np.random.choice(X_valid.shape[0], n_samples, replace=False)
            sample = X_valid[indices]
            
            # Cosine Sim
            sim = cosine_similarity(sample)
            mean_sim = np.mean(sim[np.triu_indices(n_samples, k=1)])
            
            results.append((model_name, mean_sim, X_valid.shape[0]))
            print(f"{model_name:<40} | {mean_sim:.4f}     | {X_valid.shape[0]}")
            
        except Exception as e:
            print(f"Error reading {model_name}: {e}")

    print("-" * 70)
    
    # Sort by Similarity (Descending)
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Leaderboard (Most Anisotropic First) ---")
    for key, score, count in results:
        print(f"{key}: {score:.4f}")

if __name__ == "__main__":
    main()
