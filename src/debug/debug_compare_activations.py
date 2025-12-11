"""
src/debug_compare_activations.py

Compares raw CSV activations to the processed pickle embeddings to check for corruption.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

EMBEDDINGS_PATH = Path("outputs/matrices/embeddings.pkl")
CSV_PATH = Path("outputs/raw_activations/mistral-small-24b-instruct.csv")
MODEL_KEY = "activation_mistral-small-24b-instruct"

def main():
    # 1. Load Pickle
    print(f"Loading {EMBEDDINGS_PATH}...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict) and 'embeddings' in data:
            embeddings = data['embeddings']
            mappings = data['mappings']
        else:
            embeddings = data
            mappings = data.get('mappings')
            
    if MODEL_KEY not in embeddings:
        print(f"Key {MODEL_KEY} not found in pickle.")
        return
        
    pkl_matrix = embeddings[MODEL_KEY]
    cue_to_idx = mappings['cue_to_idx']
    
    # 2. Load CSV
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Identify word column
    col_name = 'word' if 'word' in df.columns else 'cue'
    if col_name not in df.columns:
        print("Could not find word column in CSV.")
        return
        
    # Identify vector columns
    # Assuming all other columns are dimensions? Or is there a 'vector' column?
    # Based on previous logs, it seemed to be wide format 'dim_0', 'dim_1'...
    # Let's check columns
    cols = list(df.columns)
    vector_cols = [c for c in cols if c.startswith('dim_') or c.isdigit()]
    
    if not vector_cols:
        # Maybe it's a single 'vector' column with string representation?
        if 'vector' in df.columns:
            print("Detected 'vector' column (string format).")
            # Parse string "[0.1, ...]"
            import ast
            df['parsed_vec'] = df['vector'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
            use_parsed = True
        else:
            print("Could not identify vector columns.")
            return
    else:
        print(f"Detected {len(vector_cols)} dimension columns.")
        use_parsed = False

    # 3. Compare
    print("\n--- Comparison ---")
    
    # Pick 5 random words
    sample_words = df[col_name].sample(5).tolist()
    
    for word in sample_words:
        word_clean = str(word).lower().strip()
        
        if word_clean not in cue_to_idx:
            print(f"Word '{word}' not in canonical vocab. Skipping.")
            continue
            
        # Get Pickle Vector
        idx = cue_to_idx[word_clean]
        vec_pkl = pkl_matrix[idx]
        
        # Get CSV Vector
        row = df[df[col_name] == word].iloc[0]
        if use_parsed:
            vec_csv = np.array(row['parsed_vec'])
        else:
            vec_csv = row[vector_cols].values.astype(float)
            
        # Check Dimensions
        if vec_pkl.shape != vec_csv.shape:
            print(f"FAIL: Shape Mismatch for '{word}'. Pkl: {vec_pkl.shape}, CSV: {vec_csv.shape}")
            continue
            
        # Check Values
        diff = np.linalg.norm(vec_pkl - vec_csv)
        is_close = np.allclose(vec_pkl, vec_csv, atol=1e-5)
        
        print(f"Word: {word}")
        print(f"  > Diff (L2): {diff:.6f}")
        print(f"  > Identical? {is_close}")
        
        if not is_close:
            print(f"  > Sample Pkl: {vec_pkl[:5]}")
            print(f"  > Sample CSV: {vec_csv[:5]}")

if __name__ == "__main__":
    main()
