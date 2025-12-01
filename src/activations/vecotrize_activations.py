"""
src/activations/vectorize_activations.py

Pipeline A (Feature Construction).
1. Loads the Master Vocabulary (from behavioral_embeddings.pkl).
2. Reads raw activation CSVs.
3. Aligns strictly to the Master Index.
4. Fills missing words with NaN (Not a Number) to strictly denote missing data.
"""

import pandas as pd
import numpy as np
import pickle
import ast
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "outputs" / "raw_activations"
MASTER_PKL = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"
OUTPUT_PKL = BASE_DIR / "outputs" / "matrices" / "activation_embeddings.pkl"

# Map filenames to Clean Model Keys (matching behavioral naming convention)
FILENAME_TO_KEY = {
    "gemma3_27b_embeddings.csv":      "gemma-3-27b-instruct",
    "gpt_oss_20b_embeddings.csv":     "gpt-oss-20b-instruct",
    "mistral_s_24b_embeddings.csv":   "mistral-small-24b-instruct",
    "qwen3_32b_embeddings.csv":       "qwen-3-32b-instruct"
}

def load_master_mapping():
    if not MASTER_PKL.exists():
        raise FileNotFoundError(f"Master embeddings not found at {MASTER_PKL}")
    with open(MASTER_PKL, 'rb') as f:
        data = pickle.load(f)
    return data['mappings']['cue_to_idx']

def process_file(filepath, cue_to_idx):
    print(f"\nProcessing {filepath.name}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  [Error] Failed to read CSV: {e}")
        return None
    
    # 1. Identify Cue Column (First column)
    cols = list(df.columns)
    cue_col = cols[0]
    df[cue_col] = df[cue_col].astype(str).str.lower().str.strip()
    
    # 2. Determine Dimensionality & Format
    sample_row = df.iloc[0]
    
    # Check if string-encoded "[0.1, 0.2]"
    is_string_vec = False
    if len(cols) > 1 and isinstance(sample_row[cols[1]], str) and str(sample_row[cols[1]]).strip().startswith("["):
        print("  > Type: String-encoded list")
        is_string_vec = True
        parsed = ast.literal_eval(sample_row[cols[1]])
        dim = len(parsed)
    else:
        print("  > Type: Column-wise features")
        dim = len(cols) - 1
        
    print(f"  > Hidden Dimension: {dim}")
    
    # 3. Initialize Matrix with NaNs (Crucial for Strict Intersection)
    n_cues = len(cue_to_idx)
    matrix = np.full((n_cues, dim), np.nan, dtype=np.float32)
    
    # 4. Fill Matrix
    print("  > Aligning data...")
    if is_string_vec:
        df['vec'] = df[cols[1]].apply(lambda x: np.array(ast.literal_eval(str(x)), dtype=np.float32))
        data_map = dict(zip(df[cue_col], df['vec']))
    else:
        vec_data = df.iloc[:, 1:].values.astype(np.float32)
        data_map = dict(zip(df[cue_col], vec_data))
    
    hit_count = 0
    for cue, idx in cue_to_idx.items():
        if cue in data_map:
            vec = data_map[cue]
            if vec.shape[0] == dim:
                matrix[idx] = vec
                hit_count += 1
                
    print(f"  > Coverage: {hit_count}/{n_cues} ({hit_count/n_cues:.1%})")
    return matrix

def main():
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}")
        return

    cue_to_idx = load_master_mapping()
    final_embeddings = {}
    
    for filename, key_base in FILENAME_TO_KEY.items():
        fp = INPUT_DIR / filename
        if fp.exists():
            # Store with 'activation_' prefix
            final_embeddings[f"activation_{key_base}"] = process_file(fp, cue_to_idx)
        else:
            print(f"⚠️  File not found: {filename}")
            
    # Save
    if final_embeddings:
        OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
        payload = {"embeddings": final_embeddings, "mappings": {"cue_to_idx": cue_to_idx}}
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(payload, f)
        print(f"\n✅ Saved {len(final_embeddings)} matrices to {OUTPUT_PKL}")

if __name__ == "__main__":
    main()