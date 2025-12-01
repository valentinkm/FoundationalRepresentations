"""
src/activations/inspect_raw_activations.py

Diagnostic script to check the format and coverage of external activation embeddings.
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import ast

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
ACTIVATION_DIR = BASE_DIR / "outputs" / "raw_activations"
BEHAVIOR_PKL = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"

def get_master_vocab():
    print(f"Loading Master Vocab from {BEHAVIOR_PKL.name}...")
    with open(BEHAVIOR_PKL, 'rb') as f:
        data = pickle.load(f)
    return set(data['mappings']['cue_to_idx'].keys())

def inspect_file(filepath, master_vocab):
    print(f"\n{'='*60}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*60}")
    
    try:
        # Read first few rows to infer structure
        df_head = pd.read_csv(filepath, nrows=5)
        print(f"Columns found: {list(df_head.columns)}")
        
        # Determine Cue Column
        cue_col = None
        for cand in ['word', 'cue', 'term', 'label']:
            if cand in df_head.columns:
                cue_col = cand
                break
        
        if not cue_col:
            # Fallback: Assume first column is cue if string
            if df_head.iloc[:, 0].dtype == 'O':
                cue_col = df_head.columns[0]
                print(f"⚠️  No standard header found. Assuming first column '{cue_col}' is the cue.")
            else:
                print("❌ Could not identify cue column.")
                return

        # Determine Vector Format
        # Case A: One column named 'embedding' or 'vector' containing string "[0.1, 0.2]"
        # Case B: Columns dim_0, dim_1...
        
        vector_cols = [c for c in df_head.columns if c != cue_col]
        sample_val = df_head.iloc[0][vector_cols[0]]
        
        is_string_vector = False
        dim = 0
        
        if isinstance(sample_val, str) and sample_val.strip().startswith("["):
            print("Type: String-encoded vectors (e.g., '[0.1, 0.2]')")
            is_string_vector = True
            # Check dim
            parsed = ast.literal_eval(sample_val)
            dim = len(parsed)
        else:
            print("Type: Column-wise vectors")
            dim = len(vector_cols)
            
        print(f"Estimated Dimensions: {dim}")

        # --- FULL LOAD FOR COVERAGE ---
        print("Reading full file for coverage check...")
        df = pd.read_csv(filepath)
        
        # Standardize Cues
        df[cue_col] = df[cue_col].astype(str).str.lower().str.strip()
        found_vocab = set(df[cue_col].unique())
        
        # Intersection
        overlap = master_vocab.intersection(found_vocab)
        missing = master_vocab - found_vocab
        
        print(f"Total Rows: {len(df)}")
        print(f"Master Vocab Coverage: {len(overlap)} / {len(master_vocab)} ({len(overlap)/len(master_vocab):.1%})")
        
        if len(missing) > 0:
            print(f"Missing Examples: {list(missing)[:5]}")
            
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

if __name__ == "__main__":
    if not ACTIVATION_DIR.exists():
        print(f"Directory not found: {ACTIVATION_DIR}")
    else:
        vocab = get_master_vocab()
        files = sorted(list(ACTIVATION_DIR.glob("*.csv")))
        for f in files:
            inspect_file(f, vocab)