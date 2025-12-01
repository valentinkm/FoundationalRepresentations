"""
src/utils/diagnose_data_loss.py

Data Forensic Script:
Distinguishes between MISSING data (script didn't run) and FAILED data (model output garbage).

Focuses on the intersection with the Master SWOW Vocabulary.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"
NORMS_DIR = BASE_DIR / "outputs" / "raw_behavior" / "model_norms"

# The suspect model
TARGET_MODEL_FILE = "gpt-oss-20b-instruct.csv" 

def get_master_vocab():
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    return set(data['mappings']['cue_to_idx'].keys())

def diagnose_model(vocab):
    csv_path = NORMS_DIR / TARGET_MODEL_FILE
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return

    print(f"🔍 DIAGNOSING: {TARGET_MODEL_FILE}")
    print(f"Master Vocabulary Size: {len(vocab)}")
    
    # Load Raw
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Normalize
    norm_col = 'norm' if 'norm' in df.columns else 'norm_name'
    df['word'] = df['word'].astype(str).str.lower().str.strip()
    
    # Check distinct norms found in the file
    found_norms = df[norm_col].unique()
    print(f"\nFound {len(found_norms)} norms in the file.")
    
    # Analyze detailed breakdown per norm
    print(f"\n{'NORM NAME':<35} | {'PRESENT':<8} | {'MISSING':<8} | {'FAILED (NaN)':<12} | {'SUCCESS':<8}")
    print("-" * 90)
    
    aggregation = {
        "missing_count": 0,
        "failed_count": 0,
        "failed_raw_examples": []
    }
    
    for norm in found_norms:
        # Get rows for this norm
        sub = df[df[norm_col] == norm]
        
        # 1. Words present in CSV
        present_words = set(sub['word'].unique())
        
        # 2. Words completely missing (Execution Error)
        missing_words = vocab - present_words
        n_missing = len(missing_words)
        
        # 3. Words present but failed (Competence Error)
        # Check if cleaned_rating is numeric
        sub['is_valid'] = pd.to_numeric(sub['cleaned_rating'], errors='coerce').notna()
        n_failed = (~sub['is_valid']).sum()
        n_success = sub['is_valid'].sum()
        
        # Collect raw failures for diagnosis
        if n_failed > 0:
            failures = sub[~sub['is_valid']]['raw_response'].head(3).tolist()
            aggregation["failed_raw_examples"].extend([(norm, f) for f in failures])
            
        aggregation["missing_count"] += n_missing
        aggregation["failed_count"] += n_failed
        
        print(f"{norm:<35} | {len(sub):<8} | {n_missing:<8} | {n_failed:<12} | {n_success:<8}")

    print("-" * 90)
    
    # --- CONCLUSION ---
    print("\n--- FORENSIC CONCLUSION ---")
    
    total_expected = len(vocab) * len(found_norms)
    total_present = len(df) # Roughly
    
    print(f"1. EXECUTION STATUS:")
    if aggregation['missing_count'] > (total_expected * 0.5):
        print(f"   🔴 CRITICAL: Huge chunks of data are missing. The script likely crashed or stopped early.")
        print(f"   (You expected ~{len(vocab)} rows per norm, but got ~{int(total_present/len(found_norms))})")
    else:
        print(f"   ✅ Data presence looks healthy.")

    print(f"\n2. COMPETENCE STATUS:")
    if aggregation['failed_count'] > 0:
        print(f"   ⚠️  Found {aggregation['failed_count']} rows where the model failed to generate a number.")
        print(f"   Sample Raw Responses (What the model actually said):")
        for i, (n, r) in enumerate(aggregation['failed_raw_examples'][:10]):
            print(f"     [{n}]: {r}")
    else:
        print("   ✅ All present rows were successfully parsed into numbers.")

if __name__ == "__main__":
    vocab = get_master_vocab()
    diagnose_model(vocab)