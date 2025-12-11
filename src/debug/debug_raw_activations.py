"""
src/debug_raw_activations.py

Investigates if missing words in the activation matrix are present in the raw CSVs.
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

EMBEDDINGS_PATH = Path("outputs/matrices/embeddings.pkl")
RAW_DIR = Path("outputs/raw_activations")

def main():
    # 1. Load Canonical Cues
    print(f"[Loader] Loading mappings from {EMBEDDINGS_PATH}...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict) and 'mappings' in data.get('embeddings', {}): # Handle nested if needed, but usually mappings is top level or in embeddings
             # Wait, inspect_pickle said: Keys: dict_keys(['embeddings', 'mappings'])
             mappings = data['mappings']
        elif 'mappings' in data:
             mappings = data['mappings']
        else:
             print("Could not find mappings.")
             return

    cue_to_idx = mappings['cue_to_idx']
    canonical_cues = set(cue_to_idx.keys())
    print(f"[Vocab] Canonical Cues: {len(canonical_cues)}")
    
    # 2. Check a sample CSV
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("No CSVs found.")
        return
        
    target_file = csv_files[0] # Pick first one
    print(f"\n[Audit] Checking file: {target_file}")
    
    df = pd.read_csv(target_file)
    print(f"[CSV] Rows: {len(df)}")
    print(f"[CSV] Columns: {list(df.columns)[:5]}...")
    
    # Identify word column
    col_name = 'word' if 'word' in df.columns else 'cue'
    if col_name not in df.columns:
        if 'Unnamed: 0' in df.columns:
            col_name = 'Unnamed: 0'
        else:
            print("Could not identify word column.")
            return
            
    print(f"[CSV] Using word column: '{col_name}'")
    
    # 3. Analyze Coverage
    raw_words = set(df[col_name].astype(str))
    
    # Exact Match
    overlap = raw_words.intersection(canonical_cues)
    print(f"[Match] Exact Overlap: {len(overlap)}/{len(canonical_cues)} ({len(overlap)/len(canonical_cues):.1%})")
    
    # Case Insensitive
    raw_lower = {w.lower().strip() for w in raw_words}
    overlap_lower = raw_lower.intersection(canonical_cues) # canonical are already lower/stripped
    print(f"[Match] Lowercase/Strip Overlap: {len(overlap_lower)}/{len(canonical_cues)} ({len(overlap_lower)/len(canonical_cues):.1%})")
    
    # Missing
    missing = canonical_cues - overlap_lower
    print(f"[Missing] {len(missing)} words are missing from the CSV.")
    print(f"  > Sample Missing: {list(missing)[:10]}")
    
    # Check if missing words are in the CSV but weirdly formatted
    # e.g. "apple" vs "apple "
    
if __name__ == "__main__":
    main()
