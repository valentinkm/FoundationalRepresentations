"""
src/debug_preprocessing.py

Replicates the data preprocessing logic from src/evaluation/predict.py
to investigate exactly what data is being kept/dropped.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_human_swow(human_csv_path: Path, min_freq: int = 5) -> set:
    """
    Load SWOW to define the 'Canonical Vocabulary' (Simplified version from vectorize.py).
    Returns: set of valid response words.
    """
    print(f"\n[SWOW] Loading from {human_csv_path}...")
    df = pd.read_csv(human_csv_path)
    
    # 1. Melt
    base = df[["cue", "R1", "R2", "R3"]].copy()
    base["cue"] = base["cue"].astype(str).str.lower().str.strip()
    long = base.melt(id_vars=["cue"], value_vars=["R1", "R2", "R3"], 
                     var_name="slot", value_name="response_word")
    
    # 2. Clean
    long = long.dropna(subset=["response_word"])
    long["response_word"] = long["response_word"].astype(str).str.lower().str.strip()
    long = long[long["response_word"] != ""]
    
    # 3. Filter
    word_counts = long["response_word"].value_counts()
    valid_words = set(word_counts[word_counts >= min_freq].index)
    
    print(f"[SWOW] Vocabulary Size (Min Freq {min_freq}): {len(valid_words)}")
    return valid_words

def investigate_norms(norms_path: Path, vocab_set: set):
    print(f"\n[Norms] Loading from {norms_path}...")
    
    # 1. Load Raw
    df_raw = pd.read_csv(norms_path, low_memory=False)
    print(f"[Norms] Raw Rows: {len(df_raw)}")
    
    # 2. Clean (Replicating predict.py)
    df = df_raw.copy()
    df['word'] = df['word'].astype(str).str.lower().str.strip()
    
    # Check for non-numeric ratings BEFORE coercion
    non_numeric_mask = pd.to_numeric(df['human_rating'], errors='coerce').isna()
    bad_rows = df[non_numeric_mask]
    
    if not bad_rows.empty:
        print(f"\n[Norms] ⚠️ Found {len(bad_rows)} non-numeric rows.")
        print("  > Sample Bad Rows:")
        print(bad_rows[['norm_name', 'word', 'human_rating']].head(10))
        print("  > Unique 'human_rating' values in bad rows (top 20):")
        print(bad_rows['human_rating'].value_counts().head(20))
    
    # Coerce
    df['human_rating'] = pd.to_numeric(df['human_rating'], errors='coerce')
    df = df.dropna(subset=['human_rating'])
    print(f"[Norms] Cleaned Rows: {len(df)}")
    
    # 3. Analyze Norms
    unique_norms = df['norm_name'].unique()
    print(f"\n[Norms] Found {len(unique_norms)} unique norms (tasks).")
    
    norm_counts = df['norm_name'].value_counts()
    print("\n[Norms] Top 10 Norms by Word Count:")
    print(norm_counts.head(10))
    print("\n[Norms] Bottom 10 Norms by Word Count:")
    print(norm_counts.tail(10))
    
    # 4. Analyze Overlap with SWOW
    print("\n[Overlap] Checking overlap with SWOW Vocabulary...")
    
    # We check overlap for a few sample norms
    sample_norms = norm_counts.head(3).index.tolist() + norm_counts.tail(3).index.tolist()
    
    for norm in sample_norms:
        sub_df = df[df['norm_name'] == norm]
        norm_words = set(sub_df['word'].unique())
        
        overlap = norm_words.intersection(vocab_set)
        missing = norm_words - vocab_set
        
        print(f"\n  Task: {norm}")
        print(f"    > Total Words in Norm: {len(norm_words)}")
        print(f"    > Overlap with SWOW: {len(overlap)} ({len(overlap)/len(norm_words):.1%})")
        if missing:
            print(f"    > Sample Missing Words: {list(missing)[:5]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--swow_path', type=Path, required=True)
    parser.add_argument('--norms_path', type=Path, required=True)
    args = parser.parse_args()
    
    vocab_set = load_human_swow(args.swow_path)
    investigate_norms(args.norms_path, vocab_set)

if __name__ == "__main__":
    main()
