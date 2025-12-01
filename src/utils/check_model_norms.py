"""
src/utils/deep_dive_diagnostics.py

Investigates WHY specific models (like GPT-OSS) perform poorly.
1. Checks intersection size between Embeddings (SWOW Cues) and Model Norms.
2. Qualitatively inspects the ratings for specific norms to detect hallucinations.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"
NORMS_DIR = BASE_DIR / "outputs" / "raw_behavior" / "model_norms"

# Which norm to inspect qualitatively?
TARGET_NORM = "concreteness_brysbaert" # Expectation: Objects > Ideas

def load_master_cues():
    """Get the list of cues that form the rows of our X matrix."""
    print(f"Loading Master Vocabulary from {EMBEDDINGS_PATH.name}...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # The mapping dict has 'cue_to_idx'
    cues = set(data['mappings']['cue_to_idx'].keys())
    print(f"Pipeline Vocabulary Size: {len(cues)}")
    return cues

def inspect_models(master_cues):
    files = sorted(list(NORMS_DIR.glob("*.csv")))
    
    print(f"\n{'MODEL':<30} | {'TOTAL ROWS':<10} | {'OVERLAP (X ∩ y)':<15} | {'% COVERAGE':<10}")
    print("-" * 75)
    
    results = {}

    for fp in files:
        try:
            # Load and clean
            df = pd.read_csv(fp, low_memory=False)
            if 'cleaned_rating' not in df.columns: continue
            
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            
            # 1. Intersection Check
            model_vocab = set(df['word'].unique())
            overlap = master_cues.intersection(model_vocab)
            
            n_overlap = len(overlap)
            pct_coverage = (n_overlap / len(master_cues)) * 100
            
            print(f"{fp.stem:<30} | {len(df):<10} | {n_overlap:<15} | {pct_coverage:>8.1f}%")
            
            # Store for qualitative check
            results[fp.stem] = df
            
        except Exception:
            pass

    return results

def qualitative_check(model_data):
    print(f"\n{'='*30} QUALITATIVE CHECK: {TARGET_NORM} {'='*30}")
    print("Do the models understand what 'Concreteness' means?")
    
    for model_name, df in model_data.items():
        # Filter for the target norm
        # Handle flexible column names 'norm' or 'norm_name'
        norm_col = 'norm' if 'norm' in df.columns else 'norm_name'
        
        # Fuzzy match the norm name
        subset = df[df[norm_col].astype(str).str.contains("concreteness", case=False, na=False)]
        
        if subset.empty:
            continue
            
        print(f"\n--- {model_name} ---")
        
        # Sort by rating
        subset = subset.sort_values("cleaned_rating", ascending=False)
        
        top_words = subset.head(5)[['word', 'cleaned_rating']].values.tolist()
        bottom_words = subset.tail(5)[['word', 'cleaned_rating']].values.tolist()
        
        print(f"  Top (Concrete?): {top_words}")
        print(f"  Bot (Abstract?): {bottom_words}")

if __name__ == "__main__":
    cues = load_master_cues()
    data = inspect_models(cues)
    qualitative_check(data)