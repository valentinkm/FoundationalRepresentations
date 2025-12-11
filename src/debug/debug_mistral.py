"""
src/debug_mistral.py

Deep dive into why activation_mistral-small-24b-instruct is failing.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold

EMBEDDINGS_PATH = Path("outputs/matrices/embeddings.pkl")
NORMS_DIR = Path("outputs/raw_behavior/model_norms")
MODEL_NAME = "mistral-small-24b-instruct"
KEY_ACT = f"activation_{MODEL_NAME}"
KEY_PAS = f"passive_{MODEL_NAME}"

def main():
    print(f"Loading {EMBEDDINGS_PATH}...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict) and 'embeddings' in data:
            embeddings = data['embeddings']
            mappings = data['mappings']
        else:
            embeddings = data
            mappings = data.get('mappings')
            
    cue_to_idx = mappings['cue_to_idx']
    idx_to_cue = {v: k for k, v in cue_to_idx.items()}
    
    if KEY_ACT not in embeddings:
        print(f"Key {KEY_ACT} not found!")
        return
        
    X_act = embeddings[KEY_ACT]
    X_pas = embeddings[KEY_PAS]
    
    print(f"\n--- Analysis: {KEY_ACT} ---")
    print(f"Shape: {X_act.shape}")
    
    # 1. Zero Vector Check
    norms = np.linalg.norm(X_act, axis=1)
    valid_mask = norms > 0
    n_valid = np.sum(valid_mask)
    print(f"Valid Vectors: {n_valid}/{X_act.shape[0]} ({n_valid/X_act.shape[0]:.1%})")
    
    if n_valid == 0:
        print("ALL VECTORS ARE ZERO.")
        return

    # 2. Collapsed Space Check (on valid vectors)
    X_valid = X_act[valid_mask]
    print(f"Valid Shape: {X_valid.shape}")
    
    # Sample 100
    indices = np.random.choice(X_valid.shape[0], min(100, X_valid.shape[0]), replace=False)
    sample = X_valid[indices]
    sim = cosine_similarity(sample)
    mean_sim = np.mean(sim[np.triu_indices(len(sample), k=1)])
    print(f"Mean Cosine Similarity (Valid Sample): {mean_sim:.4f}")
    
    if mean_sim > 0.9:
        print("CRITICAL: Space is collapsed! All vectors are nearly identical.")
        
    # 3. Compare with Passive
    print(f"\n--- Comparison: {KEY_PAS} ---")
    norms_pas = np.linalg.norm(X_pas, axis=1)
    valid_pas = norms_pas > 0
    print(f"Valid Vectors: {np.sum(valid_pas)}/{X_pas.shape[0]}")
    
    # 4. Regression Test (Single Norm)
    # Load Mistral Norms
    norm_file = NORMS_DIR / f"{MODEL_NAME}.csv"
    if not norm_file.exists():
        print("Norm file not found.")
        return
        
    df = pd.read_csv(norm_file)
    # Pick a "good" norm (not skewed)
    # Based on audit: 'concreteness_brysbaert' usually works? Or 'valence_warriner'?
    # Let's try 'concreteness_brysbaert' if it exists, or just the first one.
    target_norm = 'concreteness_brysbaert'
    sub = df[df['norm'] == target_norm]
    if sub.empty:
        target_norm = df['norm'].unique()[0]
        sub = df[df['norm'] == target_norm]
        
    print(f"\n--- Regression Test: {target_norm} ---")
    
    # Align
    valid_cues = set(cue_to_idx.keys())
    available = set(sub['word'].unique())
    overlap = list(valid_cues.intersection(available))
    
    # Filter overlap for valid vectors ONLY
    overlap = [w for w in overlap if valid_mask[cue_to_idx[w]]]
    print(f"Overlap (Valid Vectors): {len(overlap)}")
    
    if len(overlap) < 10:
        print("Not enough overlap.")
        return
        
    # Build X, y
    row_idx = [cue_to_idx[w] for w in overlap]
    X = X_act[row_idx]
    
    # Coerce to numeric
    sub['cleaned_rating'] = pd.to_numeric(sub['cleaned_rating'], errors='coerce')
    sub = sub.dropna(subset=['cleaned_rating'])
    
    y_map = sub.groupby('word')['cleaned_rating'].mean().to_dict()
    y = np.array([y_map[w] for w in overlap])
    
    # Run Ridge
    model = RidgeCV()
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"R^2 Score (Activation): {score:.4f}")
    
    # Run Passive
    X_p = X_pas[row_idx]
    score_p = cross_val_score(model, X_p, y, cv=5).mean()
    print(f"R^2 Score (Passive):    {score_p:.4f}")

if __name__ == "__main__":
    main()
