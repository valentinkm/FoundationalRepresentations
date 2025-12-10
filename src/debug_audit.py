import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --- PATHS (Adjust if necessary) ---
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / 'outputs' / 'matrices' / 'embeddings.pkl'
NORMS_DIR = PROJECT_ROOT / 'outputs' / 'raw_behavior' / 'model_norms'

# List problem models specifically
TARGET_MODELS = ['mistral-small-24b-instruct', 'gemma-3-27b-instruct'] 

def load_data():
    print(f"Loading pickle from {EMBEDDINGS_PATH}...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['mappings']

def audit_activations(embeddings, cue_to_idx, idx_to_cue):
    print("\n=== 1. ACTIVATION VECTOR AUDIT ===")
    
    for model in TARGET_MODELS:
        key = f"activation_{model}"
        if key not in embeddings:
            print(f"MISSING: {key}")
            continue
            
        mat = embeddings[key]
        print(f"\nModel: {model}")
        print(f"  Shape: {mat.shape}")
        
        # Check for pathology
        norms = np.linalg.norm(mat, axis=1)
        zero_vecs = np.sum(norms == 0)
        nans = np.isnan(mat).sum()
        
        print(f"  Zero Vectors: {zero_vecs} / {mat.shape[0]}")
        print(f"  NaN Values:   {nans}")
        print(f"  Mean Vector Norm: {np.mean(norms):.4f} (std: {np.std(norms):.4f})")
        
        if np.std(norms) < 1e-5:
            print("  [CRITICAL] Vector norms have zero variance. Did you extract the BOS token?")

        # Check Cosine Similarity of random pair to see if space is collapsed
        # If all vectors are nearly identical, similarity will be ~1.0
        sample_idx = np.random.choice(mat.shape[0], 100, replace=False)
        sim_matrix = cosine_similarity(mat[sample_idx])
        # Mask diagonal
        np.fill_diagonal(sim_matrix, np.nan)
        avg_sim = np.nanmean(sim_matrix)
        print(f"  Avg Cosine Sim (Random Sample): {avg_sim:.4f}")
        
        if avg_sim > 0.95:
            print("  [CRITICAL] Space Collapsed: All vectors are nearly identical.")

def audit_norms(norms_dir):
    print("\n=== 2. TARGET (NORM) AUDIT ===")
    
    for model in TARGET_MODELS:
        # Find file
        candidates = list(norms_dir.glob(f"*{model}*.csv"))
        if not candidates:
            print(f"No norms found for {model}")
            continue
        
        fp = candidates[0]
        print(f"\nModel: {model} ({fp.name})")
        df = pd.read_csv(fp)
        
        # Check specific norm used in head output
        if 'cleaned_rating' not in df.columns:
            print("  Missing 'cleaned_rating' column")
            continue
            
        df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
        
        # Group by norm type
        for norm_name, gdf in df.groupby('norm'):
            ratings = gdf['cleaned_rating'].dropna()
            mean = ratings.mean()
            std = ratings.std()
            
            print(f"  Norm: {norm_name}")
            print(f"    N={len(ratings)}, Mean={mean:.2f}, Std={std:.4f}")
            print(f"    Value Counts: {dict(ratings.value_counts().sort_index())}")
            
            if std < 0.1:
                print(f"    [CRITICAL] Low Variance! Model predicted same value for everything.")

def check_alignment(embeddings, mappings, norms_dir):
    print("\n=== 3. ALIGNMENT SMOKE TEST ===")
    cue_to_idx = mappings['cue_to_idx']
    
    # We will try to correlate vector magnitude with concreteness as a rough heuristic
    # (Concrete words often have different magnitudes in some spaces, or just checking random alignment)
    
    for model in TARGET_MODELS:
        act_key = f"activation_{model}"
        if act_key not in embeddings: continue
        
        vecs = embeddings[act_key]
        
        # Load Norms
        candidates = list(norms_dir.glob(f"*{model}*.csv"))
        if not candidates: continue
        df = pd.read_csv(candidates[0])
        df = df[df['norm'] == 'concreteness_brysbaert'] # Pick one norm
        
        common_words = list(set(cue_to_idx.keys()) & set(df['word'].str.lower().str.strip()))
        if len(common_words) < 10: continue
        
        sample_words = common_words[:5]
        print(f"\nChecking Alignment for {model} (Sample of 5):")
        
        for w in sample_words:
            idx = cue_to_idx[w]
            vec = vecs[idx]
            rating = df[df['word'] == w]['cleaned_rating'].values[0]
            print(f"  Word: '{w}' | Norm: {rating} | Vec[:3]: {vec[:3]} | VecNorm: {np.linalg.norm(vec):.2f}")

if __name__ == "__main__":
    emb, map_ = load_data()
    audit_activations(emb, map_['cue_to_idx'], map_['idx_to_cue'])
    audit_norms(NORMS_DIR)
    check_alignment(emb, map_, NORMS_DIR)