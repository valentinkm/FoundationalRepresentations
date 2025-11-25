"""
src/evaluation/predict_human.py

Evaluates how well different semantic representations predict psycholinguistic norms.

Method:
- Ridge Regression (L2 Regularization) with Cross-Validation.
- Metric: R^2 (Variance Explained).
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold

# --- CONFIG ---
DEFAULT_ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
CV_FOLDS = 5
MIN_SAMPLES = 50  # Minimum overlapping words to attempt regression

def load_embeddings(pkl_path: Path):
    """Load the dictionary of matrices and mappings."""
    print(f"[Loader] Loading embeddings from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['mappings']

def load_human_norms(csv_path: Path):
    """Load Ground Truth Human Norms with strict cleaning."""
    print(f"[Loader] Loading human norms from {csv_path}...")
    # Low_memory=False prevents warning on mixed types before we clean them
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 1. Standardize text keys
    df['word'] = df['word'].astype(str).str.lower().str.strip()
    
    # 2. STRICT CLEANING: Coerce 'human_rating' to numeric
    # Errors (like 'Olfactory') become NaN
    df['human_rating'] = pd.to_numeric(df['human_rating'], errors='coerce')
    
    # 3. Drop Bad Rows
    initial_len = len(df)
    df = df.dropna(subset=['human_rating'])
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"[Loader] ⚠️ Dropped {dropped} rows containing non-numeric ratings (e.g. headers mixed in data).")
        
    return df

def align_data(embedding_mat, cue_to_idx, norm_df, norm_name):
    """
    Intersection of:
    1. Words in the embedding matrix (Rows)
    2. Words in the norm dataset (Rows for specific norm)
    
    Returns: X (features), y (targets), overlapping_words
    """
    # Filter norm data for the specific metric (e.g., 'concreteness')
    # We already dropped NaNs in loader, but safety first
    sub_df = norm_df[norm_df['norm_name'] == norm_name].copy()
    
    if sub_df.empty:
        return None, None, None

    # Find overlap
    valid_cues = set(cue_to_idx.keys())
    available_words = set(sub_df['word'].unique())
    overlap = list(valid_cues.intersection(available_words))
    
    if len(overlap) < MIN_SAMPLES:
        return None, None, None
        
    # Build X and y
    # 1. Get indices for X
    row_indices = [cue_to_idx[w] for w in overlap]
    X = embedding_mat[row_indices]
    
    # 2. Get values for y (ensure order matches X)
    # Create a lookup for speed. Handle duplicate words by taking mean if any exist
    rating_map = sub_df.groupby('word')['human_rating'].mean().to_dict()
    y = np.array([rating_map[w] for w in overlap])
    
    return X, y, overlap

def evaluate_embedding(X, y, random_state=42):
    """
    Run Ridge Regression with Nested CV.
    Inner CV: Select Alpha.
    Outer CV: Evaluate R^2.
    """
    # Pipeline: Scale features -> Ridge
    # We use RidgeCV which handles the inner loop for Alpha selection efficiently
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=DEFAULT_ALPHAS, scoring='r2')
    )
    
    # Outer Cross-Validation for robust R^2 estimation
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean(), scores.std()
    except Exception as e:
        print(f"    [Error] Regression failed: {e}")
        return np.nan, np.nan

def run_evaluation_loop(embeddings_dict, mappings, norms_df, output_path):
    """Main loop over Embeddings x Norms."""
    results = []
    
    # 1. Identify all norms
    all_norms = sorted(norms_df['norm_name'].unique())
    print(f"[Judge] Found {len(all_norms)} norms to predict.")
    
    # 2. Identify all embeddings
    # Filter out metadata keys if any persist
    emb_keys = [k for k in embeddings_dict.keys() if k != 'mappings']
    print(f"[Judge] Found {len(emb_keys)} embedding sources.")
    
    cue_to_idx = mappings['cue_to_idx']
    
    # Progress bar logic
    total_steps = len(emb_keys) * len(all_norms)
    pbar = tqdm(total=total_steps, desc="Evaluating")
    
    for emb_name in emb_keys:
        X_full = embeddings_dict[emb_name]
        
        for norm in all_norms:
            X, y, overlap = align_data(X_full, cue_to_idx, norms_df, norm)
            
            if X is None:
                # Not enough overlap
                pbar.update(1)
                continue
                
            mean_r2, std_r2 = evaluate_embedding(X, y)
            
            if not np.isnan(mean_r2):
                results.append({
                    'embedding_source': emb_name,
                    'norm_name': norm,
                    'r2_mean': mean_r2,
                    'r2_std': std_r2,
                    'n_samples': len(overlap)
                })
            
            pbar.update(1)
            
    pbar.close()
    
    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)
    print(f"\n[Judge] Evaluation complete. Results saved to {output_path}")
    
    if not res_df.empty:
        # Print a quick summary leaderboard
        print("\n--- Leaderboard (Average R^2 across all norms) ---")
        summary = res_df.groupby('embedding_source')['r2_mean'].mean().sort_values(ascending=False)
        print(summary)
    else:
        print("\n[Judge] Warning: No valid results produced.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Representations against PsychNorms.")
    parser.add_argument('--embeddings_path', type=Path, required=True, help="Pickle from vectorize.py")
    parser.add_argument('--norms_path', type=Path, required=True, help="Path to psychNorms.csv")
    parser.add_argument('--output_dir', type=Path, required=True, help="Where to save results.csv")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    embeddings, mappings = load_embeddings(args.embeddings_path)
    norms_df = load_human_norms(args.norms_path)
    
    # Run Eval
    output_csv = args.output_dir / "evaluation_results.csv"
    run_evaluation_loop(embeddings, mappings, norms_df, output_csv)

if __name__ == "__main__":
    import sys
    # Default behavior for standard repo structure
    try:
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        
        default_emb = project_root / 'outputs' / 'matrices' / 'behavioral_embeddings.pkl'
        default_norms = project_root / 'data' / 'psych_norms' / 'psychnorms_subset_filtered_by_swow.csv'
        # Fallback if filtered subset doesn't exist yet, look for main file or utils location
        if not default_norms.exists():
             default_norms = project_root / 'data' / 'SWOW' / 'utils' / 'psychnorms_subset_filtered_by_swow.csv'
        
        default_out = project_root / 'outputs' / 'results'
    except:
        default_emb = Path('.')
        default_norms = Path('.')
        default_out = Path('.')
        
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--embeddings_path', str(default_emb),
            '--norms_path', str(default_norms),
            '--output_dir', str(default_out)
        ])
        
    main()