"""
src/evaluation/predict_banded_ridge.py

Banded Ridge Regression (Dual Form) for Variance Partitioning.
Groups:
    - Band A: Behavioral Embeddings (300d)
    - Band B: Activation Embeddings (High-Dim)

Method:
    1. Z-Score features independently.
    2. Compute Linear Kernels K_A, K_B.
    3. Grid Search over regularization parameters (lambda_A, lambda_B).
    4. Solve Dual System: alpha = (K_eff + I)^-1 * y
    5. Evaluate via 5-Fold Cross-Validation.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import scipy.linalg

# --- CONFIG ---
LAMBDA_GRID = np.logspace(-3, 5, 9)  # 1e-3 to 1e5
CV_FOLDS = 5
MIN_SAMPLES = 50

def load_data(pkl_path: Path, norms_dir: Path, allowed_models: list = None):
    print(f"[Loader] Loading embeddings from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    mappings = data['mappings']
    cue_to_idx = mappings['cue_to_idx']

    # Load Norms
    norm_files = sorted(list(norms_dir.glob('*.csv')))
    model_norms = {}
    for fp in norm_files:
        if allowed_models and not any(m in fp.stem for m in allowed_models):
            continue
        try:
            df = pd.read_csv(fp)
            # Basic clean
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df = df.dropna(subset=['cleaned_rating', 'norm'])
            df['norm'] = df['norm'].astype(str).str.strip()
            model_norms[fp.stem] = df
        except Exception as e:
            print(f"[Loader] Failed to load {fp}: {e}")

    return embeddings, cue_to_idx, model_norms

 

def solve_banded_ridge_cv(X_A, X_B, y, random_state=42):
    """
    Full CV Pipeline with proper standardization.
    Now with assume_a='pos' for speed.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    
    # Store scores for each (la, lb) pair
    grid_scores = { (la, lb): [] for la in LAMBDA_GRID for lb in LAMBDA_GRID }
    
    for train_idx, test_idx in kf.split(X_A):
        # 1. Split
        X_A_train, X_A_test = X_A[train_idx], X_A[test_idx]
        X_B_train, X_B_test = X_B[train_idx], X_B[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        
        # 2. Standardize (Z-score) - Independently per band to prevent leakage
        scaler_A = StandardScaler()
        X_A_train = scaler_A.fit_transform(X_A_train)
        X_A_test = scaler_A.transform(X_A_test)
        
        scaler_B = StandardScaler()
        X_B_train = scaler_B.fit_transform(X_B_train)
        X_B_test = scaler_B.transform(X_B_test)
        
        # 3. Compute Kernels (Train & Test-Train)
        # Train Kernel: (N_train x N_train)
        K_A_train = X_A_train @ X_A_train.T
        K_B_train = X_B_train @ X_B_train.T
        
        # Test Kernel: (N_test x N_train) - For prediction
        K_A_test = X_A_test @ X_A_train.T
        K_B_test = X_B_test @ X_B_train.T
        
        # 4. Grid Search
        I = np.eye(len(y_train))
        
        # Optimization: Reuse Partial K_A for all K_B
        for la in LAMBDA_GRID:
            # Scaled Kernel A
            partial_A = K_A_train / la
            partial_A_test = K_A_test / la
            
            for lb in LAMBDA_GRID:
                # Effective Kernel
                K_train_eff = partial_A + (K_B_train / lb)
                K_train_eff.flat[::len(y_train)+1] += 1.0  # Add Identity
                
                # Solve Dual
                # Using assume_a='pos' since Kernel Matrices are Positive Definite
                try:
                    alpha = scipy.linalg.solve(K_train_eff, y_train, assume_a='pos')
                except Exception:
                    # Fallback to general solver if Cholesky fails (numerical instability)
                    alpha = scipy.linalg.solve(K_train_eff, y_train)
                
                # Predict
                K_test_eff = partial_A_test + (K_B_test / lb)
                y_pred = K_test_eff @ alpha
                
                # Score
                ss_res = np.sum((y_test - y_pred)**2)
                ss_tot = np.sum((y_test - np.mean(y_test))**2)
                r2 = 1 - (ss_res / ss_tot)
                
                grid_scores[(la, lb)].append(r2)

    # Aggregate
    avg_scores = {k: np.mean(v) for k, v in grid_scores.items()}
    best_params = max(avg_scores, key=avg_scores.get)
    best_r2 = avg_scores[best_params]
    
    return best_r2, best_params[0], best_params[1]

def process_single_norm(model_name, norm_name, df_norm, X_A_full, X_B_full, cue_to_idx, verbose=False):
    # Align
    valid_cues = df_norm[df_norm['norm'] == norm_name]
    if valid_cues.empty: return None
    
    # Overlap
    available = set(valid_cues['word'])
    vocab = set(cue_to_idx.keys())
    overlap = sorted(list(vocab.intersection(available)))
    
    if len(overlap) < MIN_SAMPLES:
        if verbose: print(f"  [Skip] {norm_name}: Insufficient overlap ({len(overlap)})")
        return None
        
    idxs = [cue_to_idx[w] for w in overlap]
    row_map = valid_cues.groupby('word')['cleaned_rating'].mean().to_dict()
    y = np.array([row_map[w] for w in overlap])
    
    # Slice Embeddings (Real)
    # Handle Sparse
    if hasattr(X_A_full, "toarray"): X_A = X_A_full[idxs].toarray()
    else: X_A = X_A_full[idxs]
    
    if hasattr(X_B_full, "toarray"): X_B = X_B_full[idxs].toarray()
    else: X_B = X_B_full[idxs]
    
    # 1. Run Real Banded Ridge
    r2_joint, la, lb = solve_banded_ridge_cv(X_A, X_B, y)
    
    # 2. Run Null Baseline
    # Band A (Behavior) + Band B (Null Noise)
    # Ensure Null Noise has same dimensionality as Band B
    n_samples, n_features_B = X_B.shape
    X_B_null = np.random.randn(n_samples, n_features_B)
    
    # We could rerun everything, but we can reuse X_A processing if we refactored solver...
    # For now, just call the solver again with (X_A, X_B_null)
    r2_null, _, _ = solve_banded_ridge_cv(X_A, X_B_null, y, random_state=None) # Random seed for noise? X_B_null is already random.
    
    return {
        'model': model_name,
        'norm': norm_name,
        'r2_joint': r2_joint,
        'r2_null': r2_null,
        'delta_r2': r2_joint - r2_null,
        'best_lambda_A': la,
        'best_lambda_B': lb,
        'n_samples': len(overlap)
    }

def main():
    parser = argparse.ArgumentParser(description="Banded Ridge Regression (Behavior vs Activation)")
    parser.add_argument('--embeddings_path', type=Path, required=True)
    parser.add_argument('--norms_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test_limit', type=int, default=0, help="Limit number of norms per model (for testing)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    embeddings, cue_to_idx, model_norms = load_data(args.embeddings_path, args.norms_dir, args.models)
    
    results = []
    
    # Iterate Models
    for model_name, norms_df in model_norms.items():
        # Match embedding keys for Band A (Behavior) and Band B (Activation)
        key_A = None
        key_B = None
        
        for k in embeddings.keys():
            if f"passive_{model_name}_300d" == k: key_A = k
            # Activation might be named 'activation_model-name'
            if f"activation_{model_name}" == k: key_B = k
            
        # Fallback substring match if exact fail
        if not key_A:
             cands = [k for k in embeddings.keys() if f"passive_" in k and "_300d" in k and model_name in k]
             if cands: key_A = cands[0]
        if not key_B:
             cands = [k for k in embeddings.keys() if f"activation_" in k and model_name in k]
             if cands: key_B = cands[0]
             
        if not key_A or not key_B:
            print(f"[Skip] {model_name}: Could not find paired embeddings (A={key_A}, B={key_B})")
            continue
            
        print(f"\n[Model] {model_name}")
        print(f"  > Band A (Behavior): {key_A}")
        print(f"  > Band B (Activation): {key_B}")
        
        X_A_full = embeddings[key_A]
        X_B_full = embeddings[key_B]
        
        unique_norms = sorted(norms_df['norm'].unique())
        
        if args.test_limit and args.test_limit > 0:
            unique_norms = unique_norms[:args.test_limit]
            print(f"  [Test Mode] Limited to first {len(unique_norms)} norms.")
            
        # Parallel Eval
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(process_single_norm)(
                model_name, norm, norms_df, X_A_full, X_B_full, cue_to_idx, args.verbose
            ) for norm in unique_norms
        )
        
        results.extend([r for r in res if r])
        
    # Save
    out_file = args.output_dir / "banded_ridge_results.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n[Done] Saved results to {out_file}")

if __name__ == "__main__":
    main()
