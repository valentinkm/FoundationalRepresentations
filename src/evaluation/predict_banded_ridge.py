"""
src/evaluation/predict_banded_ridge.py

Banded Ridge Regression (Dual Form) for Variance Partitioning.
Groups (Generalized N-Bands):
    - Band A: Passive Behavior (300d)
    - Band B: Active Behavior (300d)
    - Band C: Activation (High-Dim)

Method (Robust Nested CV):
    1. Outer CV (Evaluation): 5-Fold.
    2. Inner CV (Selection): 5-Fold, tuning N lambdas.
    3. Proper Scaling: Scalers fit on Outer Train only.
    4. Unique Value: R2_Full - R2_(Full-Band).
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
import itertools

# --- CONFIG ---
LAMBDA_GRID = np.logspace(-3, 5, 9)  # 1e-3 to 1e5
OUTER_FOLDS = 5
INNER_FOLDS = 5
MIN_SAMPLES = 50

# --- HELPERS ---

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

def make_outer_splits(n_samples, random_state=42):
    kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=random_state)
    return list(kf.split(np.zeros(n_samples)))

def make_inner_splits(n_samples, random_state=43):
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=random_state)
    return list(kf.split(np.zeros(n_samples)))

def compute_scaled_kernels(X_train, X_test):
    """
    Computes linear kernels and normalizes by trace/n_train.
    Scale factor is computed ON TRAIN ONLY.
    """
    n_train = X_train.shape[0]
    
    # Raw Kernels
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T
    
    # Scale Factor
    # Trace/N is the mean squared norm of the features (if centered)
    # Using trace of K_train
    scale = np.trace(K_train) / n_train
    
    # Avoid div by zero
    if scale == 0: scale = 1.0
        
    return K_train / scale, K_test / scale

def fit_predict_multiband(band_kernels_train, band_kernels_test, y_train, lambdas):
    """
    K_eff = Sum(K_i / lambda_i) + I
    
    band_kernels: dict {name: K_train}
    lambdas: dict {name: lambda_val}
    """
    n = len(y_train)
    K_eff_train = np.zeros_like(next(iter(band_kernels_train.values())))
    
    # Sum bands
    for name, K in band_kernels_train.items():
        l = lambdas.get(name, 1.0)
        K_eff_train += (K / l)
        
    # Add Identity
    K_eff_train.flat[::n+1] += 1.0
    
    # Solve
    try:
        alpha = scipy.linalg.solve(K_eff_train, y_train, assume_a='pos')
    except:
        alpha = scipy.linalg.solve(K_eff_train, y_train)
        
    # Predict
    K_eff_test = np.zeros_like(next(iter(band_kernels_test.values())))
    for name, K in band_kernels_test.items():
        l = lambdas.get(name, 1.0)
        K_eff_test += (K / l)
        
    y_pred = K_eff_test @ alpha
    return y_pred


# --- TUNING ---

def tune_bands_cv(X_bands_train, y_train):
    """
    Inner CV to select best lambdas for the given set of bands.
    X_bands_train: dict {name: X_numpy}
    """
    splits = make_inner_splits(len(y_train))
    band_names = sorted(list(X_bands_train.keys()))
    
    # Grid: Cartesian product of lambdas for each band
    # WARNING: 3 bands -> 9^3 = 729 combos.
    grid_vals = [LAMBDA_GRID for _ in band_names]
    grid_combos = list(itertools.product(*grid_vals))
    
    # To save time, we precompute kernels for each split
    split_kernels = []
    for train_idx, val_idx in splits:
        kernels_tr = {}
        kernels_val = {}
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        for name, X in X_bands_train.items():
            X_tr, X_val = X[train_idx], X[val_idx]
            # Standardize Inner
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_val_s = scaler.transform(X_val)
            
            K_tr, K_val = compute_scaled_kernels(X_tr_s, X_val_s)
            kernels_tr[name] = K_tr
            kernels_val[name] = K_val
            
        split_kernels.append((kernels_tr, kernels_val, y_tr, y_val))
        
    # Eval Grid
    best_score = -np.inf
    best_combo = None
    
    # optimize: pre-divide kernels? No, summing is fast enough
    
    for combo in grid_combos:
        # Construct lambda dict
        l_dict = dict(zip(band_names, combo))
        
        scores = []
        for (k_tr, k_val, y_tr, y_val) in split_kernels:
            y_pred = fit_predict_multiband(k_tr, k_val, y_tr, l_dict)
            
            res = np.sum((y_val - y_pred)**2)
            tot = np.sum((y_val - np.mean(y_val))**2)
            scores.append(1 - (res/tot))
            
        mean_r2 = np.mean(scores)
        if mean_r2 > best_score:
            best_score = mean_r2
            best_combo = l_dict
            
    return best_combo

# --- MAIN EVALUATION ---

def calc_r2(y_true, y_p):
    res = np.sum((y_true - y_p)**2)
    tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (res/tot)

def evaluate_unique_value_nested_cv(X_bands, y, random_state=42):
    """
    Generalized N-Band Evaluation.
    X_bands: dict {name: X_full}
    """
    outer_splits = make_outer_splits(len(y), random_state=random_state)
    band_names = list(X_bands.keys())
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        # 1. Prepare Data Pools
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_pool = {k: v[train_idx] for k,v in X_bands.items()}
        X_test_pool = {k: v[test_idx] for k,v in X_bands.items()}
        
        # 2. Tune FULL Joint Model
        # We assume independent tuning for sub-models is suboptimal or too expensive.
        # Actually, for "Unique Value", we usually want the BEST Full vs BEST Reduced.
        # Ideally we tune Full, and tune Reduced separately.
        # To save massive compute, heuristic: Use lambdas from Full model?
        # NO. Removing a band changes the optimal regularization for others.
        # We MUST tune Reduced models separately to be fair.
        
        # DEFINED MODELS TO EVALUATE:
        # 1. Full (All Bands)
        # 2. Drop-A (All except A)
        # 3. Drop-B ...
        
        # Tuning Full
        best_lambdas_full = tune_bands_cv(X_train_pool, y_train)
        
        # Tuning Reduced (Leave-One-Out)
        best_lambdas_reduced = {}
        for target_dropped in band_names:
            reduced_bands = {k: v for k, v in X_train_pool.items() if k != target_dropped}
            if not reduced_bands: continue # Single band case handled elsewhere?
            
            # Tune
            best_lambdas_reduced[target_dropped] = tune_bands_cv(reduced_bands, y_train)
            
        # 3. Fit & Predict (Outer)
        # Scalers & Kernels (Full Set)
        kernels_train = {}
        kernels_test = {}
        
        for name in band_names:
            s = StandardScaler().fit(X_train_pool[name])
            X_tr_s = s.transform(X_train_pool[name])
            X_te_s = s.transform(X_test_pool[name])
            kt, ke = compute_scaled_kernels(X_tr_s, X_te_s)
            kernels_train[name] = kt
            kernels_test[name] = ke
            
        # Full Prediction
        y_pred_full = fit_predict_multiband(kernels_train, kernels_test, y_train, best_lambdas_full)
        r2_full = calc_r2(y_test, y_pred_full)
        
        # Reduced Predictions & Deltas
        deltas = {}
        reduced_r2s = {}
        
        for target_dropped in band_names:
            # Subset Kernels
            k_tr_red = {k: v for k, v in kernels_train.items() if k != target_dropped}
            k_te_red = {k: v for k, v in kernels_test.items() if k != target_dropped}
            
            # Predict
            # If only 1 band originally, this is empty? Handled by check above?
            if not k_tr_red:
                # Dropping the only band -> Null model
                r2_red = 0.0 # Bsl (mean)
            else:
                l_red = best_lambdas_reduced[target_dropped]
                y_pred_red = fit_predict_multiband(k_tr_red, k_te_red, y_train, l_red)
                r2_red = calc_r2(y_test, y_pred_red)
            
            reduced_r2s[target_dropped] = r2_red
            deltas[target_dropped] = r2_full - r2_red # Unique value of dropped band
            
        # Record
        res_dict = {
            'fold': fold_idx,
            'r2_full': r2_full
        }
        for name in band_names:
            res_dict[f'delta_{name}'] = deltas[name]
            res_dict[f'lambda_{name}'] = best_lambdas_full.get(name)
            
        fold_results.append(res_dict)
        
    return fold_results

def process_single_norm(model_name, norm_name, df_norm, band_embeddings, cue_to_idx, verbose=False):
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
    X_bands = {}
    for name, X_full in band_embeddings.items():
        if hasattr(X_full, "toarray"): 
            X_bands[name] = X_full[idxs].toarray()
        else: 
            X_bands[name] = X_full[idxs]
    
    # Evaluate
    # Dictionary of bands: {'Passive': X, 'Active': X, 'Activation': X}
    fold_res = evaluate_unique_value_nested_cv(X_bands, y)
    df_res = pd.DataFrame(fold_res)
    
    # Aggregate
    out = {
        'model': model_name,
        'norm': norm_name,
        'n_samples': len(overlap),
        'r2_full_mean': df_res['r2_full'].mean(),
        'r2_full_se': df_res['r2_full'].sem(),
    }
    
    for name in band_embeddings.keys():
        out[f'delta_{name}_mean'] = df_res[f'delta_{name}'].mean()
        out[f'delta_{name}_se'] = df_res[f'delta_{name}'].sem()
        
    return out

def main():
    parser = argparse.ArgumentParser(description="Multi-Band Ridge Regression (Passive vs Active vs Activation)")
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
        # Match embedding keys for Logic Bands
        # Band A: Passive
        # Band B: Active
        # Band C: Activation
        
        bands_found = {}
        
        # Search keys
        # Passive
        k_pass = f"passive_{model_name}_300d"
        if k_pass not in embeddings:
             # Try search
             cands = [k for k in embeddings.keys() if f"passive_" in k and "_300d" in k and model_name in k]
             if cands: k_pass = cands[0]
             
        if k_pass in embeddings:
             bands_found['Passive'] = embeddings[k_pass]
             
        # Active
        k_act = f"active_{model_name}_300d"
        if k_act not in embeddings:
             cands = [k for k in embeddings.keys() if f"active_" in k and "_300d" in k and model_name in k]
             if cands: k_act = cands[0]
             
        if k_act in embeddings:
             bands_found['Active'] = embeddings[k_act]
             
        # Activation
        k_activ = f"activation_{model_name}"
        if k_activ not in embeddings:
             cands = [k for k in embeddings.keys() if f"activation_" in k and model_name in k]
             if cands: k_activ = cands[0]
             
        if k_activ in embeddings:
             bands_found['Activation'] = embeddings[k_activ]
             
        if len(bands_found) < 2:
            print(f"[Skip] {model_name}: Found {len(bands_found)} bands. Need at least 2.")
            continue
            
        print(f"\n[Model] {model_name}")
        for bname, mat in bands_found.items():
            print(f"  > Band: {bname} ({mat.shape})")
        
        unique_norms = sorted(norms_df['norm'].unique())
        
        if args.test_limit and args.test_limit > 0:
            unique_norms = unique_norms[:args.test_limit]
            print(f"  [Test Mode] Limited to first {len(unique_norms)} norms.")
            
        # Parallel Eval
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(process_single_norm)(
                model_name, norm, norms_df, bands_found, cue_to_idx, args.verbose
            ) for norm in tqdm(unique_norms, desc=f"Norms ({model_name})")
        )
        
        results.extend([r for r in res if r])
        
    # Save
    out_file = args.output_dir / "banded_ridge_results.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n[Done] Saved results to {out_file}")

if __name__ == "__main__":
    main()
