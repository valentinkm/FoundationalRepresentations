"""
src/evaluation/predict_self_consistency_parallel.py

"The Introspection Test" - Parallelized for High-Core Systems
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

# --- PARALLELIZATION IMPORTS ---
from joblib import Parallel, delayed
# threadpoolctl helps prevent numpy from launching threads inside our processes
from threadpoolctl import threadpool_limits 

# --- CONFIG ---
MIN_SAMPLES = 50

# --- MAPPINGS ---
EMBEDDING_TO_NORMS_MAP = {
    "passive_gemma-3-27-b-instruct":                "gemma-3-27b-instruct",
    "passive_gemma-3-27-b-instruct_shuffled":       "gemma-3-27b-instruct",
    "passive_gemma-3-27-b-instruct_contrast":       "gemma-3-27b-instruct",

    "passive_gpt-oss-20-b-instruct":                "gpt-oss-20b-instruct",
    "passive_gpt-oss-20-b-instruct_shuffled":       "gpt-oss-20b-instruct",
    "passive_gpt-oss-20-b-instruct_contrast":       "gpt-oss-20b-instruct",

    "passive_mistral-small-24-b-instruct":          "mistral-small-24b-instruct",
    "passive_mistral-small-24-b-instruct_shuffled": "mistral-small-24b-instruct",
    "passive_mistral-small-24-b-instruct_contrast": "mistral-small-24b-instruct",

    "passive_qwen-3-32-b-instruct":                 "qwen-3-32b-instruct",
    "passive_qwen-3-32-b-instruct_shuffled":        "qwen-3-32b-instruct",
    "passive_qwen-3-32-b-instruct_contrast":        "qwen-3-32b-instruct",
}

def load_embeddings(pkl_path):
    print(f"[Loader] Loading embeddings from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['mappings']

def load_model_norms_directory(norm_dir: Path):
    print(f"[Loader] Loading model norms from {norm_dir}...")
    norm_data = {}
    for f in norm_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            if 'cleaned_rating' not in df.columns: continue
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            norm_data[f.stem] = df
        except Exception as e:
            print(f"  [Error] Could not read {f.name}: {e}")
    print(f"Loaded norms for {len(norm_data)} model variations.")
    return norm_data

def evaluate_embedding(X, y, random_state=42):
    """
    State-of-the-Art Evaluation Routine.
    """
    # Enforce single-threaded execution for numpy/scikit-learn within this worker
    # This prevents 128 processes * 128 threads explosion.
    with threadpool_limits(limits=1, user_api='blas'):
        alphas = np.logspace(-3, 5, 20)
        rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
        scores = []
        
        try:
            for train_idx, test_idx in rkf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc = scaler.transform(X_test)
                
                model = RidgeCV(alphas=alphas, scoring='r2')
                model.fit(X_train_sc, y_train)
                
                y_pred = model.predict(X_test_sc)
                scores.append(r2_score(y_test, y_pred))
                
            return np.mean(scores), np.std(scores)
        except Exception:
            return np.nan, np.nan

def process_single_task(emb_key, norm_file_key, norm_name, X, y):
    """
    Wrapper function to be pickled and sent to workers.
    """
    r2, std = evaluate_embedding(X, y)
    
    if np.isnan(r2):
        return None
        
    return {
        'embedding_source': emb_key,
        'matched_norm_file': norm_file_key,
        'norm_name': norm_name,
        'self_r2_mean': r2,
        'self_r2_std': std,
        'n_samples': len(y)
    }

def run_self_consistency(embeddings, mappings, norm_data_dict, output_path):
    cue_to_idx = mappings['cue_to_idx']
    
    # --- STEP 1: RESOLVE MATCHES ---
    matches = [] 
    for emb_key, norm_key in EMBEDDING_TO_NORMS_MAP.items():
        if emb_key in embeddings and norm_key in norm_data_dict:
            matches.append((emb_key, norm_key, norm_data_dict[norm_key]))
            
    print(f"Found {len(matches)} model-to-data pairs to process.")

    # --- STEP 2: PREPARE TASKS ---
    tasks = []
    
    print("Preparing task list (calculating overlaps)...")
    for emb_key, norm_file_key, norms_df in matches:
        X_full = embeddings[emb_key]
        norm_col = 'norm' if 'norm' in norms_df.columns else 'norm_name'
        unique_norms = norms_df[norm_col].unique()
        
        for norm_name in unique_norms:
            sub = norms_df[norms_df[norm_col] == norm_name]
            
            valid_cues = set(cue_to_idx.keys())
            available_words = set(sub['word'].unique())
            overlap = list(valid_cues.intersection(available_words))
            
            if len(overlap) < MIN_SAMPLES:
                continue
                
            # Create specific X and y arrays for this task.
            # This slices the large matrix into a smaller one (~10-50MB)
            # which is safe to pickle and send to workers.
            row_indices = [cue_to_idx[w] for w in overlap]
            X = X_full[row_indices] # Copies data
            
            rating_map = sub.groupby('word')['cleaned_rating'].mean().to_dict()
            y = np.array([rating_map[w] for w in overlap])
            
            tasks.append((emb_key, norm_file_key, norm_name, X, y))

    print(f"Total regression tasks found: {len(tasks)}")

    # --- STEP 3: PARALLEL EXECUTION ---
    # n_jobs=-1 uses all available cores.
    # verbose=10 gives a progress bar.
    print(f"Starting parallel execution on {os.cpu_count()} cores...")
    
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_single_task)(t[0], t[1], t[2], t[3], t[4]) for t in tasks
    )

    # Filter out None results (failed runs)
    clean_results = [r for r in results if r is not None]

    # Save
    res_df = pd.DataFrame(clean_results)
    res_df.to_csv(output_path, index=False)
    print(f"\n[Self-Consistency] Results saved to {output_path}")
    
    if not res_df.empty:
        print("\n--- Self-Consistency Leaderboard (Mean R^2) ---")
        print(res_df.groupby('embedding_source')['self_r2_mean'].mean().sort_values(ascending=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', type=Path, required=True)
    parser.add_argument('--model_norms_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    
    embeddings, mappings = load_embeddings(args.embeddings_path)
    norm_data = load_model_norms_directory(args.model_norms_dir)
    
    output_csv = args.output_dir / "self_consistency_results.csv"
    run_self_consistency(embeddings, mappings, norm_data, output_csv)

if __name__ == "__main__":
    import sys
    # ... (Keep original boilerplate logic here) ...
    try:
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        default_emb = project_root / 'outputs' / 'matrices' / 'behavioral_embeddings.pkl'
        default_norms = project_root / 'outputs' / 'raw_behavior' / 'model_norms'
        default_out = project_root / 'outputs' / 'results'
    except:
        default_emb = Path('.')
        default_norms = Path('.')
        default_out = Path('.')

    if len(sys.argv) == 1:
        sys.argv.extend([
            '--embeddings_path', str(default_emb),
            '--model_norms_dir', str(default_norms),
            '--output_dir', str(default_out)
        ])
    
    main()