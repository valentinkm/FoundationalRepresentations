"""
src/evaluation/run_norm_predictions.py

Batched Ridge Regression.
OPTIMIZED: Parallelized, No PCA, Streamlined Loop.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

# --- CONFIGURATION ---
MIN_SAMPLES = 50 
ALPHAS = np.logspace(-2, 6, 12) 
CV_FOLDS = 5
N_JOBS = -1

# --- PATHS ---
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path.cwd()

BEHAVIOR_PKL = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"
ACTIVATION_PKL = BASE_DIR / "outputs" / "matrices" / "activation_embeddings.pkl"
NORM_PATH = BASE_DIR / "data" / "psych_norms" / "psychnorms_subset_filtered_by_swow.csv"
OUTPUT_CSV = BASE_DIR / "outputs" / "results" / "norm_prediction_scores.csv"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_pkl(path):
    if not path.exists():
        print(f"❌ CRITICAL: File not found: {path}")
        sys.exit(1)
    print(f"Loading {path.name}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_norms():
    paths_to_try = [
        NORM_PATH,
        BASE_DIR / "data" / "SWOW" / "utils" / "psychnorms_subset_filtered_by_swow.csv"
    ]
    for p in paths_to_try:
        if p.exists():
            print(f"Loading Norms from {p.name}...")
            df = pd.read_csv(p, low_memory=False)
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['human_rating'] = pd.to_numeric(df['human_rating'], errors='coerce')
            return df.dropna(subset=['human_rating'])
            
    print(f"❌ CRITICAL: Norms not found.")
    sys.exit(1)

def parse_embedding_key(key):
    if key == "human_matrix":
        return "Human", "Baseline"
    
    if key.startswith("activation_"):
        clean = key.replace("activation_", "").replace("-instruct", "")
        return clean, "Activation"
    
    if key.startswith("passive_"):
        clean = key.replace("passive_", "").replace("-instruct", "")
        if "_contrast" in clean:
            return clean.replace("_contrast", ""), "Behavior_Contrastive"
        elif "_shuffled" in clean:
            return clean.replace("_shuffled", ""), "Behavior_Shuffled"
        else:
            return clean, "Behavior_Standard"
            
    return "Unknown", "Unknown"

def evaluate_embedding(X, y, folds, jobs):
    # Standardize -> RidgeCV
    # RidgeCV with 'auto' mode usually uses efficient Leave-One-Out SVD
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, scoring='r2')
    )
    
    # Use standard KFold (Shuffle=True) instead of RepeatedKFold to reduce overhead
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    # Run the CV in parallel
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=jobs)
    return scores.mean(), scores.std()

def _save_buffer(buffer, path):
    if not buffer: return
    new_df = pd.DataFrame(buffer)
    if path.exists():
        try:
            old_df = pd.read_csv(path)
            # Remove duplicates if we are re-running parts
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Model', 'Embedding_Type', 'Norm'], keep='last')
            combined.to_csv(path, index=False)
        except Exception as e:
            print(f"⚠️ CSV Error: {e}. Backup saved.")
            new_df.to_csv(path.parent / f"backup_{int(time.time())}.csv", index=False)
    else:
        new_df.to_csv(path, index=False)
    print(f"  💾 Saved {len(buffer)} records to {path.name}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoketest", action="store_true", help="Run fast integrity check.")
    args = parser.parse_args()

    global CV_FOLDS, N_JOBS
    
    if args.smoketest:
        print("\n🔥 SMOKETEST MODE ENABLED 🔥")
        CV_FOLDS = 2
        N_JOBS = 1 
        norm_limit = 1
        out_csv = OUTPUT_CSV.parent / "norm_prediction_scores_SMOKETEST.csv"
    else:
        out_csv = OUTPUT_CSV
        norm_limit = None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    behav_data = load_pkl(BEHAVIOR_PKL)
    act_data = load_pkl(ACTIVATION_PKL)
    norm_df = load_norms()
    
    cue_to_idx = behav_data['mappings']['cue_to_idx']
    norm_list = norm_df['norm_name'].unique()
    
    if args.smoketest:
        norm_list = norm_list[:1]

    # 2. Collect Embeddings
    all_embeddings = []
    
    WANTED_TYPES = {
        "Baseline", 
        "Activation", 
        "Behavior_Contrastive", 
        "Behavior_Standard"
    }
    
    for key, mat in behav_data['embeddings'].items():
        model, type_ = parse_embedding_key(key)
        if type_ in WANTED_TYPES:
            all_embeddings.append({"key": key, "matrix": mat, "model": model, "type": type_})
            
    for key, mat in act_data['embeddings'].items():
        model, type_ = parse_embedding_key(key)
        if type_ in WANTED_TYPES:
            all_embeddings.append({"key": key, "matrix": mat, "model": model, "type": type_})

    all_embeddings.sort(key=lambda x: (x['model'], x['type']))

    print(f"\n✅ Found {len(all_embeddings)} Target Embedding Sources")

    # Resume Check
    processed_configs = set()
    if out_csv.exists() and not args.smoketest:
        try:
            df_exist = pd.read_csv(out_csv)
            processed_configs = set(zip(df_exist['Model'], df_exist['Embedding_Type'], df_exist['Norm']))
            if processed_configs: print(f"  -> 🔄 Resuming: Found {len(processed_configs)} completed records.")
        except: pass

    results_buffer = []

    # 3. Main Loop
    for item in all_embeddings:
        model = item['model']
        emb_type = item['type']
        X_full = item['matrix']
        max_rows = X_full.shape[0]
        
        norms_to_run = [n for n in norm_list if (model, emb_type, n) not in processed_configs]
        
        print(f"\n{'='*60}")
        print(f"Processing: {model} - {emb_type}")
        print(f"  -> Shape: {X_full.shape}")
        print(f"  -> Remaining Norms: {len(norms_to_run)} / {len(norm_list)}")
        print(f"{'='*60}")
        
        if not norms_to_run:
            continue

        # OPTIMIZATION: Create valid map once per Model
        valid_map = {word: idx for word, idx in cue_to_idx.items() if idx < max_rows}

        try:
            for norm in tqdm(norms_to_run, desc=f"{model} [{emb_type}]"):
                # Fast subsetting
                sub_df = norm_df[norm_df['norm_name'] == norm]
                sub_df = sub_df[sub_df['word'].isin(valid_map.keys())]

                if sub_df.empty: continue

                norm_map = sub_df.set_index('word')['human_rating'].to_dict()
                
                valid_indices = []
                y_vals = []
                
                for word, rating in norm_map.items():
                    # No need to check 'if word in cue_to_idx' again due to pre-filter
                    valid_indices.append(valid_map[word])
                    y_vals.append(rating)
                
                if len(valid_indices) < MIN_SAMPLES:
                    continue
                
                idxs = np.array(valid_indices)
                y = np.array(y_vals)
                
                # Check NaNs on subset only (Faster than checking whole X)
                X_sub = X_full[idxs]
                if not np.isfinite(X_sub).all():
                    # Fallback: remove NaN rows
                    mask = np.isfinite(X_sub).all(axis=1)
                    X_sub = X_sub[mask]
                    y = y[mask]
                    if len(y) < MIN_SAMPLES: continue
                
                # Run Evaluation (Parallelized)
                r2, std = evaluate_embedding(X_sub, y, CV_FOLDS, N_JOBS)
                
                results_buffer.append({
                    "Model": model,
                    "Embedding_Type": emb_type,
                    "Norm": norm,
                    "N_Samples": len(y),
                    "Dimensions": X_full.shape[1],
                    "R2_Mean": r2,
                    "R2_Std": std
                })
                
                if len(results_buffer) >= 10:
                    _save_buffer(results_buffer, out_csv)
                    results_buffer = []

        except Exception as e:
            print(f"\n❌ Error on {model} - {emb_type}: {e}")
            continue

    _save_buffer(results_buffer, out_csv)
    print(f"\n✅ Pipeline Complete. Results: {out_csv}")

if __name__ == "__main__":
    main()