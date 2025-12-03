"""
src/evaluation/compare_representations.py

Batched Ridge Regression Evaluation.
Task A: Alignment (Predicting Human Norms) -> FAST MODE: Sample 20 Global Targets.
Task B: Introspection (Predicting Model's Own Norms) -> FAST MODE: Keep ALL Norms (~15) but subsample rows.

UPDATES:
- FAST MODE: Row subsampling ENABLED (Max 1000 words).
- FAST MODE: Human Norm targets restricted to 20.
- Ensures 300d embeddings are processed.
"""

import pandas as pd
import numpy as np
import pickle
import sys
import argparse
import time
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from collections import Counter

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
SELF_OUTPUT_CSV = BASE_DIR / "outputs" / "results" / "self_prediction_scores.csv"
MODEL_NORMS_DIR = BASE_DIR / "outputs" / "raw_behavior" / "model_norms"

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

def load_model_norms(norm_dir: Path):
    if not norm_dir.exists():
        print(f"⚠️ Model norms dir missing: {norm_dir}")
        return {}
    data = {}
    for fp in norm_dir.glob("*.csv"):
        try:
            df = pd.read_csv(fp, low_memory=False)
            if 'cleaned_rating' not in df.columns:
                continue
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            data[fp.stem] = df
        except Exception as e:
            print(f"⚠️ Skip {fp.name}: {e}")
    print(f"Loaded {len(data)} model norm files.")
    return data

def parse_embedding_key(key):
    """Categorize embedding keys into types."""
    # 1. Human Baseline
    if key == "human_matrix":
        return "Human", "Baseline"
    
    # 2. Activations
    if key.startswith("activation_"):
        clean = key.replace("activation_", "").replace("-instruct", "")
        return clean, "Activation"
    
    # 3. Behavioral
    if key.startswith("passive_"):
        clean = key.replace("passive_", "").replace("-instruct", "")
        
        # Check for 300d Suffix
        suffix = ""
        if "_300d" in clean:
            clean = clean.replace("_300d", "")
            suffix = "_300d"
            
        if "_contrast" in clean:
            clean = clean.replace("_contrast", "")
            etype = "Behavior_Contrastive" + suffix
        elif "_shuffled" in clean:
            clean = clean.replace("_shuffled", "")
            etype = "Behavior_Shuffled" + suffix
        else:
            etype = "Behavior_Standard" + suffix
            
        return clean, etype
            
    return "Unknown", "Unknown"

def evaluate_embedding(X, y, folds, jobs):
    """Standardized Ridge Regression with Cross-Validation."""
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS, scoring='r2')
    )
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=jobs)
    return scores.mean(), scores.std()

def _save_buffer(buffer, path):
    if not buffer: return
    new_df = pd.DataFrame(buffer)
    if path.exists():
        try:
            old_df = pd.read_csv(path)
            # Remove duplicates based on key columns, keeping the newest run
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Model', 'Embedding_Type', 'Norm'], keep='last')
            combined.to_csv(path, index=False)
        except Exception as e:
            print(f"⚠️ CSV Error: {e}. Backup saved.")
            new_df.to_csv(path.parent / f"backup_{int(time.time())}.csv", index=False)
    else:
        new_df.to_csv(path, index=False)
    print(f"  💾 Saved {len(buffer)} records to {path.name}")

def run_self_predictions(embeddings_to_test, cue_to_idx, model_norms, out_csv, max_samples=None):
    """
    Predict each model's OWN norms using its OWN embeddings (Behavior + Activation).
    """
    if not model_norms:
        print("⚠️ Self-prediction skipped: no model norms found.")
        return

    print(f"\n{'='*60}")
    print(f"SELF-PREDICTION ROUTINE")
    if max_samples:
        print(f"   - Row Limit: {max_samples} (Subsampled)")
    else:
        print(f"   - Row Limit: None (Full Overlap)")
    print(f"{'='*60}")

    def to_canonical(name):
        clean = name.lower().replace("-instruct", "").replace("-base", "")
        return clean.replace("-", "").replace("_", "").strip()

    norm_lookup = {}
    print(f"[Self-Pred] 🔍 indexing {len(model_norms)} norm files...")
    for stem, df in model_norms.items():
        key = to_canonical(stem)
        norm_lookup[key] = (stem, df)

    # 2. Iterate and Match
    for item in embeddings_to_test:
        model = item['model']
        emb_type = item['type']
        X_full = item['matrix']
        
        # Match model embedding to its own norm file
        search_key = to_canonical(model)
        match = norm_lookup.get(search_key)
        
        if match is None:
            continue
            
        original_name, df = match
        print(f"✅ PROC: {model:<30} | {emb_type:<25} | Matched: {original_name}")

        rng = random.Random(search_key) # Consistent seeding

        model_buffer = [] 
        processed_count = 0
        
        norm_col = 'norm' if 'norm' in df.columns else 'norm_name'
        available_norms = sorted(df[norm_col].unique()) 

        for norm_name in tqdm(available_norms, desc=f"     -> {emb_type[:15]}...", leave=False):
            sub = df[df[norm_col] == norm_name]
            words = sub['word'].astype(str).str.lower().str.strip()
            overlap = sorted(list(set([w for w in words.unique() if w in cue_to_idx])))
            
            if len(overlap) < MIN_SAMPLES:
                continue
            
            # Subsample Rows (Only if max_samples is set)
            if max_samples and len(overlap) > max_samples:
                overlap = rng.sample(overlap, max_samples)
            
            idxs = [cue_to_idx[w] for w in overlap]
            y_map = sub.groupby('word')['cleaned_rating'].mean().to_dict()
            y = np.array([y_map[w] for w in overlap])
            X = X_full[idxs]
            
            r2, std = evaluate_embedding(X, y, CV_FOLDS, N_JOBS)
            
            model_buffer.append({
                "Model": model,
                "Embedding_Type": emb_type,
                "Norm_File": original_name,
                "Norm": norm_name,
                "N_Samples": len(y),
                "Dimensions": X_full.shape[1],
                "R2_Mean": r2,
                "R2_Std": std
            })
            processed_count += 1
        
        if model_buffer:
            _save_buffer(model_buffer, out_csv)
        else:
            print(f"     -> Finished (No valid norms found).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoketest", action="store_true", help="Run fast integrity check.")
    parser.add_argument("--fast_mode", action="store_true", help="Limit rows and norm targets for speed.")
    args = parser.parse_args()

    global CV_FOLDS, N_JOBS
    
    # --- CONFIGURATION LOGIC ---
    
    if args.fast_mode:
        MAX_SAMPLES = 1000  # <--- LIMIT RESTORED FOR SPEED
        HUMAN_NORM_LIMIT = 20
        print(f"\n🚀 FAST MODE ENABLED")
        print(f"   - Max Words per Norm: {MAX_SAMPLES}")
        print(f"   - Human Norm Targets: {HUMAN_NORM_LIMIT}")
    else:
        MAX_SAMPLES = None
        HUMAN_NORM_LIMIT = None

    # SMOKETEST OVERRIDES
    if args.smoketest:
        print("\n🔥 SMOKETEST MODE ENABLED 🔥")
        CV_FOLDS = 2
        N_JOBS = 1 
        HUMAN_NORM_LIMIT = 1
        MAX_SAMPLES = 100
        out_csv = OUTPUT_CSV.parent / "norm_prediction_scores_SMOKETEST.csv"
        self_out_csv = SELF_OUTPUT_CSV.parent / "self_prediction_scores_SMOKETEST.csv"
    else:
        out_csv = OUTPUT_CSV
        self_out_csv = SELF_OUTPUT_CSV

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    behav_data = load_pkl(BEHAVIOR_PKL)
    act_data = load_pkl(ACTIVATION_PKL)
    norm_df = load_norms()
    model_norms = load_model_norms(MODEL_NORMS_DIR)
    
    cue_to_idx = behav_data['mappings']['cue_to_idx']
    norm_list = sorted(norm_df['norm_name'].unique()) 
    
    # --- GLOBAL NORM REDUCTION (HUMAN LOOP) ---
    if HUMAN_NORM_LIMIT:
        random.seed(42) # Fixed Seed for Norm Selection
        norm_list = random.sample(norm_list, HUMAN_NORM_LIMIT)

    # 2. Collect Embeddings
    all_embeddings = []
    
    # Explicitly list all wanted types, including 300d
    WANTED_TYPES = {
        "Baseline", "Activation", 
        "Behavior_Contrastive", "Behavior_Standard",
        "Behavior_Contrastive_300d", "Behavior_Standard_300d"
    }
    
    def gather_embeddings(source_dict):
        for key, mat in source_dict.items():
            model, type_ = parse_embedding_key(key)
            if type_ in WANTED_TYPES:
                all_embeddings.append({"key": key, "matrix": mat, "model": model, "type": type_})

    gather_embeddings(behav_data.get('embeddings', {}))
    gather_embeddings(act_data.get('embeddings', {}))
    
    all_embeddings.sort(key=lambda x: (x['model'], x['type']))

    # --- DEBUG: PRINT FOUND TYPES ---
    print(f"\n✅ Found {len(all_embeddings)} Target Embedding Sources")
    type_counts = Counter([x['type'] for x in all_embeddings])
    print("   Breakdown:", dict(type_counts))

    # 3. Resume Check (Human Norms)
    processed_configs = set()
    if out_csv.exists() and not args.smoketest:
        try:
            df_exist = pd.read_csv(out_csv)
            processed_configs = set(zip(df_exist['Model'], df_exist['Embedding_Type'], df_exist['Norm']))
        except: pass

    # 4. Main Loop: Human Norm Prediction
    results_buffer = []
    
    for item in all_embeddings:
        model = item['model']
        emb_type = item['type']
        X_full = item['matrix']
        max_rows = X_full.shape[0]
        
        norms_to_run = [n for n in norm_list if (model, emb_type, n) not in processed_configs]
        
        if not norms_to_run: continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {model} - {emb_type}")
        print(f"  -> Remaining Norms: {len(norms_to_run)} / {len(norm_list)}")
        print(f"{'='*60}")

        clean_name = model.lower().replace("-instruct", "").replace("-base", "").replace("-", "").strip()
        rng = random.Random(clean_name)

        valid_map = {word: idx for word, idx in cue_to_idx.items() if idx < max_rows}

        try:
            for norm in tqdm(norms_to_run, desc=f"{model} [{emb_type}]"):
                sub_df = norm_df[norm_df['norm_name'] == norm]
                sub_df = sub_df[sub_df['word'].isin(valid_map.keys())]

                if sub_df.empty: continue

                norm_map = sub_df.set_index('word')['human_rating'].to_dict()
                overlap = sorted(list(norm_map.keys())) 
                
                if len(overlap) < MIN_SAMPLES: continue

                # ROW SAMPLING: Applied if MAX_SAMPLES is set
                if MAX_SAMPLES and len(overlap) > MAX_SAMPLES:
                    overlap = rng.sample(overlap, MAX_SAMPLES)
                
                idxs = [valid_map[w] for w in overlap]
                y = np.array([norm_map[w] for w in overlap])
                
                X_sub = X_full[idxs]
                if not np.isfinite(X_sub).all():
                    mask = np.isfinite(X_sub).all(axis=1)
                    X_sub = X_sub[mask]
                    y = y[mask]
                    if len(y) < MIN_SAMPLES: continue
                
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
    print(f"\n✅ Human Norm prediction complete. Results: {out_csv}")

    # 5. Secondary Loop: Self-Prediction
    self_pred_items = [
        a for a in all_embeddings 
        if a['type'].startswith("Behavior") or a['type'] == "Activation"
    ]
    
    if self_pred_items:
        # Pass MAX_SAMPLES to enforce the row limit
        run_self_predictions(self_pred_items, cue_to_idx, model_norms, self_out_csv, max_samples=MAX_SAMPLES)
    else:
        print("⚠️ No suitable embeddings found for self-prediction.")

if __name__ == "__main__":
    main()