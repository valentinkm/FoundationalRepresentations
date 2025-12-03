"""
src/evaluation/run_norm_predictions.py

Batched Ridge Regression Evaluation.
Task A: Alignment (Predicting Human Norms)
Task B: Introspection (Predicting Model's Own Norms)
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
    if key == "human_matrix":
        return "Human", "Baseline"
    
    if key.startswith("activation_"):
        clean = key.replace("activation_", "").replace("-instruct", "")
        return clean, "Activation"
    
    if key.startswith("passive_"):
        clean = key.replace("passive_", "").replace("-instruct", "")
        suffix = None
        if clean.endswith("_300d"):
            clean = clean.replace("_300d", "")
            suffix = "_300d"
        if "_contrast" in clean:
            clean = clean.replace("_contrast", "")
            etype = "Behavior_Contrastive" + (suffix or "")
        elif "_shuffled" in clean:
            clean = clean.replace("_shuffled", "")
            etype = "Behavior_Shuffled" + (suffix or "")
        else:
            etype = "Behavior_Standard" + (suffix or "")
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

def run_self_predictions(behavior_items, cue_to_idx, model_norms, out_csv):
    """
    Predict each model's OWN norms using its OWN behavioral embeddings.
    Uses 'Canonical Matching' with verbose logging.
    """
    if not model_norms:
        print("⚠️ Self-prediction skipped: no model norms found.")
        return

    print(f"\n{'='*60}")
    print(f"SELF-PREDICTION ROUTINE")
    print(f"{'='*60}")

    # --- HELPER: Canonical Name ---
    def to_canonical(name):
        clean = name.lower().replace("-instruct", "").replace("-base", "")
        return clean.replace("-", "").replace("_", "").strip()

    # 1. Build Lookup Table
    norm_lookup = {}
    print(f"[Self-Pred] 🔍 indexing {len(model_norms)} norm files...")
    for stem, df in model_norms.items():
        key = to_canonical(stem)
        norm_lookup[key] = (stem, df)

    rows = []
    
    # 2. Iterate and Match
    for item in behavior_items:
        model = item['model']
        emb_type = item['type']
        X_full = item['matrix']
        
        search_key = to_canonical(model)
        match = norm_lookup.get(search_key)
        
        # --- LOGGING MATCH STATUS ---
        if match is None:
            print(f"❌ SKIP: {model:<30} | No CSV found for key: '{search_key}'")
            continue
            
        original_name, df = match
        print(f"✅ PROC: {model:<30} | Matched to: {original_name}.csv")

        # 3. Inner Loop (Norms)
        norm_col = 'norm' if 'norm' in df.columns else 'norm_name'
        grouped = df.groupby(norm_col)
        
        # Use tqdm for visual progress on the specific norms
        processed_count = 0
        skipped_count = 0
        
        for norm_name, sub in tqdm(grouped, desc=f"     -> {emb_type[:15]}...", leave=False):
            words = sub['word'].astype(str).str.lower().str.strip()
            overlap = [w for w in words.unique() if w in cue_to_idx]
            
            if len(overlap) < MIN_SAMPLES:
                skipped_count += 1
                continue
            
            idxs = [cue_to_idx[w] for w in overlap]
            y_map = sub.groupby('word')['cleaned_rating'].mean().to_dict()
            y = np.array([y_map[w] for w in overlap])
            X = X_full[idxs]
            
            r2, std = evaluate_embedding(X, y, CV_FOLDS, N_JOBS)
            
            rows.append({
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
        
        print(f"     -> Finished. Computed: {processed_count} norms. (Skipped {skipped_count} low overlap)")

    if rows:
        _save_buffer(rows, out_csv)
    else:
        print("⚠️ Self-prediction produced no records.")

def _save_buffer(buffer, path):
    if not buffer: return
    new_df = pd.DataFrame(buffer)
    if path.exists():
        try:
            old_df = pd.read_csv(path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            # Remove duplicates based on key columns, keeping the newest run
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
    
    # --- SMOKETEST CONFIG ---
    if args.smoketest:
        print("\n🔥 SMOKETEST MODE ENABLED 🔥")
        CV_FOLDS = 2
        N_JOBS = 1 
        out_csv = OUTPUT_CSV.parent / "norm_prediction_scores_SMOKETEST.csv"
        self_out_csv = SELF_OUTPUT_CSV.parent / "self_prediction_scores_SMOKETEST.csv"
        print(f"   Outputs will be saved to: {out_csv.name} & {self_out_csv.name}")
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
    norm_list = norm_df['norm_name'].unique()
    
    if args.smoketest:
        norm_list = norm_list[:1] # Only run 1 norm for speed

    # 2. Collect Embeddings
    all_embeddings = []
    
    WANTED_TYPES = {
        "Baseline", 
        "Activation", 
        "Behavior_Contrastive", 
        "Behavior_Standard",
        "Behavior_Contrastive_300d", 
        "Behavior_Standard_300d"
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

    # 3. Resume Check (Human Norms)
    processed_configs = set()
    if out_csv.exists() and not args.smoketest:
        try:
            df_exist = pd.read_csv(out_csv)
            processed_configs = set(zip(df_exist['Model'], df_exist['Embedding_Type'], df_exist['Norm']))
            if processed_configs: print(f"  -> 🔄 Resuming: Found {len(processed_configs)} completed records.")
        except: pass

    # 4. Main Loop: Human Norm Prediction
    results_buffer = []
    
    for item in all_embeddings:
        model = item['model']
        emb_type = item['type']
        X_full = item['matrix']
        max_rows = X_full.shape[0]
        
        # Skip if already done
        norms_to_run = [n for n in norm_list if (model, emb_type, n) not in processed_configs]
        
        print(f"\n{'='*60}")
        print(f"Processing: {model} - {emb_type}")
        print(f"  -> Shape: {X_full.shape}")
        print(f"  -> Remaining Norms: {len(norms_to_run)} / {len(norm_list)}")
        print(f"{'='*60}")
        
        if not norms_to_run:
            continue

        # Create valid map once per Model
        valid_map = {word: idx for word, idx in cue_to_idx.items() if idx < max_rows}

        try:
            for norm in tqdm(norms_to_run, desc=f"{model} [{emb_type}]"):
                sub_df = norm_df[norm_df['norm_name'] == norm]
                sub_df = sub_df[sub_df['word'].isin(valid_map.keys())]

                if sub_df.empty: continue

                norm_map = sub_df.set_index('word')['human_rating'].to_dict()
                
                valid_indices = []
                y_vals = []
                
                for word, rating in norm_map.items():
                    valid_indices.append(valid_map[word])
                    y_vals.append(rating)
                
                if len(valid_indices) < MIN_SAMPLES:
                    continue
                
                idxs = np.array(valid_indices)
                y = np.array(y_vals)
                
                # Check NaNs
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
    print(f"\n✅ Norm prediction complete. Results: {out_csv}")

    # 5. Secondary Loop: Self-Prediction (Behavior only)
    # We only run self-prediction on Behavioral embeddings, as Activation mappings are tricky/less relevant here
    behavior_items = [a for a in all_embeddings if a['type'].startswith("Behavior")]
    
    if behavior_items:
        run_self_predictions(behavior_items, cue_to_idx, model_norms, self_out_csv)
    else:
        print("⚠️ No behavioral embeddings found for self-prediction.")

if __name__ == "__main__":
    main()