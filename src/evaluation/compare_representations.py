"""
src/evaluation/compare_representations.py

Mode: STRICT INTROSPECTION BENCHMARK.
Target: Compare Activation vs. ALL Behavior Variants on Self-Norm Prediction.

LOGIC:
1. Inventory: specific Activation & Behavior keys.
2. Grouping: Aggregates variants (Standard, Contrastive, 300d) by Model Family.
3. Filtering: ONLY processes families with BOTH Activation AND Behavior sources.
4. Intersection: ONLY runs on norms shared by ALL valid families (with N>=1000).
"""

import pandas as pd
import numpy as np
import pickle
import sys
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from collections import defaultdict

# --- CONFIGURATION ---
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
OUTPUT_CSV = BASE_DIR / "outputs" / "results" / "self_prediction_scores.csv"
MODEL_NORMS_DIR = BASE_DIR / "outputs" / "raw_behavior" / "model_norms"

# =============================================================================
# PARSING LOGIC
# =============================================================================

def normalize_name(name):
    """
    Normalizes names to handle mismatches (e.g., '27-b' vs '27b').
    Keeps 'instruct' vs 'base' distinct to prevent overwriting.
    """
    clean = name.lower().strip()
    # Remove separators
    clean = clean.replace("-", "").replace("_", "")
    return clean

def parse_model_family_and_type(key):
    """
    Decodes the dictionary key into:
    1. Model Family (normalized name)
    2. Embedding Type (e.g., 'Behavior_Contrastive_300d')
    """
    if key == "human_matrix": return "Human", "Baseline"
    
    # 1. Determine Broad Category & Clean Prefix
    if key.startswith("activation_"):
        category = "Activation"
        clean = key.replace("activation_", "")
    elif key.startswith("passive_"):
        category = "Behavior"
        clean = key.replace("passive_", "")
    else:
        return None, None

    # 2. Extract Variants
    is_300d = "_300d" in clean
    is_contrast = "_contrast" in clean
    is_shuffled = "_shuffled" in clean

    # 3. Clean the Model Name (Remove suffixes to find the FAMILY)
    stem = clean.replace("_300d", "").replace("_contrast", "").replace("_shuffled", "")
    
    # NOTE: We normalize dashes but KEEP 'instruct'/'base' if present 
    # so we don't mix up different model versions.
    stem = normalize_name(stem)

    # 4. Construct Type Label
    if category == "Activation":
        final_type = "Activation"
    else:
        base = "Behavior"
        if is_contrast:
            variant = "Contrastive"
        elif is_shuffled:
            variant = "Shuffled"
        else:
            variant = "Standard"
            
        dim = "_300d" if is_300d else ""
        final_type = f"{base}_{variant}{dim}"

    return stem, final_type

def load_pkl(path):
    if not path.exists():
        print(f"❌ CRITICAL: File not found: {path}")
        sys.exit(1)
    print(f"Loading {path.name}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model_norms(norm_dir: Path):
    if not norm_dir.exists(): return {}
    data = {}
    for fp in norm_dir.glob("*.csv"):
        try:
            df = pd.read_csv(fp, low_memory=False)
            if 'cleaned_rating' not in df.columns: continue
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            
            # Normalize key to match embedding keys
            clean_name = normalize_name(fp.stem)
            data[clean_name] = (fp.stem, df) 
        except Exception: pass
    print(f"Loaded {len(data)} model norm files.")
    return data

def evaluate_embedding(X, y, folds, jobs):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, scoring='r2'))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=jobs)
    return scores.mean(), scores.std()

def _save_buffer(buffer, path):
    if not buffer: return
    new_df = pd.DataFrame(buffer)
    if path.exists():
        try:
            old_df = pd.read_csv(path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Model', 'Embedding_Type', 'Norm'], keep='last')
            combined.to_csv(path, index=False)
        except:
            new_df.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)
    print(f"  💾 Saved {len(buffer)} records to {path.name}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoketest", action="store_true", help="Run verbose integrity check.")
    parser.add_argument("--fast_mode", action="store_true", help="Enable strict introspection filtering.")
    args = parser.parse_args()

    # --- SETTINGS ---
    if args.fast_mode:
        print("\n🚀 FAST MODE: Strict Introspection Enabled")
        REQUIRED_SAMPLES = 1000 
        TARGET_NORM_COUNT = 5  # Sample 5 shared norms
    else:
        REQUIRED_SAMPLES = 1000
        TARGET_NORM_COUNT = None # All shared norms

    if args.smoketest:
        print("\n🔥 SMOKETEST MODE ENABLED")
        CV_FOLDS = 2
        N_JOBS = 1
        TARGET_NORM_COUNT = 1
        REQUIRED_SAMPLES = 100 
        out_csv = OUTPUT_CSV.parent / "self_prediction_scores_SMOKETEST.csv"
    else:
        CV_FOLDS = 5
        N_JOBS = -1
        out_csv = OUTPUT_CSV

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    behav_data = load_pkl(BEHAVIOR_PKL)
    act_data = load_pkl(ACTIVATION_PKL)
    model_norms = load_model_norms(MODEL_NORMS_DIR)
    cue_to_idx = behav_data['mappings']['cue_to_idx']

    # 2. Group Embeddings by Model Family
    print(f"\n{'='*60}")
    print("🕵️‍♂️  INVENTORY CHECK")
    print(f"{'='*60}")

    model_groups = defaultdict(lambda: defaultdict(list))
    all_sources = {**behav_data.get('embeddings', {}), **act_data.get('embeddings', {})}

    for key, mat in all_sources.items():
        family, etype = parse_model_family_and_type(key)
        if family and family != "Human":
            model_groups[family][etype] = mat

    # 3. Validate Families
    valid_families = []
    
    print(f"{'Model Family':<25} | {'Act':<5} | {'Std':<5} | {'Cont':<5} | {'300d':<5} | {'Norms':<5} | Status")
    print("-" * 85)

    for family in sorted(model_groups.keys()):
        types = model_groups[family].keys()
        
        has_act = "Activation" in types
        has_std = any("Standard" in t for t in types)
        has_cont = any("Contrastive" in t for t in types)
        has_300 = any("300d" in t for t in types)
        
        # Check if norms exist for this family key (normalized)
        has_norms = False
        for norm_key in model_norms.keys():
            if norm_key in family or family in norm_key: # Fuzzy match for safety
                has_norms = True
                break

        # CRITERIA: Must have Activation AND (Standard OR Contrastive) AND Norms
        is_valid = has_act and (has_std or has_cont) and has_norms
        
        status = "✅ Ready" if is_valid else "❌ Skip"
        if is_valid: valid_families.append(family)

        if args.smoketest or is_valid:
            print(f"{family:<25} | {str(has_act)[0]:<5} | {str(has_std)[0]:<5} | {str(has_cont)[0]:<5} | {str(has_300)[0]:<5} | {str(has_norms)[0]:<5} | {status}")

    print(f"\n🎯 Qualified Model Families: {len(valid_families)}")
    if not valid_families:
        print("❌ CRITICAL: No qualified model families found.")
        sys.exit(1)

    # 4. Global Norm Intersection (Constraint: Norms must exist for ALL valid families)
    print(f"\n🔗 CALCULATING SHARED NORM INTERSECTION")
    
    family_valid_norms = {}
    
    for family in valid_families:
        # Find the matching norm file
        matched_key = next(k for k in model_norms.keys() if k in family or family in k)
        original_name, df_norms = model_norms[matched_key]
        
        norm_col = 'norm' if 'norm' in df_norms.columns else 'norm_name'
        
        valid_set = set()
        for n in df_norms[norm_col].unique():
            sub = df_norms[df_norms[norm_col] == n]
            words = sub['word'].astype(str).str.lower().str.strip()
            overlap_count = sum(1 for w in words if w in cue_to_idx)
            if overlap_count >= REQUIRED_SAMPLES:
                valid_set.add(n)
        family_valid_norms[family] = valid_set

    shared_norms = set.intersection(*family_valid_norms.values())
    sorted_shared = sorted(list(shared_norms))
    
    print(f"✅ Shared Valid Norms: {len(sorted_shared)}")
    if not sorted_shared:
        print(f"❌ No norms meet the {REQUIRED_SAMPLES} word requirement across all models.")
        print(f"   (Try reducing REQUIRED_SAMPLES if this is too strict)")
        sys.exit(1)

    # Select Targets
    if TARGET_NORM_COUNT and len(sorted_shared) > TARGET_NORM_COUNT:
        random.seed(42) 
        target_norms = random.sample(sorted_shared, TARGET_NORM_COUNT)
        print(f"🎯 Selected {TARGET_NORM_COUNT} Targets: {target_norms}")
    else:
        target_norms = sorted_shared

    # 5. Execution Loop
    print(f"\n🚀 STARTING BENCHMARK (Running only on Qualified Families)")
    results_buffer = []

    for family in valid_families:
        print(f"\nModel Family: {family}")
        
        # Find norm file again
        matched_key = next(k for k in model_norms.keys() if k in family or family in k)
        original_norm_name, df_norms = model_norms[matched_key]
        norm_col = 'norm' if 'norm' in df_norms.columns else 'norm_name'
        
        variants = sorted(model_groups[family].keys())
        
        for etype in variants:
            if "Shuffled" in etype: continue 
            
            X_full = model_groups[family][etype]

            for norm_name in target_norms:
                sub = df_norms[df_norms[norm_col] == norm_name]
                words = sub['word'].astype(str).str.lower().str.strip()
                overlap = sorted(list(set([w for w in words if w in cue_to_idx])))
                
                # Deterministic Sampling per (Family + Norm)
                rng_norm = random.Random(f"{family}_{norm_name}_seed")
                
                if len(overlap) < REQUIRED_SAMPLES: continue
                selected_words = rng_norm.sample(overlap, REQUIRED_SAMPLES)
                
                idxs = [cue_to_idx[w] for w in selected_words]
                y_map = sub.groupby('word')['cleaned_rating'].mean().to_dict()
                y = np.array([y_map[w] for w in selected_words])
                X = X_full[idxs]
                
                r2, std = evaluate_embedding(X, y, CV_FOLDS, N_JOBS)
                
                results_buffer.append({
                    "Model": family, 
                    "Embedding_Type": etype,
                    "Norm_File": original_norm_name,
                    "Norm": norm_name,
                    "N_Samples": len(y),
                    "Dimensions": X_full.shape[1],
                    "R2_Mean": r2,
                    "R2_Std": std
                })
                
                if args.smoketest:
                     print(f"   -> {etype:<25} | {norm_name} | R2: {r2:.3f}")

        _save_buffer(results_buffer, out_csv)
        results_buffer = [] 

    print(f"\n✅ Benchmark Complete. Results in: {out_csv}")

if __name__ == "__main__":
    main()