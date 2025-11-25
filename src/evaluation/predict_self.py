"""
src/evaluation/predict_self_consistency.py

"The Introspection Test"
Can a model's foundational representation predict its OWN explicit judgments?

Method:
- Ridge Regression (L2 Regularization) with Nested Cross-Validation.
- Outer Loop: Repeated K-Fold (5x5) for robust R^2 estimation.
- Inner Loop: Leave-One-Out CV (RidgeCV) for hyperparameter tuning.
- Match Model X to Model Y using an EXPLICIT HARDCODED MAPPING.
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
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

# --- CONFIG ---
MIN_SAMPLES = 50

# --- MAPPINGS ---
EMBEDDING_TO_NORMS_MAP = {
    "passive_Llama-3.1-8B_instruct":      "llama-3.1-8b-instruct",
    "passive_Llama-3.3-70B_instruct":     "llama-3.3-70b-instruct",
    "passive_Mistral-Small-24b_instruct": "mistral-small-24b-instruct",
    "passive_Qwen3-32b_instruct":         "qwen-3-32b-instruct",
    "passive_falcon-3-10B_instruct":      "falcon-3-10b-instruct",
    "passive_gemma-3-27b_instruct":       "gemma-3-27b-instruct",
    "passive_gpt-oss-120b_instruct":      "gpt-oss-120b-instruct",
    "passive_gpt-oss-20b_instruct":       "gpt-oss-20b-instruct",
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
            if 'cleaned_rating' not in df.columns:
                print(f"  [Skip] {f.name} missing 'cleaned_rating'")
                continue
                
            df['word'] = df['word'].astype(str).str.lower().str.strip()
            df['cleaned_rating'] = pd.to_numeric(df['cleaned_rating'], errors='coerce')
            df = df.dropna(subset=['cleaned_rating'])
            
            # Store using the exact filename stem
            norm_data[f.stem] = df
            
        except Exception as e:
            print(f"  [Error] Could not read {f.name}: {e}")
            
    print(f"Loaded norms for {len(norm_data)} model variations.")
    return norm_data

def evaluate_embedding(X, y, random_state=42):
    """
    State-of-the-Art Evaluation Routine.
    Outer Loop: Repeated K-Fold (5x5).
    Inner Loop: RidgeCV with Log-Space Alpha Grid.
    """
    # 1. Define the Rigorous Hyperparameter Grid
    alphas = np.logspace(-3, 5, 20)
    
    # 2. Define Outer Validation Scheme
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    
    scores = []
    
    try:
        # 3. Manual Loop for Maximum Control
        for train_idx, test_idx in rkf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # A. Scaling (Fit on Train, Transform Test)
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            # B. Hyperparameter Tuning & Fitting (Inner Loop)
            model = RidgeCV(alphas=alphas, scoring='r2')
            model.fit(X_train_sc, y_train)
            
            # C. Evaluation (Outer Loop)
            y_pred = model.predict(X_test_sc)
            score = r2_score(y_test, y_pred)
            
            scores.append(score)
            
        # 4. Aggregate
        return np.mean(scores), np.std(scores)

    except Exception as e:
        return np.nan, np.nan

def run_self_consistency(embeddings, mappings, norm_data_dict, output_path):
    results = []
    cue_to_idx = mappings['cue_to_idx']
    
    # --- STEP 1: RESOLVE MATCHES ---
    print("\n" + "="*90)
    print(f"{'EMBEDDING SOURCE':<45} | {'TARGET NORM FILE':<30} | {'STATUS'}")
    print("-" * 90)
    
    matches = [] 
    
    for emb_key, norm_key in EMBEDDING_TO_NORMS_MAP.items():
        if emb_key not in embeddings:
            print(f"{emb_key:<45} | {norm_key:<30} | ❌ Missing Embedding")
            continue
        if norm_key not in norm_data_dict:
            print(f"{emb_key:<45} | {norm_key:<30} | ❌ Missing CSV")
            continue
            
        print(f"{emb_key:<45} | {norm_key:<30} | ✅ READY")
        matches.append((emb_key, norm_key, norm_data_dict[norm_key]))

    print("="*90 + "\n")

    # --- STEP 2: RUN REGRESSION ---
    if not matches:
        print("[Error] No valid matches found to process.")
        return

    for emb_key, norm_file_key, norms_df in tqdm(matches, desc="Running Regressions"):
        
        X_full = embeddings[emb_key]
        
        # Identify norm column
        norm_col = 'norm' if 'norm' in norms_df.columns else 'norm_name'
        unique_norms = norms_df[norm_col].unique()
        
        for norm_name in unique_norms:
            sub = norms_df[norms_df[norm_col] == norm_name]
            
            # Align
            valid_cues = set(cue_to_idx.keys())
            available_words = set(sub['word'].unique())
            overlap = list(valid_cues.intersection(available_words))
            
            if len(overlap) < MIN_SAMPLES:
                continue
                
            # Build X, y
            row_indices = [cue_to_idx[w] for w in overlap]
            X = X_full[row_indices]
            
            rating_map = sub.groupby('word')['cleaned_rating'].mean().to_dict()
            y = np.array([rating_map[w] for w in overlap])
            
            # Regression
            r2, std = evaluate_embedding(X, y)
            
            if not np.isnan(r2):
                results.append({
                    'embedding_source': emb_key,
                    'matched_norm_file': norm_file_key,
                    'norm_name': norm_name,
                    'self_r2_mean': r2,
                    'self_r2_std': std,
                    'n_samples': len(overlap)
                })

    # Save
    res_df = pd.DataFrame(results)
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