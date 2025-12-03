"""
src/behavior/vectorize_behavior.py

The "Feature Factory" for Behavioral Representations.
Constructs Semantic Matrices aligned with the "Contrastive" methodology.

FEATURES:
- Matches SVD dimensions to specific Model Hidden States (and also exports fixed 300d variants).
- Generates 'human_matrix' embedding (Baseline).
- Auto-switches to Randomized SVD for high-dimensions (Speed Fix).
- Dimension strategy options:
  - activation (default): mirror dims from activation_embeddings.pkl
  - mapping: use MODEL_TARGET_DIMS
  - fixed: user-specified dim for all non-human matrices

Inputs:
1. Human SWOW Data (Ground Truth for Vocab/Rows)
2. Passive Real: Log-probability CSVs
3. Passive Deranged: Log-probability CSVs (Shuffled/Deranged)
4. Active Behavior: Generated Association JSONLs

Outputs:
- 'behavioral_embeddings.pkl' containing:
  - 'human_matrix': Human Counts SVD (300d)
  - 'passive_{model}': Real SVD embedding (Matched Dim)
  - 'passive_{model}_contrast': Real+Deranged SVD embedding (Matched Dim)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import argparse
from scipy.sparse import csr_matrix, hstack
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
import warnings

# --- CONFIGURATION ---
ROW_NORM_EPS = 1e-12
MIN_FREQ_THRESHOLD = 5

# Models to keep in limited mode (default)
LIMITED_MODE_MODELS = {
    # GPT-20B
    "gpt-oss-20b-instruct",
    "gpt-oss-20-b-instruct",
    # Mistral
    "mistral-small-24b-instruct",
    "mistral-small-24-b-instruct",
    # Gemma 27B
    "gemma-3-27b-instruct",
    "gemma-3-27-b-instruct",
    # Qwen 32B
    "qwen-3-32b-instruct",
    "qwen-3-32-b-instruct",
}

# Map normalized model names to their Activation Hidden Sizes (used as fallback)
MODEL_TARGET_DIMS = {}

# Human data and unknown models default to standard semantic space size
DEFAULT_FALLBACK_DIM = 300 

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def infer_default_paths():
    script_dir = Path(__file__).parent.resolve()
    # Try to find project root by looking for 'data' folder up the tree
    project_root = script_dir
    for parent in script_dir.parents:
        if (parent / 'data').exists():
            project_root = parent
            break

    default_swow = project_root / 'data' / 'SWOW' / 'Human_SWOW-EN.R100.20180827.csv'
    default_passive = project_root / 'outputs' / 'raw_behavior' / 'model_swow_logprobs'
    default_deranged = project_root / 'outputs' / 'raw_behavior' / 'model_swow_logprobs_deranged'
    default_active = project_root / 'outputs' / 'raw_behavior' / 'model_swow'
    default_out = project_root / 'outputs' / 'matrices'
    
    return default_swow, default_passive, default_deranged, default_active, default_out

def normalize_model_name(raw: str) -> str:
    """Standardize model identifiers."""
    import re
    # Remove specific suffixes for cleaner matching
    name = raw.replace("-deranged-results", "").replace("-results", "")
    
    name = name.strip().lower().replace(' ', '-').replace('_', '-')
    name = re.sub(r'(?<=[a-z])(?=\d)', '-', name)
    name = re.sub(r'(?<=\d)(?=[a-z])', '-', name)
    while '--' in name:
        name = name.replace('--', '-')
    return name

def should_process_model(model_name: str, allowed_models: set | None) -> bool:
    """Return True if the model should be processed under the current mode."""
    return allowed_models is None or model_name in allowed_models

def load_activation_dims(activation_pkl: Path) -> dict:
    """
    Load activation embedding shapes so we can match target dimensions.
    Returns {normalized_model_name: dim}.
    """
    if not activation_pkl.exists():
        print(f"[Dims] Activation pickle not found at {activation_pkl}.")
        return {}
    try:
        with open(activation_pkl, "rb") as f:
            payload = pickle.load(f)
        embeddings = payload.get("embeddings", {})
    except Exception as e:
        print(f"[Dims] Failed to read activation pickle: {e}")
        return {}

    dim_map = {}
    for key, mat in embeddings.items():
        if not key.startswith("activation_"):
            continue
        model_raw = key.replace("activation_", "")
        model_norm = normalize_model_name(model_raw)
        dim_map[model_norm] = mat.shape[1]
    print(f"[Dims] Loaded activation dims for {len(dim_map)} models.")
    return dim_map

def extract_model_from_key(key: str) -> str:
    """Strip prefixes/suffixes to recover the model name portion."""
    model = key
    for prefix in ("passive_", "active_", "activation_"):
        if model.startswith(prefix):
            model = model[len(prefix):]
    for suffix in ("_contrast", "_shuffled"):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
    return normalize_model_name(model)

def load_human_swow(human_csv_path: Path, min_freq: int) -> tuple[pd.DataFrame, dict, dict]:
    print(f"[Human] Loading SWOW from {human_csv_path}...")
    df = pd.read_csv(human_csv_path)
    
    # 1. Melt to long format
    base = df[["cue", "R1", "R2", "R3"]].copy()
    base["cue"] = base["cue"].astype(str).str.lower().str.strip()
    long = base.melt(id_vars=["cue"], value_vars=["R1", "R2", "R3"], 
                     var_name="slot", value_name="response_word")
    
    # 2. Basic Cleaning
    long = long.dropna(subset=["response_word"])
    long["response_word"] = long["response_word"].astype(str).str.lower().str.strip()
    long = long[long["response_word"] != ""]
    
    # 3. Filter Vocabulary by Frequency
    word_counts = long["response_word"].value_counts()
    valid_words = set(word_counts[word_counts >= min_freq].index)
    df_filtered = long[long["response_word"].isin(valid_words)].copy()
    
    # 4. Build Index Mappings
    all_cues = sorted(df_filtered["cue"].unique())
    all_responses = sorted(list(valid_words))
    
    cue_to_idx = {c: i for i, c in enumerate(all_cues)}
    idx_to_cue = {i: c for i, c in enumerate(all_cues)}
    response_to_idx = {r: i for i, r in enumerate(all_responses)}
    
    mappings = {
        'cue_to_idx': cue_to_idx, 
        'idx_to_cue': idx_to_cue,
        'response_to_idx': response_to_idx
    }
    
    print(f"[Vocab] Defined Space: {len(all_cues)} Cues x {len(all_responses)} Responses.")
    return df_filtered, mappings, valid_words

def build_prob_matrix(df: pd.DataFrame, mappings: dict, num_cues: int, num_responses: int) -> csr_matrix:
    """
    Helper: Converts Tidy DF (cue, response, logprob) -> Row-Normalized CSR Matrix
    """
    cue_to_idx = mappings['cue_to_idx']
    response_to_idx = mappings['response_to_idx']

    # Map indices
    row_idx = df['cue'].map(cue_to_idx).values
    col_idx = df['response_set'].map(response_to_idx).values
    
    # Aggregate Duplicates (avg logprob per cue-response pair)
    temp = pd.DataFrame({'row': row_idx, 'col': col_idx, 'lp': df['normalized_log_prob']})
    agg = temp.groupby(['row', 'col'])['lp'].mean().reset_index()
    
    # LogProb -> Probability
    agg['score'] = np.exp(agg['lp'])
    
    # Row Normalization (Prob Distribution)
    row_sums = agg.groupby('row')['score'].transform('sum')
    agg['prob'] = np.where(row_sums > 0, agg['score'] / row_sums, 0.0)
    
    return csr_matrix((agg['prob'], (agg['row'], agg['col'])), 
                      shape=(num_cues, num_responses))

def process_passive_behavior(real_dir: Path, deranged_dir: Path, mappings: dict, vocab_set: set,
                             allowed_models: set | None = None) -> dict:
    """
    Ingest Passive Logprobs (Real) + Passive Logprobs (Deranged).
    """
    if not real_dir.exists():
        print(f"[Passive] Directory not found: {real_dir}")
        return {}
        
    matrices = {}
    num_cues = len(mappings['cue_to_idx'])
    num_responses = len(mappings['response_to_idx'])
    
    # 1. Index Real Files
    real_files = sorted(list(real_dir.glob('*.csv')))
    print(f"[Passive] Found {len(real_files)} real logprob files.")
    
    # 2. Index Deranged Files
    deranged_map = {}
    if deranged_dir.exists():
        for f in deranged_dir.glob('*.csv'):
            norm_name = normalize_model_name(f.stem)
            deranged_map[norm_name] = f
            
    for fp in real_files:
        raw_name = fp.stem
        model_name = normalize_model_name(raw_name)
        
        if "deranged" in model_name: continue
        if not should_process_model(model_name, allowed_models):
            print(f"[Passive] Skipping {model_name} (limited mode).")
            continue

        try:
            # --- A. Process REAL ---
            df = pd.read_csv(fp)
            df['cue'] = df['cue'].astype(str).str.lower().str.strip()
            df['response_set'] = df['response_set'].astype(str).str.lower().str.split(',')
            df = df.explode('response_set')
            df['response_set'] = df['response_set'].astype(str).str.strip()
            
            mask = (df['cue'].isin(mappings['cue_to_idx'])) & (df['response_set'].isin(vocab_set))
            df = df[mask].copy()
            
            if df.empty: continue

            mat_real = build_prob_matrix(df, mappings, num_cues, num_responses)
            matrices[f"passive_{model_name}"] = mat_real
            
            # --- B. Process DERANGED & CONTRAST ---
            deranged_path = deranged_map.get(model_name)
            
            if deranged_path:
                df_d = pd.read_csv(deranged_path)
                df_d['cue'] = df_d['cue'].astype(str).str.lower().str.strip()
                df_d['response_set'] = df_d['response_set'].astype(str).str.lower().str.split(',')
                df_d = df_d.explode('response_set')
                df_d['response_set'] = df_d['response_set'].astype(str).str.strip()
                
                mask_d = (df_d['cue'].isin(mappings['cue_to_idx'])) & (df_d['response_set'].isin(vocab_set))
                df_d = df_d[mask_d].copy()
                
                if not df_d.empty:
                    mat_deranged = build_prob_matrix(df_d, mappings, num_cues, num_responses)
                    
                    matrices[f"passive_{model_name}_shuffled"] = mat_deranged
                    
                    # Contrastive (Real | Deranged) -> 2x width
                    mat_contrast = hstack([mat_real, mat_deranged], format='csr')
                    matrices[f"passive_{model_name}_contrast"] = mat_contrast
                    
                    print(f"[Passive] {model_name}: Real {mat_real.shape} | Contrast {mat_contrast.shape}")
                else:
                    print(f"[Passive] {model_name}: Deranged file empty after filtering.")
            else:
                print(f"[Passive] {model_name}: No corresponding deranged file found.")
                
        except Exception as e:
            print(f"[Passive] Error processing {model_name}: {e}")
            
    return matrices

def process_active_generation(input_dir: Path, mappings: dict, vocab_set: set,
                              allowed_models: set | None = None) -> dict:
    """Ingest Active Generated JSONLs."""
    if not input_dir.exists(): return {}

    matrices = {}
    cue_to_idx = mappings['cue_to_idx']
    response_to_idx = mappings['response_to_idx']
    num_cues = len(cue_to_idx)
    num_responses = len(response_to_idx)

    files = sorted(list(input_dir.glob('*.jsonl')))
    print(f"[Active] Found {len(files)} generation files.")

    for fp in files:
        raw_model_name = fp.stem
        model_name = normalize_model_name(raw_model_name)
        data_rows = []
        
        if not should_process_model(model_name, allowed_models):
            print(f"[Active] Skipping {model_name} (limited mode).")
            continue
        
        try:
            with open(fp, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    cue = entry.get('cue', '').lower().strip()
                    if cue not in cue_to_idx: continue
                        
                    raw_resps = entry.get('responses', [])
                    cleaned_resps = []
                    for r in raw_resps:
                        txt = r.get('response', '') if isinstance(r, dict) else str(r)
                        txt = ''.join([c for c in txt.lower() if c.isalnum() or c.isspace()]).strip()
                        if txt in vocab_set:
                            cleaned_resps.append(txt)
                            
                    for cr in cleaned_resps:
                        data_rows.append((cue_to_idx[cue], response_to_idx[cr]))
            
            if not data_rows: continue
                
            df_counts = pd.DataFrame(data_rows, columns=['row', 'col'])
            df_counts = df_counts.groupby(['row', 'col']).size().reset_index(name='count')
            
            mat = csr_matrix((df_counts['count'], (df_counts['row'], df_counts['col'])),
                             shape=(num_cues, num_responses))
            matrices[f"active_{model_name}"] = mat
            print(f"[Active] Processed {model_name} ({mat.sum()} tokens)")
            
        except Exception as e:
            print(f"[Active] Error processing {model_name}: {e}")

    return matrices

# =============================================================================
# TRANSFORMATION & SANITIZATION
# =============================================================================

def calculate_ppmi(matrix: csr_matrix, smooth: float = 1e-10) -> csr_matrix:
    total_sum = matrix.sum()
    if total_sum == 0: return matrix
    
    row_sums = np.asarray(matrix.sum(axis=1)).squeeze()
    col_sums = np.asarray(matrix.sum(axis=0)).squeeze()
    
    rows, cols = matrix.nonzero()
    data = matrix.data
    
    denom = (row_sums[rows] * col_sums[cols]) + smooth
    pmi_values = np.log2((data * total_sum) / denom)
    ppmi_values = np.maximum(0, pmi_values)
    
    return csr_matrix((ppmi_values, (rows, cols)), shape=matrix.shape)

def get_target_dim(key: str, dim_lookup: dict, strategy: str, fallback_dim: int) -> int:
    """Return target embedding dimension for a given matrix key."""
    if key == 'human_matrix':
        return 300

    model_name = extract_model_from_key(key)

    if strategy == "activation" and model_name in dim_lookup:
        return dim_lookup[model_name]

    if model_name in MODEL_TARGET_DIMS:
        return MODEL_TARGET_DIMS[model_name]

    if strategy == "fixed":
        return fallback_dim

    print(f"    WARNING: No dimension match found for '{key}'. Using fallback {fallback_dim}.")
    return fallback_dim

def derive_dense_embeddings(matrices: dict, dim_lookup: dict, strategy: str, fallback_dim: int) -> dict:
    dense_embeddings = {}
    
    for key, mat in matrices.items():
        if key == 'mappings': continue
        
        target_dim = get_target_dim(key, dim_lookup, strategy, fallback_dim)
        print(f"  - Transforming {key} -> Target Dim: {target_dim}")
        
        ppmi = calculate_ppmi(mat.astype(np.float64))
        
        min_matrix_dim = min(ppmi.shape)
        k = min(target_dim, min_matrix_dim - 1)
        
        if k < 2:
            print(f"    WARNING: Matrix too small for SVD (dim={min_matrix_dim}). Returning Zeros.")
            embeddings = np.zeros((ppmi.shape[0], target_dim))
        else:
            if k < target_dim:
                print(f"    NOTE: Matrix rank limit ({min_matrix_dim}) < Target ({target_dim}). Using k={k}.")
            
            # --- SMART SOLVER SELECTION ---
            # If k > 500, avoid 'svds' (ARPACK) as it hangs on large k/n ratios.
            # Use Randomized SVD for speed.
            if k > 500:
                print(f"    [Speed] Large k ({k}) detected. Using Randomized SVD directly.")
                U, Sigma, VT = randomized_svd(ppmi, n_components=k, random_state=42)
                embeddings = U * np.sqrt(Sigma)
            else:
                try:
                    U, Sigma, VT = svds(ppmi, k=k)
                    idx = np.argsort(Sigma)[::-1]
                    U, Sigma = U[:, idx], Sigma[idx]
                    embeddings = U * np.sqrt(Sigma)
                except Exception as e:
                    print(f"    SVD Failed ({e}). Falling back to randomized SVD.")
                    U, Sigma, VT = randomized_svd(ppmi, n_components=k, random_state=42)
                    embeddings = U * np.sqrt(Sigma)
            
            # Pad with zeros if we couldn't reach the target dimension
            if k < target_dim:
                padding = np.zeros((embeddings.shape[0], target_dim - k))
                embeddings = np.hstack([embeddings, padding])

        dense_embeddings[key] = np.nan_to_num(embeddings, nan=0.0)
        
    return dense_embeddings

def align_and_sanitize_rows(final_embeddings: dict, mappings: dict, eps: float = 1e-12):
    keys = list(final_embeddings.keys())
    if not keys: return final_embeddings, mappings
    
    print("\n[Sanitize] Aligning rows across all embeddings...")
    n_rows = final_embeddings[keys[0]].shape[0]
    keep_mask = np.ones(n_rows, dtype=bool)
    
    for k in keys:
        mat = final_embeddings[k]
        is_finite = np.isfinite(mat).all(axis=1)
        # Relax norm check for sparse high-dim
        has_norm = np.linalg.norm(mat, axis=1) > eps
        keep_mask &= (is_finite & has_norm)
        
    dropped_count = n_rows - keep_mask.sum()
    print(f"  - Keeping {keep_mask.sum()} / {n_rows} rows. Dropping {dropped_count}.")
    
    for k in keys:
        final_embeddings[k] = final_embeddings[k][keep_mask]
        
    old_idx_to_cue = mappings['idx_to_cue']
    valid_indices = np.where(keep_mask)[0]
    new_idx_to_cue = {new_i: old_idx_to_cue[old_i] for new_i, old_i in enumerate(valid_indices)}
    new_cue_to_idx = {cue: new_i for new_i, cue in new_idx_to_cue.items()}
    
    new_mappings = {
        'cue_to_idx': new_cue_to_idx,
        'idx_to_cue': new_idx_to_cue,
        'response_to_idx': mappings['response_to_idx']
    }
    
    return final_embeddings, new_mappings

# =============================================================================
# MAIN
# =============================================================================

def main():
    default_swow, default_passive, default_deranged, default_active, default_out = infer_default_paths()
    default_activation_pkl = default_out / "activation_embeddings.pkl"

    parser = argparse.ArgumentParser(description="Build Behavioral Vectors (Matched Dims + 300d variants + Human).")
    parser.add_argument('--swow_path', type=Path, default=default_swow)
    parser.add_argument('--passive_dir', type=Path, default=default_passive)
    parser.add_argument('--deranged_dir', type=Path, default=default_deranged)
    parser.add_argument('--active_dir', type=Path, default=default_active)
    parser.add_argument('--output_dir', type=Path, default=default_out)
    parser.add_argument('--limited_mode', action='store_true',
                        help="Process only the core set of models.")
    parser.add_argument('--dim_strategy', choices=['activation', 'mapping', 'fixed'], default='activation',
                        help="activation: match dims from activation_embeddings.pkl; "
                             "mapping: use MODEL_TARGET_DIMS; fixed: use --fixed_dim.")
    parser.add_argument('--activation_pkl', type=Path, default=default_activation_pkl,
                        help="Path to activation embeddings pickle for --dim_strategy=activation.")
    parser.add_argument('--fixed_dim', type=int, default=DEFAULT_FALLBACK_DIM,
                        help="Target dim when using --dim_strategy=fixed or as general fallback.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    allowed_models = LIMITED_MODE_MODELS if args.limited_mode else None
    if allowed_models is not None:
        print("[Mode] Limited mode enabled. Only processing human plus core models.")
    else:
        print("[Mode] Full mode enabled. Processing all available models.")

    dim_lookup = {}
    if args.dim_strategy == "activation":
        dim_lookup = load_activation_dims(args.activation_pkl)
        if not dim_lookup:
            print(f"[Dims] No activation dims found. Falling back to MODEL_TARGET_DIMS/fixed_dim={args.fixed_dim}.")
    elif args.dim_strategy == "mapping":
        print("[Dims] Using MODEL_TARGET_DIMS mapping for target dimensions.")
    else:
        print(f"[Dims] Using fixed dimension: {args.fixed_dim}")
    
    # 1. Vocab & Human Matrix
    human_df, mappings, vocab_set = load_human_swow(args.swow_path, min_freq=MIN_FREQ_THRESHOLD)
    
    row = human_df['cue'].map(mappings['cue_to_idx']).values
    col = human_df['response_word'].map(mappings['response_to_idx']).values
    counts = pd.DataFrame({'row': row, 'col': col}).groupby(['row', 'col']).size().reset_index(name='c')
    human_mat = csr_matrix((counts['c'], (counts['row'], counts['col'])), 
                           shape=(len(mappings['cue_to_idx']), len(mappings['response_to_idx'])))
    
    # Add Human Matrix to the set (it will be transformed to 300d)
    matrices = {'human_matrix': human_mat}
    
    # 2. Ingest Data
    matrices.update(process_passive_behavior(args.passive_dir, args.deranged_dir, mappings, vocab_set,
                                             allowed_models=allowed_models))
    matrices.update(process_active_generation(args.active_dir, mappings, vocab_set,
                                              allowed_models=allowed_models))
    
    if not matrices:
        print("WARNING: No matrices created.")
        return

    # 3. Transform
    print("\n[Transformation] Applying PPMI + SVD...")
    # Matched-to-activation embeddings
    dense_matched = derive_dense_embeddings(matrices, dim_lookup, "activation", args.fixed_dim)
    # Fixed 300d embeddings (for behavioral comparisons)
    dense_fixed = derive_dense_embeddings(matrices, {}, "fixed", 300)
    dense_fixed = {k if k == 'human_matrix' else f"{k}_300d": v for k, v in dense_fixed.items()}
    
    # Merge and sanitize
    merged = {}
    merged.update(dense_matched)
    merged.update(dense_fixed)
    
    merged, mappings = align_and_sanitize_rows(merged, mappings)
    
    export_payload = {
        'embeddings': merged,
        'mappings': mappings
    }
    
    out_path = args.output_dir / "behavioral_embeddings.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(export_payload, f)
        
    print(f"\n[Success] Saved {len(merged)} aligned matrices to {out_path}")

if __name__ == "__main__":
    main()
