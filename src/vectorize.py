"""
src/vectorize.py

The "Feature Factory" for Behavioral Representations.
This script converts raw model outputs into standardized Semantic Matrices.

Inputs:
1. Human SWOW Data (Ground Truth for Vocab/Rows)
2. Passive Behavior: Log-probability CSVs (outputs/raw_behavior/model_swow_logprobs)
3. Active Behavior: Generated Association JSONLs (outputs/raw_behavior/model_swow)

Outputs:
- A pickled dictionary containing:
  - 'mappings': {cue_to_idx, idx_to_cue}
  - 'human_matrix': Sparse Matrix (Counts)
  - 'model_X_passive': Dense Matrix (SVD of PPMI of Logprobs)
  - 'model_Y_active': Dense Matrix (SVD of PPMI of Generated Counts)
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
from scipy.sparse.linalg import svds
import warnings
import ast
from joblib import Parallel, delayed

# --- CONSTANTS ---
ROW_NORM_EPS = 1e-12
DEFAULT_N_COMPONENTS = 300
MIN_FREQ_THRESHOLD = 5

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_human_swow(human_csv_path: Path, min_freq: int, verbose: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    """
    Load SWOW to define the 'Canonical Vocabulary'.
    Returns:
        - df_filtered: The human data used for counts.
        - mappings: dict containing cue and response maps.
        - vocab_set: set of allowed response words.
    """
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
    if verbose:
        print(f"  > Sample Cues: {all_cues[:5]}")
        print(f"  > Sample Responses: {all_responses[:5]}")
    return df_filtered, mappings, valid_words


def _process_single_passive(fp: Path, cue_to_idx: dict, response_to_idx: dict, vocab_set: set, allowed_models: list = None, verbose: bool = False, allow_deranged: bool = False, existing_keys: set = None):
    """Helper for parallel processing of passive logprobs."""
    model_name = fp.stem
    if not allow_deranged and "deranged" in model_name: 
        return None
    if allowed_models:
            return None

    # Append Check
    if existing_keys:
        # Expected Standard Key
        std_key = f"passive_{model_name}_300d"
        # Expected Contrastive Key (if we are building deranged, we assume we want contrastive)
        # Construct Contrastive Key: 'passive_contrastive_{stem}_300d'
        # If fp is actually a deranged file, we handle it differently (skipped unless allow_deranged)
        
        # If this is a DERANGED file (allow_deranged=True)
        if allow_deranged:
            # Deranged files have stems like: 'model-deranged'

            real_stem = model_name.replace('-deranged', '').replace('_deranged', '')
            contr_key = f"passive_contrastive_{real_stem}_300d"
            if contr_key in existing_keys:
                if verbose:
                     print(f"  [Skip] {model_name} (Contrastive Result {contr_key} exists)")
                return None
                
        else:
            # Real File
            # We need to process if:
            # 1. Standard result is missing.
            # 2. Contrastive result is missing (and we intend to generate it).
            
            contr_key = f"passive_contrastive_{model_name}_300d"
            
            # If standard exists, we don't need to re-calc standard.
            # If contrastive exists, we don't need re-calc contrastive.
            # If BOTH exist, we can skip.
            # If Standard exists but Contrastive MISSING -> We yield matrix (so contrastive step can pick it up).
            # If Standard MISSING -> We yield matrix.
            
            if std_key in existing_keys and contr_key in existing_keys:
                 if verbose: print(f"  [Skip] {model_name} (Results exist)")
                 return None

            
    try:
        df = pd.read_csv(fp)
        cols = set(df.columns)
        if not {'cue', 'response_set', 'normalized_log_prob'}.issubset(cols):
            return None
            
        df['cue'] = df['cue'].astype(str).str.lower().str.strip()
        df['response_set'] = df['response_set'].astype(str).str.lower().str.split(',')
        df = df.explode('response_set')
        df['response_set'] = df['response_set'].astype(str).str.strip()
        
        initial_rows = len(df)
        mask = (df['cue'].isin(cue_to_idx)) & (df['response_set'].isin(vocab_set))
        df = df[mask].copy()
        dropped = initial_rows - len(df)
        
        if df.empty:
            print(f"[Passive] {model_name}: No overlap with human vocab. (Dropped {dropped}/{initial_rows} rows)")
            return None
        
        if verbose:
            print(f"  > {model_name}: Kept {len(df)}/{initial_rows} rows ({len(df)/initial_rows:.1%}). Dropped {dropped}.")
            if len(df) > 0:
                print(f"  > Sample Logprobs:\n{df[['cue', 'response_set', 'normalized_log_prob']].head(3)}")

        row_idx = df['cue'].map(cue_to_idx).values
        col_idx = df['response_set'].map(response_to_idx).values
        
        df['row'] = row_idx
        df['col'] = col_idx
        
        agg = df.groupby(['row', 'col'])['normalized_log_prob'].mean().reset_index()
        agg['score'] = np.exp(agg['normalized_log_prob'])
        row_sums = agg.groupby('row')['score'].transform('sum')
        agg['prob'] = np.where(row_sums > 0, agg['score'] / row_sums, 0.0)
        
        num_cues = len(cue_to_idx)
        num_responses = len(response_to_idx)
        mat = csr_matrix((agg['prob'], (agg['row'], agg['col'])), 
                         shape=(num_cues, num_responses))
        
        print(f"[Passive] Processed {model_name} ({mat.count_nonzero()} entries)")
        return f"passive_{model_name}", mat
        
    except Exception as e:
        print(f"[Passive] Error processing {model_name}: {e}")
        return None

def process_passive_logprobs(input_dir: Path, mappings: dict, vocab_set: set, allowed_models: list = None, verbose: bool = False, n_jobs: int = 1, allow_deranged: bool = False, existing_keys: set = None) -> dict:
    """
    Ingest 'Passive' CSVs (Logprobs).
    Logic: LogProb -> Exp -> Normalize -> Sparse Matrix.
    """
    if not input_dir.exists():
        print(f"[Passive] Directory not found: {input_dir}")
        return {}
        
    cue_to_idx = mappings['cue_to_idx']
    response_to_idx = mappings['response_to_idx']
    
    files = sorted(list(input_dir.glob('*.csv')))
    print(f"[Passive] Found {len(files)} logprob files. Processing with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_passive)(fp, cue_to_idx, response_to_idx, vocab_set, allowed_models, verbose, allow_deranged, existing_keys)
        for fp in files
    )
    
    matrices = {k: v for r in results if r for k, v in [r]}
    return matrices


def _process_single_active(fp: Path, cue_to_idx: dict, response_to_idx: dict, vocab_set: set, allowed_models: list = None, verbose: bool = False, existing_keys: set = None):
    """Helper for parallel processing of active generation."""
    model_name = fp.stem
    if allowed_models:
        if not any(m in model_name for m in allowed_models):
            return None
            
    if existing_keys:
         std_key = f"active_{model_name}_300d"
         if std_key in existing_keys:
             if verbose: print(f"  [Skip] {model_name} (Result {std_key} exists)")
             return None
            
    data_rows = []
    try:
        with open(fp, 'r') as f:
            for line in f:
                entry = json.loads(line)
                cue = entry.get('cue', '').lower().strip()
                
                if cue not in cue_to_idx:
                    continue
                    
                raw_resps = entry.get('responses', [])
                cleaned_resps = []
                for r in raw_resps:
                    txt = r.get('response', '') if isinstance(r, dict) else str(r)
                    txt = ''.join([c for c in txt.lower() if c.isalnum() or c.isspace()]).strip()
                    if txt in vocab_set:
                        cleaned_resps.append(txt)
                        
                for cr in cleaned_resps:
                    data_rows.append((cue_to_idx[cue], response_to_idx[cr]))
        
        if not data_rows:
            print(f"[Active] {model_name}: No valid responses found.")
            return None
            
        df_counts = pd.DataFrame(data_rows, columns=['row', 'col'])
        df_counts = df_counts.groupby(['row', 'col']).size().reset_index(name='count')
        
        num_cues = len(cue_to_idx)
        num_responses = len(response_to_idx)
        mat = csr_matrix((df_counts['count'], (df_counts['row'], df_counts['col'])),
                         shape=(num_cues, num_responses))
        
        print(f"[Active] Processed {model_name} ({mat.sum()} total tokens)")
        if verbose:
            # Reconstruct sample from last entry for logging (approximate)
            print(f"  > Sample Generated Responses (Last Entry): {cleaned_resps[:5] if 'cleaned_resps' in locals() else 'None'}")
            
        return f"active_{model_name}", mat
        
    except Exception as e:
        print(f"[Active] Error processing {model_name}: {e}")
        return None

def process_active_generation(input_dir: Path, mappings: dict, vocab_set: set, allowed_models: list = None, verbose: bool = False, n_jobs: int = 1, existing_keys: set = None) -> dict:
    """
    Ingest 'Active' JSONLs (Generated Text).
    Logic: Raw Text -> Count -> Normalize -> Sparse Matrix.
    """
    if not input_dir.exists():
        print(f"[Active] Directory not found: {input_dir}")
        return {}

    cue_to_idx = mappings['cue_to_idx']
    response_to_idx = mappings['response_to_idx']

    files = sorted(list(input_dir.glob('*.jsonl')))
    print(f"[Active] Found {len(files)} generation files. Processing with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_active)(fp, cue_to_idx, response_to_idx, vocab_set, allowed_models, verbose, existing_keys)
        for fp in files
    )
    
    matrices = {k: v for r in results if r for k, v in [r]}
    return matrices

# =============================================================================
# ACTIVATIONS (Raw Dense Vectors)
# =============================================================================

def _process_single_activation(fp: Path, cue_to_idx: dict, allowed_models: list = None, verbose: bool = False, existing_keys: set = None):
    """Helper for parallel processing of activations."""
    model_name = fp.stem
    if allowed_models:
        if not any(m in model_name for m in allowed_models):
            return None
            
    clean_name = model_name.replace('_embeddings', '')
    key = f"activation_{clean_name}"
    
    if existing_keys and key in existing_keys:
         # For activations, the Raw key IS the final key.
         if verbose: print(f"  [Skip] {model_name} (Result {key} exists)")
         return None
    
    print(f"  - Processing {model_name}...")
    
    try:
        df = pd.read_csv(fp, low_memory=False)
        cols = list(df.columns)
        if not cols:
            return None
            
        cue_col = cols[0]
        df[cue_col] = df[cue_col].astype(str).str.lower().str.strip()
        
        sample_val = df.iloc[0, 1] if len(df.columns) > 1 else None
        data_map = {}
        dim = 0
        
        if isinstance(sample_val, str) and str(sample_val).strip().startswith("["):
            def parse_vec(x):
                try:
                    return np.array(ast.literal_eval(str(x)), dtype=np.float32)
                except:
                    return None
            df['vec'] = df.iloc[:, 1].apply(parse_vec)
            df = df.dropna(subset=['vec'])
            if df.empty:
                print(f"    [Warning] No valid vectors found in {model_name}")
                return None
            data_map = dict(zip(df[cue_col], df['vec']))
            dim = len(df['vec'].iloc[0])
        else:
            vec_data = df.iloc[:, 1:].values.astype(np.float32)
            data_map = dict(zip(df[cue_col], vec_data))
            dim = vec_data.shape[1]
        
        n_cues = len(cue_to_idx)
        matrix = np.zeros((n_cues, dim), dtype=np.float32)
        hit_count = 0
        
        for cue, idx in cue_to_idx.items():
            if cue in data_map:
                vec = data_map[cue]
                if vec.shape[0] == dim:
                    matrix[idx] = vec
                    hit_count += 1
        
        # Sanity Checks
        non_zeros = matrix[matrix != 0]
        if len(non_zeros) == 0:
            print(f"    [Warning] {key}: Matrix is ALL ZEROS.")
        elif np.isnan(matrix).any():
            print(f"    [Warning] {key}: Matrix contains NaNs.")
            matrix = np.nan_to_num(matrix)
        
        print(f"    > Processed {key}: {hit_count}/{n_cues} coverage, {dim} dims.")
        if verbose and len(non_zeros) > 0:
             print(f"    > Stats: Mean={non_zeros.mean():.4f}, Std={non_zeros.std():.4f}, Min={non_zeros.min():.4f}, Max={non_zeros.max():.4f}")
        
        return key, matrix
        
    except Exception as e:
        print(f"    [Error] Failed to process {model_name}: {e}")
        return None

def process_activations(input_dir: Path, mappings: dict, allowed_models: list = None, verbose: bool = False, n_jobs: int = 1, existing_keys: set = None) -> dict:
    """
    Ingest 'Activation' CSVs (Raw Dense Vectors).
    Logic: Raw Vector -> Align to Cue Index -> Dense Matrix.
    """
    if not input_dir.exists():
        print(f"[Activations] Directory not found: {input_dir}")
        return {}

    cue_to_idx = mappings['cue_to_idx']
    
    files = sorted(list(input_dir.glob('*.csv')))
    print(f"[Activations] Found {len(files)} activation files. Processing with n_jobs={n_jobs}...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_activation)(fp, cue_to_idx, allowed_models, verbose, existing_keys)
        for fp in files
    )
    
    matrices = {k: v for r in results if r for k, v in [r]}
    return matrices

# =============================================================================
# TRANSFORMATION (PPMI + SVD)
# =============================================================================

def calculate_ppmi(matrix: csr_matrix, smooth: float = 1e-10) -> csr_matrix:
    """Calculate Positive Pointwise Mutual Information."""
    total_sum = matrix.sum()
    if total_sum == 0: return matrix

    row_sums = np.asarray(matrix.sum(axis=1)).squeeze()
    col_sums = np.asarray(matrix.sum(axis=0)).squeeze()

    rows, cols = matrix.nonzero()
    data = matrix.data

    denom = (row_sums[rows] * col_sums[cols]) + smooth
    pmi_values = np.log2((data * total_sum) / denom)
    
    # Clip negative values (PPMI)
    ppmi_values = np.maximum(0, pmi_values)
    
    return csr_matrix((ppmi_values, (rows, cols)), shape=matrix.shape)

def _process_single_transform(key: str, mat: csr_matrix, activation_dims: dict, verbose: bool):
    """Helper for parallel PPMI+SVD with Multi-Dim Output."""
    if key == 'mappings': return None
    
    # Check if Outputs Exist
    
    # Identify Model Name from Key
    # keys: 'passive_model', 'active_model', 'passive_contrastive_model'
    clean_name = key.replace('passive_', '').replace('active_', '').replace('contrastive_', '')
    
    configs = []
    
    # 1. Standard 300d
    configs.append((300, "_300d"))
    
    # 2. High Dim (if activation dim found)
    if clean_name in activation_dims:
        target_d = activation_dims[clean_name]
        if target_d != 300: # Avoid duplicate if by chance it's 300
             configs.append((target_d, f"_{target_d}d"))
             
    print(f"  - Transforming {key} -> {len(configs)} variants: {[c[1] for c in configs]}")

    # PPMI
    ppmi = calculate_ppmi(mat.astype(np.float64))
    min_dim = min(ppmi.shape)
    
    # Optimization: Run SVD once for max requested k
    max_k = max(c[0] for c in configs)
    actual_k = min(max_k, min_dim - 1)
    
    results = []
    
    if actual_k < 2:
         print(f"    WARNING: Matrix too small for SVD ({min_dim}). Returning Zeros.")
         for d, suffix in configs:
             results.append((f"{key}{suffix}", np.zeros((ppmi.shape[0], d))))
         return results

    try:
        U, Sigma, VT = svds(ppmi, k=actual_k)
        # Sort (svds returns increasing order)
        idx = np.argsort(Sigma)[::-1]
        U, Sigma = U[:, idx], Sigma[idx]
        
        for d, suffix in configs:
            eff_d = min(d, actual_k)
            
            U_slice = U[:, :eff_d]
            S_slice = Sigma[:eff_d]
            emb = U_slice * np.sqrt(S_slice)
            
            # Pad if rank deficient
            if emb.shape[1] < d:
                padding = d - emb.shape[1]
                emb = np.pad(emb, ((0,0), (0, padding)), mode='constant')
                
            results.append((f"{key}{suffix}", emb))
            
    except Exception as e:
        print(f"    [Warning] SVD Failed ({e}). Using randomized SVD fallback.")
        U, Sigma, VT = randomized_svd(ppmi, n_components=actual_k, random_state=42)
        
        for d, suffix in configs:
            eff_d = min(d, actual_k)
            U_slice = U[:, :eff_d]
            S_slice = Sigma[:eff_d]
            emb = U_slice * np.sqrt(S_slice)
            
            if emb.shape[1] < d:
                padding = d - emb.shape[1]
                emb = np.pad(emb, ((0,0), (0, padding)), mode='constant')
            
            results.append((f"{key}{suffix}", emb))

    # Sanitize
    sanitized = []
    for k_out, emb in results:
        emb = np.nan_to_num(emb, nan=0.0)
        sanitized.append((k_out, emb))
        if verbose:
             print(f"    > {k_out}: {emb.shape}")
             
    return sanitized

def derive_dense_embeddings(matrices: dict, activation_dims: dict, verbose: bool = False, n_jobs: int = 1) -> dict:
    """
    Convert Sparse Count/Prob Matrices -> PPMI -> SVD Dense Vectors (Multi-Dim).
    """
    print(f"[Transformation] Applying PPMI + SVD (All Variants) with n_jobs={n_jobs}...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_transform)(key, mat, activation_dims, verbose)
        for key, mat in matrices.items()
        if key != 'mappings'
    )
    
    # Flatten list of lists
    dense_embeddings = {}
    for sublist in results:
        if sublist:
            for k, v in sublist:
                dense_embeddings[k] = v
                
    return dense_embeddings

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Behavioral Vectors (Passive & Active).")
    parser.add_argument('--swow_path', type=Path, required=True, help="Path to Human SWOW CSV")
    parser.add_argument('--passive_dir', type=Path, required=True, help="Dir containing Logprob CSVs")
    parser.add_argument('--deranged_dir', type=Path, required=False, help="Dir containing Contrastive (Deranged) Logprob CSVs")
    parser.add_argument('--active_dir', type=Path, required=True, help="Dir containing Generated JSONLs")
    parser.add_argument('--activation_dir', type=Path, required=True, help="Dir containing Activation CSVs (Required for High-Dim)")
    parser.add_argument('--output_dir', type=Path, required=True, help="Dir to save output pickle")
    parser.add_argument('--models', nargs='*', help="List of model names to process (substring match)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of parallel jobs (-1 for all)")
    args = parser.parse_args()

    # 1. Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Existing & Mappings
    existing_embeddings = {}
    existing_keys = set()
    out_path = args.output_dir / "embeddings.pkl"
    
    if out_path.exists():
        print(f"[Append] Found existing pickle at {out_path}. Loading...")
        try:
            with open(out_path, 'rb') as f:
                data = pickle.load(f)
            existing_embeddings = data['embeddings']
            mappings = data['mappings']
            existing_keys = set(existing_embeddings.keys())
            
            # Reconstruct Human Vocab Set from Mappings (needed for filtering)
            vocab_set = set(mappings['response_to_idx'].keys())
            
            # If 'human_matrix' is in existing_embeddings, we can skip loading SWOW.
            if 'human_matrix' in existing_embeddings:
                 print("[Append] Human Matrix exists. Skipping SWOW Load.")
                 human_mat = existing_embeddings['human_matrix']
            else:
                 # load SWOW safely if needed.
                 print("[Append] Human Matrix missing. Reloading SWOW.")
                 human_df, _, _ = load_human_swow(args.swow_path, min_freq=MIN_FREQ_THRESHOLD, verbose=args.verbose)
                 # Reconstruct matrix
                 row = human_df['cue'].map(mappings['cue_to_idx']).values
                 col = human_df['response_word'].map(mappings['response_to_idx']).values
                 counts = pd.DataFrame({'row': row, 'col': col}).groupby(['row', 'col']).size().reset_index(name='c')
                 human_mat = csr_matrix((counts['c'], (counts['row'], counts['col'])), 
                            shape=(len(mappings['cue_to_idx']), len(mappings['response_to_idx'])))
                 
        except Exception as e:
            print(f"[Append] Error loading existing pickle: {e}. Starting Fresh.")
            existing_embeddings = {}
            existing_keys = set()
            out_path = args.output_dir / "embeddings.pkl" # Reset
            # Fallback to load fresh
            human_df, mappings, vocab_set = load_human_swow(args.swow_path, min_freq=MIN_FREQ_THRESHOLD, verbose=args.verbose)
            # Build Human
            row = human_df['cue'].map(mappings['cue_to_idx']).values
            col = human_df['response_word'].map(mappings['response_to_idx']).values
            counts = pd.DataFrame({'row': row, 'col': col}).groupby(['row', 'col']).size().reset_index(name='c')
            human_mat = csr_matrix((counts['c'], (counts['row'], counts['col'])), 
                                   shape=(len(mappings['cue_to_idx']), len(mappings['response_to_idx'])))
    else:
        # Fresh Start
        human_df, mappings, vocab_set = load_human_swow(args.swow_path, min_freq=MIN_FREQ_THRESHOLD, verbose=args.verbose)
        # Build Human
        row = human_df['cue'].map(mappings['cue_to_idx']).values
        col = human_df['response_word'].map(mappings['response_to_idx']).values
        counts = pd.DataFrame({'row': row, 'col': col}).groupby(['row', 'col']).size().reset_index(name='c')
        human_mat = csr_matrix((counts['c'], (counts['row'], counts['col'])), 
                               shape=(len(mappings['cue_to_idx']), len(mappings['response_to_idx'])))

    # 3. Ingest Data
    matrices = {}
    if 'human_matrix' not in existing_embeddings:
        matrices['human_matrix'] = human_mat
    
    # Passive
    matrices.update(process_passive_logprobs(args.passive_dir, mappings, vocab_set, allowed_models=args.models, verbose=args.verbose, n_jobs=args.n_jobs, existing_keys=existing_keys))
    
    # Contrastive (Joint Latent Space)
    # Logic: Contrastive = hstack([Real, Deranged])
    if args.deranged_dir:
        print("\n--- Processing Contrastive Data (Joint Latent Space) ---")
        deranged_mats = process_passive_logprobs(args.deranged_dir, mappings, vocab_set, allowed_models=args.models, verbose=args.verbose, n_jobs=args.n_jobs, allow_deranged=True, existing_keys=existing_keys)
        
        # 1. Identify common models
        
        real_keys = set(k for k in matrices.keys() if k.startswith("passive_") and not "contrastive" in k)
        
        for r_key in real_keys:
            # Match 'passive_{model}' with 'passive_{model}-deranged'
            # Note: Deranged files typically have the stem 'model-deranged'
            
            model_stem = r_key.replace("passive_", "")
            
            # Find matching deranged key
            d_key = None
            for candidate in deranged_mats.keys():
                if model_stem in candidate:
                    d_key = candidate
                    break
            
            if d_key:
                real_mat = matrices[r_key]
                deranged_mat = deranged_mats[d_key]
                
                # Check shapes
                if real_mat.shape[0] != deranged_mat.shape[0]:
                    print(f"  [Skip] Contrastive Pair {model_stem}: Row count mismatch ({real_mat.shape[0]} vs {deranged_mat.shape[0]})")
                    continue
                
                # HSTACK
                # Result shape: (N_Cues, N_Resp * 2)
                joint_mat = hstack([real_mat, deranged_mat])
                
                new_key = f"passive_contrastive_{model_stem}"
                matrices[new_key] = joint_mat
                print(f"  [Joint] Created {new_key} with shape {joint_mat.shape}")
                
            else:
                if args.verbose:
                    print(f"  [Info] No matching deranged data for {model_stem}")

    else:
        print("[Info] No deranged_dir provided. Skipping Contrastive Embeddings.")

    # Active
    matrices.update(process_active_generation(args.active_dir, mappings, vocab_set, allowed_models=args.models, verbose=args.verbose, n_jobs=args.n_jobs, existing_keys=existing_keys))
    
    # Activations (Raw) - These bypass PPMI/SVD
    # WE NEED THEM FOR DIMS
    activation_matrices = {}
    if args.activation_dir:
        activation_matrices = process_activations(args.activation_dir, mappings, allowed_models=args.models, verbose=args.verbose, n_jobs=args.n_jobs, existing_keys=existing_keys)
    else:
        print("[Error] Activation Dir required for Unified Vectorization (to determine High Dims).")
        return
    
    if len(matrices) == 1:
        print("WARNING: No model matrices created. Check input directories.")
    
    # Extract Dims (Combine New + Existing for checks?)
    
    activation_dims = {}
    
    # From New
    for k, mat in activation_matrices.items():
        m_name = k.replace('activation_', '')
        if hasattr(mat, "shape"): d = mat.shape[1]
        else: d = len(mat[0])
        activation_dims[m_name] = d
        
    # From Existing (if needed for a new transformation? unlikely but safe)
    for k, mat in existing_embeddings.items():
        if k.startswith('activation_'):
            m_name = k.replace('activation_', '')
            if m_name not in activation_dims:
                 if hasattr(mat, "shape"): d = mat.shape[1]
                 else: d = len(mat[0])
                 activation_dims[m_name] = d
    
    if args.verbose:
        print(f"[Dims] Known activation dims: {list(activation_dims.keys())}")

    # 4. Transform (PPMI -> SVD (300d + HighDim))
    dense_results = derive_dense_embeddings(matrices, activation_dims=activation_dims, verbose=args.verbose, n_jobs=args.n_jobs)
    
    # 5. Export
    # Merge dense results with raw activation matrices AND Existing
    final_embeddings = {**existing_embeddings, **dense_results, **activation_matrices}
    
    export_payload = {
        'embeddings': final_embeddings,
        'mappings': mappings
    }
    
    out_path = args.output_dir / "embeddings.pkl"
    with open(out_path, 'wb') as f: # Overwrite with the merged full dict
        pickle.dump(export_payload, f)
        
    print(f"\n[Success] Saved {len(final_embeddings)} matrices to {out_path}")

if __name__ == "__main__":
    try:
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        for candidate in (script_dir, *script_dir.parents):
            if (candidate / 'data').exists():
                project_root = candidate
                break
        
        # Defaults
        default_swow = project_root / 'data' / 'SWOW' / 'Human_SWOW-EN.R100.20180827.csv'
        default_passive = project_root / 'outputs' / 'raw_behavior' / 'model_swow_logprobs'
        default_active = project_root / 'outputs' / 'raw_behavior' / 'model_swow'
        default_activation = project_root / 'outputs' / 'raw_activations'
        default_out = project_root / 'outputs' / 'matrices'
    except:
        default_swow = Path('.')
        default_passive = Path('.')
        default_active = Path('.')
        default_activation = Path('.')
        default_out = Path('.')

    # Hack to allow running without args if paths match structure
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--swow_path', str(default_swow),
            '--passive_dir', str(default_passive),
            '--active_dir', str(default_active),
            '--activation_dir', str(default_activation),
            '--output_dir', str(default_out)
        ])
        
    main()
