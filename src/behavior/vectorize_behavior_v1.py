"""
src/behavior/vectorize_behavior.py

The "Feature Factory" for Behavioral Representations.
Constructs Semantic Matrices aligned with the "Contrastive" methodology.

Inputs:
1. Human SWOW Data (Ground Truth for Vocab/Rows)
2. Passive Real: Log-probability CSVs
3. Passive Deranged: Log-probability CSVs (Shuffled/Deranged)
4. Active Behavior: Generated Association JSONLs

Outputs:
- 'behavioral_embeddings.pkl' containing:
  - 'passive_{model}': Real SVD embedding
  - 'passive_{model}_contrast': Real+Deranged SVD embedding (Contrastive)
  - 'passive_{model}_shuffled': Deranged SVD embedding
  - 'active_{model}': Generated SVD embedding
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

# --- CONSTANTS ---
ROW_NORM_EPS = 1e-12
DEFAULT_N_COMPONENTS = 300
MIN_FREQ_THRESHOLD = 5

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def infer_default_paths():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parents[1] # Assuming src/behavior/script.py
    
    # Check if we are in the right structure
    if not (project_root / 'data').exists():
        # Fallback for different execution contexts
        project_root = Path.cwd()

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

def process_passive_behavior(real_dir: Path, deranged_dir: Path, mappings: dict, vocab_set: set) -> dict:
    """
    Ingest Passive Logprobs (Real) + Passive Logprobs (Deranged).
    Constructs: 
      1. Real Matrix
      2. Shuffled Matrix
      3. Contrast Matrix (hstack [Real | Shuffled])
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
    
    # 2. Index Deranged Files (Mapping normalized name -> path)
    deranged_map = {}
    if deranged_dir.exists():
        for f in deranged_dir.glob('*.csv'):
            norm_name = normalize_model_name(f.stem)
            deranged_map[norm_name] = f
            
    for fp in real_files:
        raw_name = fp.stem
        model_name = normalize_model_name(raw_name)
        
        # Skip deranged files if they accidentally sit in the real folder
        if "deranged" in model_name: continue

        try:
            # --- A. Process REAL ---
            df = pd.read_csv(fp)
            # Preprocess
            df['cue'] = df['cue'].astype(str).str.lower().str.strip()
            df['response_set'] = df['response_set'].astype(str).str.lower().str.split(',')
            df = df.explode('response_set')
            df['response_set'] = df['response_set'].astype(str).str.strip()
            
            # Filter
            mask = (df['cue'].isin(mappings['cue_to_idx'])) & (df['response_set'].isin(vocab_set))
            df = df[mask].copy()
            
            if df.empty:
                continue

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
                    
                    # Store Shuffled
                    matrices[f"passive_{model_name}_shuffled"] = mat_deranged
                    
                    # Store Contrastive (Horizontal Stack)
                    # Shape becomes (N_cues, 2 * N_responses)
                    mat_contrast = hstack([mat_real, mat_deranged], format='csr')
                    matrices[f"passive_{model_name}_contrast"] = mat_contrast
                    
                    print(f"[Passive] {model_name}: Real {mat_real.shape} | Contrast {mat_contrast.shape}")
                else:
                    print(f"[Passive] {model_name}: Deranged file found but empty after filter.")
            else:
                print(f"[Passive] {model_name}: No corresponding deranged file found.")
                
        except Exception as e:
            print(f"[Passive] Error processing {model_name}: {e}")
            
    return matrices

def process_active_generation(input_dir: Path, mappings: dict, vocab_set: set) -> dict:
    """Ingest Active Generated JSONLs."""
    if not input_dir.exists():
        return {}

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

def derive_dense_embeddings(matrices: dict, n_components: int = 300) -> dict:
    dense_embeddings = {}
    
    for key, mat in matrices.items():
        if key == 'mappings': continue
        
        print(f"  - Transforming {key}...")
        ppmi = calculate_ppmi(mat.astype(np.float64))
        
        min_dim = min(ppmi.shape)
        k = min(n_components, min_dim - 1)
        
        if k < 2:
            embeddings = np.zeros((ppmi.shape[0], n_components))
        else:
            try:
                U, Sigma, VT = svds(ppmi, k=k)
                idx = np.argsort(Sigma)[::-1]
                U, Sigma = U[:, idx], Sigma[idx]
                embeddings = U * np.sqrt(Sigma)
            except:
                U, Sigma, VT = randomized_svd(ppmi, n_components=k, random_state=42)
                embeddings = U * np.sqrt(Sigma)
                
        dense_embeddings[key] = np.nan_to_num(embeddings, nan=0.0)
        
    return dense_embeddings

def align_and_sanitize_rows(final_embeddings: dict, mappings: dict, eps: float = 1e-12):
    """
    Ensure all embeddings share the same valid rows.
    Drops rows that are zero-vectors in ANY of the embeddings.
    """
    keys = list(final_embeddings.keys())
    if not keys: return final_embeddings, mappings
    
    print("\n[Sanitize] Aligning rows across all embeddings...")
    
    # 1. Create a joint validity mask
    n_rows = final_embeddings[keys[0]].shape[0]
    keep_mask = np.ones(n_rows, dtype=bool)
    
    for k in keys:
        mat = final_embeddings[k]
        # Valid if finite AND has norm > epsilon
        is_finite = np.isfinite(mat).all(axis=1)
        has_norm = np.linalg.norm(mat, axis=1) > eps
        keep_mask &= (is_finite & has_norm)
        
    dropped_count = n_rows - keep_mask.sum()
    print(f"  - Keeping {keep_mask.sum()} / {n_rows} rows. Dropping {dropped_count}.")
    
    # 2. Filter Embeddings
    for k in keys:
        final_embeddings[k] = final_embeddings[k][keep_mask]
        
    # 3. Update Mappings
    old_idx_to_cue = mappings['idx_to_cue']
    valid_indices = np.where(keep_mask)[0]
    
    new_idx_to_cue = {new_i: old_idx_to_cue[old_i] for new_i, old_i in enumerate(valid_indices)}
    new_cue_to_idx = {cue: new_i for new_i, cue in new_idx_to_cue.items()}
    
    new_mappings = {
        'cue_to_idx': new_cue_to_idx,
        'idx_to_cue': new_idx_to_cue,
        'response_to_idx': mappings['response_to_idx'] # responses unaffected
    }
    
    return final_embeddings, new_mappings

# =============================================================================
# MAIN
# =============================================================================

def main():
    default_swow, default_passive, default_deranged, default_active, default_out = infer_default_paths()

    parser = argparse.ArgumentParser(description="Build Behavioral Vectors (Contrastive).")
    parser.add_argument('--swow_path', type=Path, default=default_swow, help="Path to Human SWOW CSV")
    parser.add_argument('--passive_dir', type=Path, default=default_passive, help="Real Logprobs")
    parser.add_argument('--deranged_dir', type=Path, default=default_deranged, help="Deranged Logprobs")
    parser.add_argument('--active_dir', type=Path, default=default_active, help="Generated JSONLs")
    parser.add_argument('--output_dir', type=Path, default=default_out, help="Output Dir")
    parser.add_argument('--n_components', type=int, default=300, help="SVD Dimensions")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Vocab & Human Matrix
    human_df, mappings, vocab_set = load_human_swow(args.swow_path, min_freq=MIN_FREQ_THRESHOLD)
    
    # Build Human CSR (Counts)
    row = human_df['cue'].map(mappings['cue_to_idx']).values
    col = human_df['response_word'].map(mappings['response_to_idx']).values
    counts = pd.DataFrame({'row': row, 'col': col}).groupby(['row', 'col']).size().reset_index(name='c')
    human_mat = csr_matrix((counts['c'], (counts['row'], counts['col'])), 
                           shape=(len(mappings['cue_to_idx']), len(mappings['response_to_idx'])))
    
    matrices = {'human_matrix': human_mat}
    
    # 2. Ingest Data (Passive Real + Contrastive)
    matrices.update(process_passive_behavior(args.passive_dir, args.deranged_dir, mappings, vocab_set))
    
    # 3. Ingest Active
    matrices.update(process_active_generation(args.active_dir, mappings, vocab_set))
    
    if len(matrices) == 1:
        print("WARNING: No model matrices created.")
        return

    # 4. Transform (PPMI -> SVD)
    print("\n[Transformation] Applying PPMI + SVD...")
    dense_results = derive_dense_embeddings(matrices, n_components=args.n_components)
    
    # 5. Sanitize (Ensure row alignment)
    dense_results, mappings = align_and_sanitize_rows(dense_results, mappings)
    
    # 6. Export
    export_payload = {
        'embeddings': dense_results,
        'mappings': mappings
    }
    
    out_path = args.output_dir / "behavioral_embeddings.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(export_payload, f)
        
    print(f"\n[Success] Saved {len(dense_results)} aligned matrices to {out_path}")

if __name__ == "__main__":
    main()