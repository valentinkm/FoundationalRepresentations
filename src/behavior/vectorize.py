"""
src/behavior/vectorize.py

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
import warnings

# --- CONSTANTS ---
ROW_NORM_EPS = 1e-12
DEFAULT_N_COMPONENTS = 300
MIN_FREQ_THRESHOLD = 5

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_human_swow(human_csv_path: Path, min_freq: int) -> tuple[pd.DataFrame, dict, dict]:
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
    return df_filtered, mappings, valid_words


def process_passive_logprobs(input_dir: Path, mappings: dict, vocab_set: set) -> dict:
    """
    Ingest 'Passive' CSVs (Logprobs).
    Logic: LogProb -> Exp -> Normalize -> Sparse Matrix.
    """
    if not input_dir.exists():
        print(f"[Passive] Directory not found: {input_dir}")
        return {}
        
    matrices = {}
    cue_to_idx = mappings['cue_to_idx']
    response_to_idx = mappings['response_to_idx']
    num_cues = len(cue_to_idx)
    num_responses = len(response_to_idx)
    
    files = sorted(list(input_dir.glob('*.csv')))
    print(f"[Passive] Found {len(files)} logprob files.")

    for fp in files:
        model_name = fp.stem
        # Skip deranged files if they are mixed in (optional check)
        if "deranged" in model_name: 
            continue
            
        try:
            df = pd.read_csv(fp)
            # Basic validation
            cols = set(df.columns)
            if not {'cue', 'response_set', 'normalized_log_prob'}.issubset(cols):
                continue
                
            # Preprocess
            df['cue'] = df['cue'].astype(str).str.lower().str.strip()
            df['response_set'] = df['response_set'].astype(str).str.lower().str.split(',')
            df = df.explode('response_set')
            df['response_set'] = df['response_set'].astype(str).str.strip()
            
            # Filter to Canon Vocab
            mask = (df['cue'].isin(cue_to_idx)) & (df['response_set'].isin(vocab_set))
            df = df[mask].copy()
            
            if df.empty:
                print(f"[Passive] {model_name}: No overlap with human vocab.")
                continue

            # Map to Indices
            row_idx = df['cue'].map(cue_to_idx).values
            col_idx = df['response_set'].map(response_to_idx).values
            
            # Aggregate Duplicates (avg logprob)
            df['row'] = row_idx
            df['col'] = col_idx
            
            agg = df.groupby(['row', 'col'])['normalized_log_prob'].mean().reset_index()
            
            # Convert to Probability Distribution
            agg['score'] = np.exp(agg['normalized_log_prob'])
            row_sums = agg.groupby('row')['score'].transform('sum')
            agg['prob'] = np.where(row_sums > 0, agg['score'] / row_sums, 0.0)
            
            # Build CSR
            mat = csr_matrix((agg['prob'], (agg['row'], agg['col'])), 
                             shape=(num_cues, num_responses))
            
            matrices[f"passive_{model_name}"] = mat
            print(f"[Passive] Processed {model_name} ({mat.count_nonzero()} entries)")
            
        except Exception as e:
            print(f"[Passive] Error processing {model_name}: {e}")
            
    return matrices


def process_active_generation(input_dir: Path, mappings: dict, vocab_set: set) -> dict:
    """
    Ingest 'Active' JSONLs (Generated Text).
    Logic: Raw Text -> Count -> Normalize -> Sparse Matrix.
    """
    if not input_dir.exists():
        print(f"[Active] Directory not found: {input_dir}")
        return {}

    matrices = {}
    cue_to_idx = mappings['cue_to_idx']
    response_to_idx = mappings['response_to_idx']
    num_cues = len(cue_to_idx)
    num_responses = len(response_to_idx)

    files = sorted(list(input_dir.glob('*.jsonl')))
    print(f"[Active] Found {len(files)} generation files.")

    for fp in files:
        model_name = fp.stem
        data_rows = []
        
        try:
            with open(fp, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    cue = entry.get('cue', '').lower().strip()
                    
                    if cue not in cue_to_idx:
                        continue
                        
                    # Extract responses (handling the structure from generate.py)
                    # Expected: "responses": [{"response": "word"}, ...] or list of strings
                    raw_resps = entry.get('responses', [])
                    
                    cleaned_resps = []
                    for r in raw_resps:
                        # Handle if r is dict or string
                        txt = r.get('response', '') if isinstance(r, dict) else str(r)
                        # Simple cleaning: lowercase, strip punctuation
                        txt = ''.join([c for c in txt.lower() if c.isalnum() or c.isspace()]).strip()
                        if txt in vocab_set:
                            cleaned_resps.append(txt)
                            
                    for cr in cleaned_resps:
                        data_rows.append((cue_to_idx[cue], response_to_idx[cr]))
            
            if not data_rows:
                print(f"[Active] {model_name}: No valid responses found.")
                continue
                
            # Build Counts
            df_counts = pd.DataFrame(data_rows, columns=['row', 'col'])
            df_counts = df_counts.groupby(['row', 'col']).size().reset_index(name='count')
            
            # Build CSR
            mat = csr_matrix((df_counts['count'], (df_counts['row'], df_counts['col'])),
                             shape=(num_cues, num_responses))
            
            matrices[f"active_{model_name}"] = mat
            print(f"[Active] Processed {model_name} ({mat.sum()} total tokens)")
            
        except Exception as e:
            print(f"[Active] Error processing {model_name}: {e}")

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

def derive_dense_embeddings(matrices: dict, n_components: int = 300) -> dict:
    """
    Convert Sparse Count/Prob Matrices -> PPMI -> SVD Dense Vectors.
    """
    dense_embeddings = {}
    
    for key, mat in matrices.items():
        if key == 'mappings': continue # Skip metadata
        
        print(f"  - Transforming {key}...")
        
        # 1. PPMI
        ppmi = calculate_ppmi(mat.astype(np.float64))
        
        # 2. SVD
        min_dim = min(ppmi.shape)
        k = min(n_components, min_dim - 1)
        
        if k < 2:
            print(f"    WARNING: Matrix too small for SVD ({min_dim}). Returning Zeros.")
            embeddings = np.zeros((ppmi.shape[0], n_components))
        else:
            try:
                U, Sigma, VT = svds(ppmi, k=k)
                # Sort components by singular value (svds doesn't guarantee order)
                idx = np.argsort(Sigma)[::-1]
                U, Sigma = U[:, idx], Sigma[idx]
                embeddings = U * np.sqrt(Sigma)
            except Exception as e:
                print(f"    SVD Failed ({e}). Using randomized SVD.")
                U, Sigma, VT = randomized_svd(ppmi, n_components=k, random_state=42)
                embeddings = U * np.sqrt(Sigma)
                
        # 3. Sanitize
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        dense_embeddings[key] = embeddings
        
    return dense_embeddings

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Behavioral Vectors (Passive & Active).")
    parser.add_argument('--swow_path', type=Path, required=True, help="Path to Human SWOW CSV")
    parser.add_argument('--passive_dir', type=Path, required=True, help="Dir containing Logprob CSVs")
    parser.add_argument('--active_dir', type=Path, required=True, help="Dir containing Generated JSONLs")
    parser.add_argument('--output_dir', type=Path, required=True, help="Dir to save output pickle")
    parser.add_argument('--n_components', type=int, default=300, help="SVD Dimensions")
    args = parser.parse_args()

    # 1. Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Build Human Matrix (The Ground Truth for Vocabulary)
    human_df, mappings, vocab_set = load_human_swow(args.swow_path, min_freq=MIN_FREQ_THRESHOLD)
    
    # Build Human CSR
    # Re-using logic: build count matrix
    row = human_df['cue'].map(mappings['cue_to_idx']).values
    col = human_df['response_word'].map(mappings['response_to_idx']).values
    counts = pd.DataFrame({'row': row, 'col': col}).groupby(['row', 'col']).size().reset_index(name='c')
    human_mat = csr_matrix((counts['c'], (counts['row'], counts['col'])), 
                           shape=(len(mappings['cue_to_idx']), len(mappings['response_to_idx'])))
    
    # 3. Ingest Data
    matrices = {'human_matrix': human_mat}
    
    # Passive
    matrices.update(process_passive_logprobs(args.passive_dir, mappings, vocab_set))
    
    # Active
    matrices.update(process_active_generation(args.active_dir, mappings, vocab_set))
    
    if len(matrices) == 1:
        print("WARNING: No model matrices created. Check input directories.")
    
    # 4. Transform (PPMI -> SVD)
    print("\n[Transformation] Applying PPMI + SVD...")
    dense_results = derive_dense_embeddings(matrices, n_components=args.n_components)
    
    # 5. Export
    export_payload = {
        'embeddings': dense_results,
        'mappings': mappings
    }
    
    out_path = args.output_dir / "behavioral_embeddings.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(export_payload, f)
        
    print(f"\n[Success] Saved {len(dense_results)} matrices to {out_path}")

if __name__ == "__main__":
    # Default paths based on your provided tree
    try:
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        
        # Defaults
        default_swow = project_root / 'data' / 'SWOW' / 'Human_SWOW-EN.R100.20180827.csv'
        default_passive = project_root / 'outputs' / 'raw_behavior' / 'model_swow_logprobs'
        default_active = project_root / 'outputs' / 'raw_behavior' / 'model_swow'
        default_out = project_root / 'outputs' / 'matrices'
    except:
        default_swow = Path('.')
        default_passive = Path('.')
        default_active = Path('.')
        default_out = Path('.')

    # Hack to allow running without args if paths match structure
    import sys
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--swow_path', str(default_swow),
            '--passive_dir', str(default_passive),
            '--active_dir', str(default_active),
            '--output_dir', str(default_out)
        ])
        
    main()