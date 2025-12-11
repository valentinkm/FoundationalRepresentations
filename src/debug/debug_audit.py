"""
src/debug_audit.py

Diagnostic script to perform 4 layers of integrity checks on the pipeline artifacts.
Run this to audit Embeddings, Raw Data, Model Norms, and Alignment.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys

# --- CONFIG ---
SWOW_PATH = Path("data/SWOW/Human_SWOW-EN.R100.20180827.csv")
EMBEDDINGS_PATH = Path("outputs/matrices/embeddings.pkl")
RAW_ACTIVATIONS_DIR = Path("outputs/raw_activations")
MODEL_NORMS_DIR = Path("outputs/raw_behavior/model_norms")

def load_swow_vocab(path: Path, min_freq=5):
    """Load canonical SWOW vocabulary."""
    print(f"[Loader] Loading SWOW vocab from {path}...")
    try:
        df = pd.read_csv(path)
        # Replicate vectorize.py logic briefly
        base = df[["cue", "R1", "R2", "R3"]].copy()
        base["cue"] = base["cue"].astype(str).str.lower().str.strip()
        long = base.melt(id_vars=["cue"], value_vars=["R1", "R2", "R3"], value_name="response")
        long = long.dropna(subset=["response"])
        long["response"] = long["response"].astype(str).str.lower().str.strip()
        long = long[long["response"] != ""]
        counts = long["response"].value_counts()
        vocab = set(counts[counts >= min_freq].index)
        print(f"[Loader] SWOW Vocab Size: {len(vocab)}")
        return vocab
    except Exception as e:
        print(f"[Loader] Failed to load SWOW: {e}")
        return set()

def check_embeddings(embeddings_dict):
    """Layer 1: The Embedding Audit (X Matrix)"""
    print("\n=== Layer 1: Embedding Audit (X) ===")
    report = {}
    
    for name, mat in embeddings_dict.items():
        if name == 'mappings': continue
        if not hasattr(mat, 'shape'):
            print(f"Skipping {name} (Type: {type(mat)})")
            continue
        
        print(f"Checking {name} {mat.shape}...")
        issues = []
        status = "PASS"
        
        # 1.1 Silent Zero
        norms = np.linalg.norm(mat, axis=1)
        zero_rows = np.sum(norms == 0)
        if zero_rows > 0:
            issues.append(f"CRITICAL: {zero_rows} rows are Zero Vectors")
            status = "FAIL"
            
        # 1.2 Collapsed Space
        # Sample 100 random rows
        if mat.shape[0] > 100:
            indices = np.random.choice(mat.shape[0], 100, replace=False)
            sample = mat[indices]
            # Pairwise cosine similarity
            # Handle zero vectors in sample to avoid div by zero warnings in cosine_similarity
            # (though sklearn handles it by returning 0 usually, or we sanitized)
            sim_mat = cosine_similarity(sample)
            # Mean of upper triangle (excluding diagonal)
            upper_tri = sim_mat[np.triu_indices(100, k=1)]
            mean_sim = np.mean(upper_tri)
            if mean_sim > 0.95:
                issues.append(f"FAIL: Collapsed Space (Mean Sim = {mean_sim:.4f} > 0.95)")
                status = "FAIL"
        
        # 1.3 Dead Neuron
        stds = np.std(mat, axis=0)
        dead_cols = np.sum(stds < 1e-9)
        dead_pct = dead_cols / mat.shape[1]
        if dead_pct > 0.10:
            issues.append(f"WARNING: {dead_pct:.1%} Dead Neurons ({dead_cols}/{mat.shape[1]})")
            if status == "PASS": status = "WARNING"
            
        report[name] = {"status": status, "issues": issues, "zero_indices": np.where(norms == 0)[0]}
        
    return report

def check_raw_data(vocab_set):
    """Layer 2: The Raw Data Audit (CSVs)"""
    print("\n=== Layer 2: Raw Data Audit (CSVs) ===")
    report = {}
    
    files = list(RAW_ACTIVATIONS_DIR.glob("*.csv"))
    if not files:
        print("[Layer 2] No raw activation files found.")
        return report
        
    for fp in files:
        name = fp.stem
        print(f"Checking {name}...")
        issues = []
        status = "PASS"
        
        try:
            df = pd.read_csv(fp)
            
            # 2.1 Coverage Check
            # Assuming column 'word' or 'cue' exists. 
            # Based on previous logs, it seems to be 'word' or index? 
            # Let's check columns.
            col_name = 'word' if 'word' in df.columns else 'cue'
            if col_name not in df.columns:
                # Maybe it has no header?
                if 'Unnamed: 0' in df.columns:
                     # Sometimes index is the word
                     words = set(df['Unnamed: 0'].astype(str).str.lower().str.strip())
                else:
                     issues.append("FAIL: Could not identify word column")
                     status = "FAIL"
                     words = set()
            else:
                words = set(df[col_name].astype(str).str.lower().str.strip())
            
            if words:
                overlap = words.intersection(vocab_set)
                coverage = len(overlap) / len(vocab_set) if len(vocab_set) > 0 else 0
                if coverage < 0.95:
                    issues.append(f"FAIL: Low Coverage ({coverage:.1%}). Found {len(overlap)}/{len(vocab_set)} SWOW words.")
                    status = "FAIL"
            
            # 2.2 Formatting Check
            # Check a sample vector column (assuming 'vector' or similar, or all other cols)
            # Actually, raw activations usually have many columns (dims).
            # If it's a wide format (word, dim1, dim2...), we check if values are numeric.
            # If it's a 'vector' column with strings "[0.1, 0.2]", we check that.
            # Let's assume wide format or check dtypes.
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 10: # Arbitrary threshold for "has vector data"
                 # Maybe it's string formatted?
                 if 'vector' in df.columns:
                     # Check if it looks like a list
                     sample = df['vector'].iloc[0]
                     if not (isinstance(sample, str) and sample.startswith('[') and sample.endswith(']')):
                         issues.append("FAIL: 'vector' column format invalid")
                         status = "FAIL"
                 else:
                     issues.append("FAIL: No vector data found (few numeric cols and no 'vector' col)")
                     status = "FAIL"
            
        except Exception as e:
            issues.append(f"FAIL: Error reading file: {e}")
            status = "FAIL"
            
        report[name] = {"status": status, "issues": issues}
        
    return report

def check_norms():
    """Layer 3: The Target Audit (y Norms)"""
    print("\n=== Layer 3: Target Audit (y Norms) ===")
    report = {}
    
    files = list(MODEL_NORMS_DIR.glob("*.csv"))
    if not files:
        print("[Layer 3] No model norm files found.")
        return report
        
    for fp in files:
        name = fp.stem
        print(f"Checking {name}...")
        issues = []
        status = "PASS"
        
        try:
            df = pd.read_csv(fp)
            # Expected cols: norm, word, cleaned_rating
            if 'cleaned_rating' not in df.columns:
                issues.append("FAIL: Missing 'cleaned_rating' column")
                status = "FAIL"
                report[name] = {"status": status, "issues": issues}
                continue
                
            norms = df['norm'].unique()
            for norm in norms:
                sub = df[df['norm'] == norm]
                ratings = pd.to_numeric(sub['cleaned_rating'], errors='coerce').dropna()
                
                if len(ratings) == 0:
                    continue
                    
                # 3.1 Lazy Model (Variance)
                std = ratings.std()
                if std < 0.1:
                    issues.append(f"FAIL: {norm} - Lazy Model (Std={std:.4f} < 0.1)")
                    status = "FAIL"
                
                # 3.2 Distribution Skew
                mode_count = ratings.value_counts().iloc[0]
                mode_pct = mode_count / len(ratings)
                if mode_pct > 0.90:
                    issues.append(f"WARNING: {norm} - Skewed ({mode_pct:.1%} is mode)")
                    if status == "PASS": status = "WARNING"
                    
        except Exception as e:
            issues.append(f"FAIL: Error reading file: {e}")
            status = "FAIL"
            
        report[name] = {"status": status, "issues": issues}
        
    return report

def check_alignment(embeddings_dict, emb_report, model_norms_report):
    """Layer 4: The Alignment Audit (Merge)"""
    print("\n=== Layer 4: Alignment Audit ===")
    report = {}
    
    cue_to_idx = embeddings_dict.get('mappings', {}).get('cue_to_idx', {})
    idx_to_cue = {v: k for k, v in cue_to_idx.items()}
    
    # We need to match embedding keys to norm keys
    # Embedding keys: 'passive_MODEL', 'activation_MODEL'
    # Norm keys: 'MODEL' (filename stem)
    
    norm_files = {fp.stem: fp for fp in MODEL_NORMS_DIR.glob("*.csv")}
    
    for emb_name, mat in embeddings_dict.items():
        if emb_name == 'mappings': continue
        
        # Find matching norm file
        matched_model = None
        for model_key in norm_files.keys():
            if model_key in emb_name:
                matched_model = model_key
                break
        
        if not matched_model:
            continue
            
        print(f"Checking Alignment: {emb_name} <-> {matched_model}...")
        issues = []
        status = "PASS"
        
        # Load norms to get words
        try:
            df_norms = pd.read_csv(norm_files[matched_model])
            norm_words = set(df_norms['word'].astype(str).str.lower().str.strip())
            
            # Get embedding words (cues)
            # The matrix is aligned to cue_to_idx
            emb_cues = set(cue_to_idx.keys())
            
            # 4.1 Intersection Count
            overlap = norm_words.intersection(emb_cues)
            if len(overlap) < 100:
                issues.append(f"FAIL: Low Intersection ({len(overlap)} words < 100)")
                status = "FAIL"
            
            # 4.2 Ghost Match
            # Check if any overlap word corresponds to a zero vector in this embedding
            zero_indices = set(emb_report.get(emb_name, {}).get('zero_indices', []))
            if zero_indices:
                ghosts = 0
                for w in overlap:
                    idx = cue_to_idx.get(w)
                    if idx in zero_indices:
                        ghosts += 1
                
                if ghosts > 0:
                    issues.append(f"CRITICAL FAIL: {ghosts} Ghost Matches (Overlap words with Zero Vectors)")
                    status = "FAIL"
                    
        except Exception as e:
            issues.append(f"FAIL: Error during check: {e}")
            status = "FAIL"
            
        report[f"{emb_name} <-> {matched_model}"] = {"status": status, "issues": issues}
        
    return report

def check_vocab_consistency(embeddings_dict, vocab_set):
    """Layer 5: Vocabulary Consistency Audit"""
    print("\n=== Layer 5: Vocabulary Consistency Audit ===")
    report = {}
    
    cue_to_idx = embeddings_dict.get('mappings', {}).get('cue_to_idx', {})
    idx_to_cue = {v: k for k, v in cue_to_idx.items()}
    
    # 1. Check Canonical Vocab vs SWOW
    canonical_cues = set(cue_to_idx.keys())
    missing_from_swow = vocab_set - canonical_cues
    extra_in_pipeline = canonical_cues - vocab_set
    
    print(f"Canonical Vocab Size: {len(canonical_cues)}")
    print(f"SWOW Vocab Size: {len(vocab_set)}")
    print(f"Missing from SWOW: {len(missing_from_swow)}")
    print(f"Extra in Pipeline: {len(extra_in_pipeline)}")
    
    # 2. Check Effective Coverage (Non-Zero Rows) per Matrix
    effective_vocabs = {}
    
    for name, mat in embeddings_dict.items():
        if name == 'mappings': continue
        if not hasattr(mat, 'shape'): continue
        
        # Identify non-zero rows
        norms = np.linalg.norm(mat, axis=1)
        non_zero_indices = np.where(norms > 0)[0]
        
        effective_words = {idx_to_cue[idx] for idx in non_zero_indices}
        effective_vocabs[name] = effective_words
        
        coverage_pct = len(effective_words) / len(canonical_cues)
        print(f"  > {name}: {len(effective_words)}/{len(canonical_cues)} ({coverage_pct:.1%}) non-zero rows.")
        
        if coverage_pct < 0.99:
            report[name] = {"status": "FAIL", "issues": [f"Low Coverage: {coverage_pct:.1%}"]}
        else:
            report[name] = {"status": "PASS", "issues": []}

    # 3. Pairwise Consistency
    # Check if all matrices have the SAME effective vocabulary
    print("\n  > Consistency Check:")
    keys = list(effective_vocabs.keys())
    if not keys:
        return report
        
    base_key = keys[0]
    base_vocab = effective_vocabs[base_key]
    
    all_consistent = True
    for k in keys[1:]:
        vocab = effective_vocabs[k]
        diff = base_vocab.symmetric_difference(vocab)
        if diff:
            all_consistent = False
            print(f"    FAIL: {base_key} vs {k} differ by {len(diff)} words.")
            # print(f"      Sample diff: {list(diff)[:5]}")
        else:
            print(f"    PASS: {base_key} vs {k} are identical.")
            
    if all_consistent:
        print("    PASS: All matrices have identical effective vocabularies.")
    else:
        print("    FAIL: Vocabularies are NOT identical across models.")

    return report

def print_summary(report_name, report_data):
    print(f"\n--- {report_name} Report ---")
    for key, data in report_data.items():
        status = data['status']
        issues = data['issues']
        color = ""
        if status == "FAIL": color = "🔴 "
        elif status == "WARNING": color = "jq "
        else: color = "🟢 "
        
        print(f"{color}[{status}] {key}")
        for issue in issues:
            print(f"    - {issue}")

def main():
    # 0. Load Data
    vocab_set = load_swow_vocab(SWOW_PATH)
    
    print(f"[Loader] Loading embeddings from {EMBEDDINGS_PATH}...")
    if not EMBEDDINGS_PATH.exists():
        print("Embeddings file not found!")
        return
        
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        
    if isinstance(data, dict) and 'embeddings' in data:
        print("[Loader] Detected nested pickle structure. Unwrapping...")
        embeddings_dict = data['embeddings']
        embeddings_dict['mappings'] = data.get('mappings', {})
    else:
        embeddings_dict = data
        
    # 1. Embedding Audit
    emb_report = check_embeddings(embeddings_dict)
    
    # 2. Raw Data Audit
    raw_report = check_raw_data(vocab_set)
    
    # 3. Norms Audit
    norms_report = check_norms()
    
    # 4. Alignment Audit
    align_report = check_alignment(embeddings_dict, emb_report, norms_report)
    
    # 5. Vocab Consistency
    vocab_report = check_vocab_consistency(embeddings_dict, vocab_set)
    
    # Print Summaries
    print_summary("Layer 1: Embeddings", emb_report)
    print_summary("Layer 2: Raw Data", raw_report)
    print_summary("Layer 3: Norms", norms_report)
    print_summary("Layer 4: Alignment", align_report)
    print_summary("Layer 5: Vocab Consistency", vocab_report)

if __name__ == "__main__":
    main()