"""
src/pipeline.py

Orchestration script for the Foundational Representations Pipeline.

This script executes the end-to-end workflow to generate, evaluate, and analyze
semantic representations from Large Language Models (LLMs).

Workflow Stages:
1.  **Vectorization (Standard)**: Converts raw model outputs (logprobs/tokens) into 300d embeddings.
2.  **Self-Consistency & Specificity**:
    - Evaluates 300d embeddings against the model's own norms (Self-Prediction).
    - Evaluates embeddings against Human Norms (if provided).
    - Runs specificy cross-evaluation (Model A -> Model B).
3.  **Robustness (High-Dim)**: Included in the unified evaluation where applicable.
4.  **Consolidation**: Merges all results into a single master CSV.

Usage:
    python src/pipeline.py --n_jobs 40
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

def run_command(cmd):
    print(f"\n[Pipeline] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Pipeline] Error running command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full Behavioral Representation Pipeline.")
    parser.add_argument('--models', nargs='*', help="List of models to process (substring match). If empty, runs all.")
    parser.add_argument('--skip_vectorize', action='store_true', help="Skip the vectorization step.")
    # parser.add_argument('--run_predict_human', action='store_true', help="Run the prediction (human) step (Skipped by default).") # DEPRECATED
    parser.add_argument('--skip_consistency', action='store_true', help="Skip the self-consistency step.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs (default: -1 for all)")
    parser.add_argument('--test_mode', action='store_true', help="Run a fast smoke test (1 model, 5 norms)")
    args = parser.parse_args()

    # Define paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    vectorize_script = script_dir / "vectorize.py"
    predict_script = script_dir / "evaluation" / "predict.py"
    
    # Data Paths
    # Data Paths
    swow_path = project_root / "data" / "SWOW" / "Human_SWOW-EN.R100.20180827.csv"
    passive_dir = project_root / "outputs" / "raw_behavior" / "model_swow_logprobs"
    active_dir = project_root / "outputs" / "raw_behavior" / "model_swow"
    activation_dir = project_root / "outputs" / "raw_activations"
    
    norms_path = project_root / "data" / "psych_norms" / "psychnorms_subset_filtered_by_swow.csv"
    if not norms_path.exists():
         norms_path = project_root / "data" / "SWOW" / "utils" / "psychnorms_subset_filtered_by_swow.csv"

    output_dir = project_root / "outputs" / "matrices"
    embeddings_pkl = output_dir / "embeddings.pkl"
    results_dir = project_root / "outputs" / "results"
    
    # Deranged Dir for Contrastive
    deranged_dir = project_root / "outputs" / "raw_behavior" / "model_swow_logprobs_deranged"

    # --- TEST MODE LOGIC ---
    test_limit_arg = []
    if args.test_mode:
        print("\n" + "="*60)
        print("  !!! TEST MODE ENABLED !!!")
        print("  - Selecting 1 model.")
        print("  - Limiting evaluation to 5 norms.")
        print("="*60 + "\n")
        
        # Heuristic: Scan passive dir for first model
        possible = sorted(list(passive_dir.glob('*.csv')))
        if possible:
            first_model = possible[0].stem
            print(f"[Test] Test Model Selected: {first_model}")
            args.models = [first_model]
        else:
            print("[Test] Could not find any passive files to pick a model from!")
            
            
        test_limit_arg = ['--test_limit', '5']

    # 1. Vectorization
    if not args.skip_vectorize:
        print("\n=== STEP 1: VECTORIZATION ===")
        cmd = [
            sys.executable, str(vectorize_script),
            '--swow_path', str(swow_path),
            '--passive_dir', str(passive_dir),
            '--active_dir', str(active_dir),
            '--activation_dir', str(activation_dir),
            '--output_dir', str(output_dir),
            '--n_jobs', str(args.n_jobs)
        ]
        if args.models:
            cmd.extend(['--models'] + args.models)
        if args.verbose:
            cmd.append('--verbose')
            
        # Add Deranged Dir if it exists
        if deranged_dir.exists():
             cmd.extend(['--deranged_dir', str(deranged_dir)])
        else:
             print(f"[Pipeline] Warning: Deranged dir not found at {deranged_dir}")
        
        run_command(cmd)
    else:
        print("\n[Pipeline] Skipping Vectorization.")

    # 2. Prediction (Self-Consistency & Specificity)
    # UNIFIED STEP: Run predict_self_consistency with --cross_evaluate
    # Covers:
    # 1. Self-Consistency (Model A -> Model A Norms)
    # 2. Specificity (Model A -> Model B Norms)
    # 3. Human Norms (Model A -> Human)
    # 4. All Variants (300d, High-Dim, Contrastive)
    
    if not args.skip_consistency:
        print("\n=== STEP 2: SELF-CONSISTENCY & SPECIFICITY (UNIFIED) ===")
        consistency_script = script_dir / "evaluation" / "predict_self_consistency.py"
        model_norms_dir = project_root / 'outputs' / 'raw_behavior' / 'model_norms'
        
        # We output to results/self_consistency_results.csv 
        # (The script defaults to this name in output_dir)
        
        cmd = [
            sys.executable, str(consistency_script),
            '--embeddings_path', str(embeddings_pkl),
            '--norms_dir', str(model_norms_dir),
            '--human_norms_path', str(norms_path),
            '--output_dir', str(results_dir),
            '--n_jobs', str(args.n_jobs),
            '--cross_evaluate' 
        ]
        if args.models:
            cmd.extend(['--models'] + args.models)
        if args.verbose:
            cmd.append('--verbose')
            
        # Pass test limit
        if args.test_mode:
            cmd.extend(test_limit_arg)
        
        run_command(cmd)
    else:
        print("\n[Pipeline] Skipping Prediction (Self-Consistency).")

    # 3. Partitioning (Banded Ridge)
    print("\n=== STEP 3: PARTITIONING (BANDED RIDGE) ===")
    ridge_script = script_dir / "evaluation" / "predict_banded_ridge.py"
    
    cmd_ridge = [
        sys.executable, str(ridge_script),
        '--embeddings_path', str(embeddings_pkl),
        '--norms_dir', str(model_norms_dir),
        '--output_dir', str(results_dir),
        '--n_jobs', str(args.n_jobs)
    ]
    if args.models:
        cmd_ridge.extend(['--models'] + args.models)
    if args.verbose:
        cmd_ridge.append('--verbose')
    if args.test_mode:
        cmd_ridge.extend(test_limit_arg)
        
    run_command(cmd_ridge)

    # 4. Consolidation
    print("\n=== STEP 4: SUMMARY ===")
    
    merged_results = []
    
    # Self-Consistency
    std_res_path = results_dir / "self_consistency_results.csv"
    if std_res_path.exists():
        df = pd.read_csv(std_res_path)
        df['run_type'] = 'self_consistency'
        merged_results.append(df)
        print(f"\n[Summary] Loaded {len(df)} results from {std_res_path.name}")
    else:
        print("[Pipeline] No self-consistency results found.")

    # Banded Ridge
    ridge_res_path = results_dir / "banded_ridge_results.csv"
    if ridge_res_path.exists():
        df_ridge = pd.read_csv(ridge_res_path)
        # Ridge has: model, norm, r2_joint, best_lambda_A, best_lambda_B, n_samples
        # Self-Con has: embedding_source, target_model, norm_name, r2_mean, r2_std, n_samples
        
        # Summary for Ridge
        print(f"\n[Summary] Loaded {len(df_ridge)} results from {ridge_res_path.name}")
        print("\n--- Banded Ridge Leaderboard ---")
        print(df_ridge.groupby('model')['r2_joint'].mean().sort_values(ascending=False))
        
    # Standard Leaderboard
    if std_res_path.exists():
        df = pd.read_csv(std_res_path)
        print("\n--- Self-Consistency Leaderboard (Top 10) ---")
        cols = ['embedding_source', 'target_model', 'norm_name', 'r2_mean']
        if all(c in df.columns for c in cols):
             print(df.sort_values('r2_mean', ascending=False).head(10)[cols])
        else:
             print("[Pipeline] Warning: Columns mismatch in results.csv")

    print("\n[Pipeline] Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
