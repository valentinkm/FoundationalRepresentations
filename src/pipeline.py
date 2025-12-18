"""
src/pipeline.py

Orchestration script for the Foundational Representations Pipeline.

This script executes the end-to-end workflow to generate, evaluate, and analyze
semantic representations from Large Language Models (LLMs).

Workflow Stages:
1.  **Vectorization (Standard)**: Converts raw model outputs (logprobs/tokens) into 300d embeddings.
2.  **Prediction (Human)**: [Optional] Evaluates embeddings against human psycholinguistic norms.
3.  **Self-Consistency (Standard)**: Evaluates 300d embeddings against the model's own norms (Self-Prediction).
4.  **Robustness & Specificity**:
    - Generates High-Dimensional embeddings (matched to model activations).
    - Runs Cross-Evaluation (All-vs-All) to determine model specificity.
5.  **Consolidation**: Merges all results into a single master CSV.

Usage:
    python src/pipeline.py --n_jobs 40
    python src/pipeline.py --run_predict_human --verbose
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
    parser.add_argument('--run_predict_human', action='store_true', help="Run the prediction (human) step (Skipped by default).")
    parser.add_argument('--skip_consistency', action='store_true', help="Skip the self-consistency step.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging.")
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of parallel jobs (default: -1 for all)")
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

    # 2. Prediction (Human)
    if args.run_predict_human:
        print("\n=== STEP 2: PREDICTION (HUMAN NORMS) ===")
        cmd = [
            sys.executable, str(predict_script),
            '--embeddings_path', str(embeddings_pkl),
            '--norms_path', str(norms_path),
            '--output_dir', str(results_dir),
            '--n_jobs', str(args.n_jobs)
        ]
        if args.models:
            cmd.extend(['--models'] + args.models)
        if args.verbose:
            cmd.append('--verbose')
        run_command(cmd)
    else:
        print("\n[Pipeline] Skipping Prediction (Human).")

    # 4. Prediction (Self-Consistency & Robustness)
    # UNIFIED STEP: Run predict_self_consistency with --cross_evaluate
    # This covers:
    # 1. Self-Consistency (Model A -> Model A Norms)
    # 2. Specificity (Model A -> Model B Norms)
    # 3. All Variants (300d, High-Dim, Contrastive) are in the single embeddings.pkl
    
    if not args.skip_consistency:
        print("\n=== STEP 3: SELF-CONSISTENCY & SPECIFICITY (UNIFIED) ===")
        consistency_script = script_dir / "evaluation" / "predict_self_consistency.py"
        model_norms_dir = project_root / 'outputs' / 'raw_behavior' / 'model_norms'
        
        # We output to results/self_consistency_results.csv 
        # (The script defaults to this name in output_dir)
        
        cmd = [
            sys.executable, str(consistency_script),
            '--embeddings_path', str(embeddings_pkl),
            '--norms_dir', str(model_norms_dir),
            '--output_dir', str(results_dir),
            '--n_jobs', str(args.n_jobs),
            '--cross_evaluate' 
        ]
        if args.models:
            cmd.extend(['--models'] + args.models)
        if args.verbose:
            cmd.append('--verbose')
        
        run_command(cmd)
    else:
        print("\n[Pipeline] Skipping Prediction (Self-Consistency).")

    # 5. Consolidation (Simplified - just checking the main file)
    print("\n=== STEP 4: SUMMARY ===")
    
    std_res_path = results_dir / "self_consistency_results.csv"
    if std_res_path.exists():
        df = pd.read_csv(std_res_path)
        print(f"\n[Summary] Loaded {len(df)} results from {std_res_path.name}")
        
        print("\n--- Leaderboard (Top 10 by R^2) ---")
        # Group by embedding source to see best performers
        # summary = df.groupby('embedding_source')['r2_mean'].mean().sort_values(ascending=False).head(15)
        # Actually just show top rows
        cols = ['embedding_source', 'target_model', 'norm', 'r2']
        if all(c in df.columns for c in cols):
             print(df.sort_values('r2', ascending=False).head(10)[cols])
        else:
             print(df.head())
             
    else:
        print("[Pipeline] No results file found.")

    print("\n[Pipeline] Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
