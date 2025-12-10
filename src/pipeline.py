"""
src/pipeline.py

Orchestration script for the Behavioral Representation Pipeline.
Runs:
1. Vectorization (src/vectorize.py)
2. Prediction (src/evaluation/predict.py)

Usage:
    python src/pipeline.py --models qwen gpt --skip_vectorize
"""

import argparse
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument('--skip_predict', action='store_true', help="Skip the prediction step.")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent.resolve()
    vectorize_script = script_dir / "vectorize.py"
    predict_script = script_dir / "evaluation" / "predict.py"
    
    # Paths (Hardcoded defaults for now, matching the repo structure)
    project_root = script_dir.parent
    swow_path = project_root / 'data' / 'SWOW' / 'Human_SWOW-EN.R100.20180827.csv'
    passive_dir = project_root / 'outputs' / 'raw_behavior' / 'model_swow_logprobs'
    active_dir = project_root / 'outputs' / 'raw_behavior' / 'model_swow'
    activation_dir = project_root / 'outputs' / 'raw_activations'
    matrices_dir = project_root / 'outputs' / 'matrices'
    norms_path = project_root / 'data' / 'psych_norms' / 'psychnorms_subset_filtered_by_swow.csv'
    # Fallback for norms
    if not norms_path.exists():
         norms_path = project_root / 'data' / 'SWOW' / 'utils' / 'psychnorms_subset_filtered_by_swow.csv'
    results_dir = project_root / 'outputs' / 'results'
    embeddings_pkl = matrices_dir / "embeddings.pkl"

    # 1. Vectorization
    if not args.skip_vectorize:
        print("\n=== STEP 1: VECTORIZATION ===")
        cmd = [
            sys.executable, str(vectorize_script),
            '--swow_path', str(swow_path),
            '--passive_dir', str(passive_dir),
            '--active_dir', str(active_dir),
            '--activation_dir', str(activation_dir),
            '--output_dir', str(matrices_dir)
        ]
        if args.models:
            cmd.extend(['--models'] + args.models)
        run_command(cmd)
    else:
        print("\n[Pipeline] Skipping Vectorization.")

    # 2. Prediction
    if not args.skip_predict:
        print("\n=== STEP 2: PREDICTION ===")
        cmd = [
            sys.executable, str(predict_script),
            '--embeddings_path', str(embeddings_pkl),
            '--norms_path', str(norms_path),
            '--output_dir', str(results_dir)
        ]
        if args.models:
            cmd.extend(['--models'] + args.models)
        run_command(cmd)
    else:
        print("\n[Pipeline] Skipping Prediction.")

    print("\n[Pipeline] Done.")

if __name__ == "__main__":
    main()
