"""
Orchestrate the end-to-end workflow:
1) Build activation embeddings
2) Build behavioral embeddings (optionally match activation dims)
3) Run representation comparison

Usage examples:
- Default: python run_pipeline.py
- Skip activations: python run_pipeline.py --skip-activations
- Fixed 512d behavioral SVD: python run_pipeline.py --behavior-dim-strategy fixed --behavior-fixed-dim 512
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str]) -> None:
    """Run a subprocess step and fail fast on errors."""
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    default_activation_pkl = BASE_DIR / "outputs" / "matrices" / "activation_embeddings.pkl"

    parser = argparse.ArgumentParser(description="Orchestrate activations -> behavior -> comparison pipeline.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--skip-activations", action="store_true", help="Skip vectorizing activations.")
    parser.add_argument("--skip-behavior", action="store_true", help="Skip vectorizing behavior.")
    parser.add_argument("--skip-compare", action="store_true", help="Skip comparison step.")
    parser.add_argument("--behavior-full-mode", action="store_true",
                        help="Pass --full_mode to behavior vectorization.")
    parser.add_argument("--behavior-dim-strategy", choices=["activation", "mapping", "fixed"], default="activation",
                        help="Dimension selection for behavior vectorization.")
    parser.add_argument("--behavior-activation-pkl", type=Path, default=default_activation_pkl,
                        help="Activation pickle path used when --behavior-dim-strategy=activation.")
    parser.add_argument("--behavior-fixed-dim", type=int, default=300,
                        help="Target dimension when --behavior-dim-strategy=fixed or as fallback.")
    parser.add_argument("--compare-smoketest", action="store_true",
                        help="Pass --smoketest to compare_representations.py")
    args = parser.parse_args()

    # 1) Activations
    if not args.skip_activations:
        master_behav = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"
        if not master_behav.exists():
            print(f"⚠️  Master behavioral embeddings not found at {master_behav}. "
                  "vectorize_activations.py needs it for cue indexing.")
        run_step(
            "Vectorizing Activations",
            [args.python, "src/activations/vectorize_activations.py"],
        )

    # 2) Behavior
    if not args.skip_behavior:
        behavior_cmd = [
            args.python,
            "src/behavior/vectorize_behavior.py",
            "--dim_strategy",
            args.behavior_dim_strategy,
            "--fixed_dim",
            str(args.behavior_fixed_dim),
        ]
        if args.behavior_dim_strategy == "activation":
            behavior_cmd += ["--activation_pkl", str(args.behavior_activation_pkl)]
        if args.behavior_full_mode:
            behavior_cmd.append("--full_mode")
        run_step("Vectorizing Behavior", behavior_cmd)

    # 3) Comparison
    if not args.skip_compare:
        compare_cmd = [args.python, "src/evaluation/compare_representations.py"]
        if args.compare_smoketest:
            compare_cmd.append("--smoketest")
        run_step("Comparing Representations", compare_cmd)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
