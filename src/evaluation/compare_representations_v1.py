"""
Legacy head-to-head comparison (v1).

Compares activation vs behavioral embeddings on psych norm prediction and writes
`outputs/results/comparison_results.csv` along with a small metadata summary.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIGURATION ---
MIN_SAMPLES = 50
ALPHAS = np.logspace(-2, 6, 12)
CV_FOLDS = 5
N_JOBS = -1

# --- PATHS ---
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path.cwd()

BEHAVIOR_PKL = BASE_DIR / "outputs" / "matrices" / "behavioral_embeddings.pkl"
ACTIVATION_PKL = BASE_DIR / "outputs" / "matrices" / "activation_embeddings.pkl"
NORM_PATH = BASE_DIR / "data" / "psych_norms" / "psychnorms_subset_filtered_by_swow.csv"
OUTPUT_CSV = BASE_DIR / "outputs" / "results" / "comparison_results.csv"
METADATA_JSON = BASE_DIR / "outputs" / "results" / "comparison_metadata.json"

MODEL_NAME_MAP = {
    "gemma-3-27b": "Gemma 3 27B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "mistral-small-24b": "Mistral 24B",
    "qwen-3-32b": "Qwen 3 32B",
}


# =============================================================================
# UTILITIES
# =============================================================================
def load_pkl(path: Path) -> dict:
    if not path.exists():
        print(f"❌ Missing file: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_norms(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"❌ Norm file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df["human_rating"] = pd.to_numeric(df["human_rating"], errors="coerce")
    return df.dropna(subset=["human_rating"])


def canonical_model_key(key: str) -> str:
    """Strip modality/variant prefixes to match behavior and activation keys."""
    name = key
    for prefix in ("passive_", "activation_", "active_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    # Drop variant suffixes so we only keep the base model identity
    for suffix in ("_contrast", "_shuffled", "_300d"):
        name = name.replace(suffix, "")
    return name.replace("_", "-")


def display_name(canonical: str) -> str:
    base = canonical.replace("-instruct", "").replace("-base", "").strip("-_")
    pretty = MODEL_NAME_MAP.get(base)
    if pretty:
        return pretty
    return base.replace("-", " ").replace("_", " ").title()


def overlapping_models(
    beh_embeds: Dict[str, np.ndarray], act_embeds: Dict[str, np.ndarray]
) -> List[Tuple[str, str, str]]:
    """Return (canonical, behavior_key, activation_key) for models present in both."""
    beh_map = {}
    for key in beh_embeds:
        if key == "human_matrix":
            continue
        # Ignore shuffled/contrastive for the legacy comparison
        if "_contrast" in key or "_shuffled" in key:
            continue
        beh_map[canonical_model_key(key)] = key

    act_map = {canonical_model_key(k): k for k in act_embeds}
    overlap = sorted(set(beh_map) & set(act_map))
    return [(name, beh_map[name], act_map[name]) for name in overlap]


def build_tasks(
    norm_df: pd.DataFrame,
    cue_to_idx: Dict[str, int],
    max_samples: int | None = None,
    norm_limit: int | None = None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Precompute index/y arrays per norm for reuse across models."""
    norm_df = norm_df[norm_df["word"].isin(cue_to_idx)]
    grouped = (
        norm_df.groupby(["norm_name", "word"])["human_rating"]
        .mean()
        .reset_index()
    )

    norm_names = sorted(grouped["norm_name"].unique())
    if norm_limit:
        norm_names = norm_names[:norm_limit]

    rng = np.random.default_rng(42)
    tasks: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for norm in norm_names:
        sub = grouped[grouped["norm_name"] == norm]
        words = sub["word"].tolist()
        ratings = sub["human_rating"].to_numpy(dtype=float)

        if max_samples and len(words) > max_samples:
            idx = rng.choice(len(words), size=max_samples, replace=False)
            words = [words[i] for i in idx]
            ratings = ratings[idx]

        if len(words) < MIN_SAMPLES:
            continue

        idxs = np.array([cue_to_idx[w] for w in words])
        tasks.append((norm, idxs, ratings))

    return tasks


def evaluate_embedding(X: np.ndarray, y: np.ndarray, folds: int, jobs: int) -> Tuple[float, float]:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, scoring="r2"))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=jobs)
    return scores.mean(), scores.std()


def write_metadata(df: pd.DataFrame, vocab_size: int, out_path: Path) -> None:
    meta = {}
    for model, sub in df.groupby("model"):
        avg_samples = sub["n_samples"].mean()
        meta[model] = {
            "coverage_pct": (avg_samples / vocab_size) * 100,
            "avg_sample_size": float(avg_samples),
            "norms_computed": int(sub["norm"].nunique()),
        }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy activation vs behavior comparison.")
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run a tiny subset for quick validation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on items per norm (applied before filtering).",
    )
    args = parser.parse_args()

    cv_folds = 2 if args.smoketest else CV_FOLDS
    n_jobs = 1 if args.smoketest else N_JOBS
    norm_limit = 5 if args.smoketest else None

    # If smoketest, cap norms modestly even if user passed a larger value
    max_samples = args.max_samples
    if args.smoketest:
        max_samples = min(max_samples or 200, 200)

    out_csv = (
        OUTPUT_CSV.parent / "comparison_results_SMOKETEST.csv"
        if args.smoketest
        else OUTPUT_CSV
    )
    out_meta = (
        METADATA_JSON.parent / "comparison_metadata_SMOKETEST.json"
        if args.smoketest
        else METADATA_JSON
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    beh_data = load_pkl(BEHAVIOR_PKL)
    act_data = load_pkl(ACTIVATION_PKL)
    norm_df = load_norms(NORM_PATH)

    cue_to_idx = beh_data.get("mappings", {}).get("cue_to_idx", {})
    if not cue_to_idx:
        print("❌ cue_to_idx mapping missing from behavioral embeddings.")
        sys.exit(1)

    model_pairs = overlapping_models(beh_data.get("embeddings", {}), act_data.get("embeddings", {}))
    if not model_pairs:
        print("❌ No overlapping models between behavior and activations.")
        sys.exit(1)

    print(f"📊 Models to compare: {[p[0] for p in model_pairs]}")

    tasks = build_tasks(norm_df, cue_to_idx, max_samples=max_samples, norm_limit=norm_limit)
    if not tasks:
        print("❌ No norm tasks available after filtering.")
        sys.exit(1)

    records = []
    for canonical, beh_key, act_key in model_pairs:
        beh_mat = beh_data["embeddings"][beh_key]
        act_mat = act_data["embeddings"][act_key]

        print(f"\n=== {display_name(canonical)} ===")
        for norm_name, idxs, y in tqdm(tasks, desc="Norms", leave=False):
            beh_rows = beh_mat[idxs]
            act_rows = act_mat[idxs]

            mask = np.isfinite(beh_rows).all(axis=1) & np.isfinite(act_rows).all(axis=1)
            if mask.sum() < MIN_SAMPLES:
                continue

            y_use = y[mask]
            beh_use = beh_rows[mask]
            act_use = act_rows[mask]

            beh_r2, _ = evaluate_embedding(beh_use, y_use, cv_folds, n_jobs)
            act_r2, _ = evaluate_embedding(act_use, y_use, cv_folds, n_jobs)

            records.append(
                {
                    "model": display_name(canonical),
                    "norm": norm_name,
                    "n_samples": len(y_use),
                    "r2_behavior": beh_r2,
                    "r2_activation": act_r2,
                    "delta": act_r2 - beh_r2,
                }
            )

    if not records:
        print("❌ No results produced.")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    write_metadata(df, len(cue_to_idx), out_meta)

    print(f"\n✅ Saved comparison results to {out_csv}")
    print(f"✅ Saved metadata to {out_meta}")


if __name__ == "__main__":
    main()
