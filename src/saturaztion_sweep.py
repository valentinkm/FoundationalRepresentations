"""
Sweep embedding dimensions for behavior vs activations across many norms.

- Loads behavior/activation embeddings from the saved PKLs.
- Samples a subset of items per norm (to keep runtime reasonable).
- Runs PCA + Ridge with cross-validation for each dimension.
- Writes a long-form CSV of R2 scores per norm / modality / dimension.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NORM_PATH = Path("data/psych_norms/psychnorms_subset_filtered_by_swow.csv")
BEHAVIOR_PKL = Path("outputs/matrices/behavioral_embeddings.pkl")
ACTIVATION_PKL = Path("outputs/matrices/activation_embeddings.pkl")

# Dimensions to try (will be clipped to valid ranges per modality/norm)
DIMS_TO_TEST = [16, 32, 64, 128, 256, 384, 512, 768, 1024]

# Norm sampling / filtering
# Keep all norms, but drastically cut per-norm sample to reduce runtime.
NORM_SAMPLE_PER_NORM = 640  # None to use all; smaller -> faster
MIN_SAMPLES_PER_NORM = 40   # skip norms with fewer usable items than this

# Modeling / CV
RIDGE_ALPHA = 1.0
CV_SPLITS = 5
CV_REPEATS = 1
RANDOM_SEED = 42
N_JOBS = -1  # use all cores for CV scoring

# Output
OUTPUT_CSV = Path("outputs/results/saturation_sweep.csv")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_embeddings() -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load behavior and activation embeddings plus cue index."""
    if not BEHAVIOR_PKL.exists():
        raise FileNotFoundError(f"Behavioral embeddings not found: {BEHAVIOR_PKL}")
    if not ACTIVATION_PKL.exists():
        raise FileNotFoundError(f"Activation embeddings not found: {ACTIVATION_PKL}")

    with open(BEHAVIOR_PKL, "rb") as f:
        beh_payload = pickle.load(f)
    with open(ACTIVATION_PKL, "rb") as f:
        act_payload = pickle.load(f)

    cue_to_idx = beh_payload.get("mappings", {}).get("cue_to_idx")
    if cue_to_idx is None:
        raise KeyError("cue_to_idx mapping missing from behavioral embeddings payload")

    return beh_payload.get("embeddings", {}), act_payload.get("embeddings", {}), cue_to_idx


def canonical_name(name: str) -> str:
    """Strip modality prefixes and normalize naming quirks to match across sources."""
    for prefix in ("passive_", "activation_", "active_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Align patterns like "27-b-instruct" with "27b-instruct"
    name = name.replace("-b-", "b-")
    return name


def overlapping_models(
    beh_embeds: Dict[str, np.ndarray], act_embeds: Dict[str, np.ndarray]
) -> List[Tuple[str, str, str]]:
    """
    Find overlapping model names across behavior/activation embeddings.
    Returns list of (canonical, behavior_key, activation_key).
    """
    beh_map = {canonical_name(k): k for k in beh_embeds if k != "human_matrix"}
    act_map = {canonical_name(k): k for k in act_embeds}
    overlap = sorted(set(beh_map) & set(act_map))
    return [(name, beh_map[name], act_map[name]) for name in overlap]


def load_norm_tasks(cue_to_idx: Dict[str, int]) -> List[dict]:
    """
    Build norm prediction tasks.
    Returns a list of dicts: {"norm": str, "idxs": np.ndarray, "y": np.ndarray}
    """
    if not NORM_PATH.exists():
        raise FileNotFoundError(f"Norm file not found: {NORM_PATH}")

    df = pd.read_csv(NORM_PATH)
    # Standardize
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df["human_rating"] = pd.to_numeric(df["human_rating"], errors="coerce")
    df = df.dropna(subset=["human_rating"])
    df = df[df["word"].isin(cue_to_idx)]

    # Aggregate duplicate word/norm pairs by mean
    grouped = (
        df.groupby(["norm_name", "word"])["human_rating"]
        .mean()
        .reset_index()
    )

    rng_state = RANDOM_SEED
    tasks: List[dict] = []
    for norm_name, g in grouped.groupby("norm_name"):
        if NORM_SAMPLE_PER_NORM:
            sample_n = min(NORM_SAMPLE_PER_NORM, len(g))
            g = g.sample(n=sample_n, random_state=rng_state, replace=False)
        idxs = g["word"].map(cue_to_idx).to_numpy()
        y = g["human_rating"].to_numpy(dtype=float)

        if len(idxs) < MIN_SAMPLES_PER_NORM:
            continue

        tasks.append({"norm": norm_name, "idxs": idxs, "y": y})

    return tasks


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def filter_rows_for_embedding(X: np.ndarray, idxs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter out rows with NaNs in the selected embedding rows."""
    X_subset = X[idxs]
    mask = ~np.isnan(X_subset).any(axis=1)
    return X_subset[mask], y[mask]


def valid_dims_for_task(X: np.ndarray, dims: Iterable[int], n_samples: int) -> List[int]:
    """
    Keep only dimensions that are valid for PCA given features and CV splits.
    Ensures n_components <= min(n_features, min_train_samples).
    """
    n_features = X.shape[1]
    min_train = n_samples * (CV_SPLITS - 1) // CV_SPLITS
    return [d for d in dims if 1 <= d <= n_features and d < min_train]


def sweep_modality(
    X: np.ndarray,
    task: dict,
    dims: Iterable[int],
    modality: str,
    model_key: str,
) -> List[dict]:
    """Run PCA + Ridge CV across dimensions for a single modality and norm."""
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    X_use, y_use = filter_rows_for_embedding(X, task["idxs"], task["y"])
    if len(y_use) < MIN_SAMPLES_PER_NORM:
        return []

    records: List[dict] = []
    dims_valid = valid_dims_for_task(X_use, dims, len(y_use))
    if not dims_valid:
        return records

    for d in dims_valid:
        pipe = make_pipeline(
            PCA(n_components=d, svd_solver="randomized", random_state=RANDOM_SEED),
            Ridge(alpha=RIDGE_ALPHA),
        )
        score = cross_val_score(pipe, X_use, y_use, cv=cv, scoring="r2", n_jobs=N_JOBS).mean()
        records.append(
            {
                "norm": task["norm"],
                "modality": modality,
                "model": model_key,
                "dim": d,
                "r2": score,
                "n_samples": len(y_use),
            }
        )
    return records


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_sweep() -> pd.DataFrame:
    beh_embeds, act_embeds, cue_to_idx = load_embeddings()
    model_triples = overlapping_models(beh_embeds, act_embeds)
    if not model_triples:
        raise RuntimeError("No overlapping models between behavioral and activation embeddings.")

    tasks = load_norm_tasks(cue_to_idx)

    if not tasks:
        raise RuntimeError("No norm tasks after filtering/sampling.")

    print(f"Found {len(tasks)} norms with >= {MIN_SAMPLES_PER_NORM} samples after subsampling.")

    all_records: List[dict] = []
    # Rough upper bound on jobs for ETA: norms * models * modalities * dims
    total_jobs = len(tasks) * len(model_triples) * 2 * len(DIMS_TO_TEST)
    jobs_done = 0
    t0 = time.time()

    print(f"Overlapping models: {[m[0] for m in model_triples]}")

    for task in tasks:
        print(f"Norm '{task['norm']}': {len(task['idxs'])} examples before NaN filtering.")
        for canonical, beh_key, act_key in model_triples:
            for modality, X, model_key in (
                ("behavior", np.asarray(beh_embeds[beh_key]), canonical),
                ("activation", np.asarray(act_embeds[act_key]), canonical),
            ):
                records = sweep_modality(X, task, DIMS_TO_TEST, modality, model_key)
                all_records.extend(records)
                jobs_done += len(records)
                # ETA based on average time per completed job
                elapsed = time.time() - t0
                if jobs_done > 0 and total_jobs > 0:
                    avg = elapsed / jobs_done
                    remaining = max(total_jobs - jobs_done, 0)
                    eta_min = (remaining * avg) / 60
                    print(
                        f"  Progress: {jobs_done}/{total_jobs} jobs "
                        f"({jobs_done/total_jobs:.1%}), ETA ~ {eta_min:.1f} min"
                    )

    if not all_records:
        raise RuntimeError("No scores computed; check dimensions and sampling settings.")

    df_out = pd.DataFrame(all_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved sweep results to {OUTPUT_CSV} ({len(df_out)} rows).")
    return df_out


if __name__ == "__main__":
    run_sweep()
