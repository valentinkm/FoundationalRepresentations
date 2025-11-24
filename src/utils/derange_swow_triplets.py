#!/usr/bin/env python3
"""Derange SWOW cue-response triplets while preserving participant metadata."""

import argparse
import random
from typing import List, Set

import pandas as pd
from pandas.api.types import is_numeric_dtype

DEFAULT_INPUT_PATH = "data/SWOW/SWOW-EN.R100.20180827.csv"
DEFAULT_OUTPUT_PATH = "data/SWOW-EN.R100.20180827.deranged.csv"
DEFAULT_SEED = 42
UNQUOTED_STRING_COLUMNS: Set[str] = {"created_at"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly reassign cue-response triplets from the SWOW dataset "
            "using a strict derangement that keeps triplets intact."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to the source SWOW CSV file "
            f"(default: {DEFAULT_INPUT_PATH})."
        ),
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Destination path for the deranged CSV output "
            f"(default: {DEFAULT_OUTPUT_PATH})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed to make the derangement reproducible.",
    )
    return parser.parse_args()


def sattolo_permutation(n: int, rng: random.Random) -> List[int]:
    """Return a single-cycle permutation of indices 0..n-1."""
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i - 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def derange_triplets(triplets: List[tuple[str, str, str]], rng: random.Random) -> List[int]:
    """Derange triplets so no cue retains an identical triplet."""
    n = len(triplets)
    if n < 2:
        raise ValueError("Need at least two triplets to compute a derangement.")

    def regenerate() -> List[int]:
        return sattolo_permutation(n, rng)

    perm = regenerate()

    is_conflict = [triplets[perm[i]] == triplets[i] for i in range(n)]
    bad_stack = [i for i, conflict in enumerate(is_conflict) if conflict]
    if not bad_stack:
        return perm

    good_indices = [i for i, conflict in enumerate(is_conflict) if not conflict]
    good_positions = {idx: pos for pos, idx in enumerate(good_indices)}

    def remove_from_good(idx: int) -> None:
        pos = good_positions.pop(idx, None)
        if pos is None:
            return
        last = good_indices.pop()
        if pos < len(good_indices):
            good_indices[pos] = last
            good_positions[last] = pos

    def add_to_good(idx: int) -> None:
        if idx in good_positions:
            return
        good_positions[idx] = len(good_indices)
        good_indices.append(idx)

    resets_remaining = 10

    while bad_stack:
        i = bad_stack.pop()
        if not is_conflict[i]:
            continue

        if not good_indices:
            if resets_remaining == 0:
                raise RuntimeError("Unable to resolve derangement conflicts.")
            perm = regenerate()
            is_conflict = [triplets[perm[k]] == triplets[k] for k in range(n)]
            bad_stack = [k for k, conflict in enumerate(is_conflict) if conflict]
            good_indices = [k for k, conflict in enumerate(is_conflict) if not conflict]
            good_positions = {idx: pos for pos, idx in enumerate(good_indices)}
            resets_remaining -= 1
            continue

        resolved = False
        for _ in range(200):
            j = good_indices[rng.randrange(len(good_indices))]
            if triplets[perm[j]] == triplets[i] or triplets[perm[i]] == triplets[j]:
                continue

            perm[i], perm[j] = perm[j], perm[i]

            is_conflict[i] = triplets[perm[i]] == triplets[i]
            is_conflict[j] = triplets[perm[j]] == triplets[j]

            if is_conflict[i]:
                remove_from_good(i)
                bad_stack.append(i)
            else:
                add_to_good(i)

            if is_conflict[j]:
                remove_from_good(j)
                bad_stack.append(j)
            else:
                add_to_good(j)

            resolved = True
            break

        if not resolved:
            if resets_remaining == 0:
                raise RuntimeError("Unable to resolve derangement conflicts.")
            perm = regenerate()
            is_conflict = [triplets[perm[k]] == triplets[k] for k in range(n)]
            bad_stack = [k for k, conflict in enumerate(is_conflict) if conflict]
            good_indices = [k for k, conflict in enumerate(is_conflict) if not conflict]
            good_positions = {idx: pos for pos, idx in enumerate(good_indices)}
            resets_remaining -= 1

    return perm


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    df = pd.read_csv(args.input_path, keep_default_na=False, low_memory=False)

    if "Unnamed: 0" in df.columns and "" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": ""})

    if not {"R1", "R2", "R3"}.issubset(df.columns):
        missing = {"R1", "R2", "R3"} - set(df.columns)
        raise ValueError(f"Input file is missing expected columns: {sorted(missing)}")

    triplets = list(zip(df["R1"], df["R2"], df["R3"]))
    permuted_indices = derange_triplets(triplets, rng)
    deranged_triplets = [triplets[i] for i in permuted_indices]

    df.loc[:, ["R1", "R2", "R3"]] = pd.DataFrame(
        deranged_triplets, columns=["R1", "R2", "R3"]
    )

    for col in ("R1", "R2", "R3"):
        df[col] = df[col].replace("NA", pd.NA)

    if "education" in df.columns:
        df["education"] = pd.to_numeric(df["education"], errors="coerce").astype("Int64")

    if "" in df.columns:
        df[""] = df[""].astype(str)

    numeric_columns = {col for col in df.columns if is_numeric_dtype(df[col])}

    with open(args.output_path, "w", newline="", encoding="utf-8") as handle:
        header = ",".join(f'"{col}"' for col in df.columns)
        handle.write(header + "\n")

        for row_values in df.itertuples(index=False, name=None):
            formatted = []
            for value, col_name in zip(row_values, df.columns):
                if pd.isna(value):
                    formatted.append("NA")
                elif col_name in numeric_columns:
                    if isinstance(value, float) and value.is_integer():
                        formatted.append(str(int(value)))
                    else:
                        formatted.append(str(value))
                elif col_name in UNQUOTED_STRING_COLUMNS:
                    text = str(value).replace('"', '""')
                    formatted.append(text)
                else:
                    text = str(value).replace('"', '""')
                    formatted.append(f'"{text}"')
            handle.write(",".join(formatted) + "\n")


if __name__ == "__main__":
    main()
