"""
Utility to summarize stored embedding pickles.
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def describe_value(value: object) -> str:
    """Produce a short description for a value."""
    if isinstance(value, np.ndarray):
        return f"array shape {value.shape} dtype={value.dtype}"
    if isinstance(value, Mapping):
        return f"dict len={len(value)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} len={len(value)}"
    return f"type {type(value)}"


def summarize_pickle(path: Path) -> None:
    data = pickle.loads(path.read_bytes())
    print(f"\n{path}:")

    if not isinstance(data, Mapping):
        print(f"  top-level: {describe_value(data)}")
        return

    embeddings = data.get("embeddings", {})
    mappings = data.get("mappings", {})

    print(f"  embeddings ({len(embeddings)} entries):")
    for name in sorted(embeddings):
        value = embeddings[name]
        print(f"    {name}: {describe_value(value)}")

    print("  mappings:")
    if isinstance(mappings, Mapping):
        for name in sorted(mappings):
            value = mappings[name]
            print(f"    {name}: {describe_value(value)}")
    else:
        print(f"    {describe_value(mappings)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize embedding pickle files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("outputs/matrices/behavioral_embeddings.pkl"),
            Path("outputs/matrices/activation_embeddings.pkl"),
        ],
        help="Paths to .pkl files to summarize",
    )
    args = parser.parse_args()

    for path in args.paths:
        summarize_pickle(path)


if __name__ == "__main__":
    main()
