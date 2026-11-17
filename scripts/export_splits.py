#!/usr/bin/env python3
"""Export the exact 5-outer/3-inner nested-CV row assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.splits import export_nested_cv_splits, sha256_file  # noqa: E402


EXPECTED_DATA_SHA256 = (
    "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_hash = sha256_file(args.data)
    if data_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError(
            f"Unexpected diabetic_data.csv SHA-256: {data_hash}. "
            f"Expected {EXPECTED_DATA_SHA256}."
        )
    raw = pd.read_csv(args.data, usecols=["readmitted"])
    y = raw["readmitted"].eq("<30").astype("int8")
    summary = export_nested_cv_splits(
        y, args.output, seed=args.seed, source_csv=args.data
    )
    print(
        f"Exported {summary['outer_manifest_rows']:,} outer and "
        f"{summary['inner_manifest_rows']:,} inner assignments to {args.output}"
    )


if __name__ == "__main__":
    main()

