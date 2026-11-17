"""Deterministic nested cross-validation split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_nested_cv_splits(
    y,
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build exact row-level outer-test and inner-validation assignments.

    ``source_row_index`` is the zero-based data-row index in diabetic_data.csv,
    excluding its header. For outer fold k, rows with ``outer_test_fold == k``
    are test rows and all other rows are training rows. Within that outer
    training set, rows with ``inner_validation_fold == j`` are validation rows
    for inner fold j and the remaining rows are its training rows.
    """
    labels = np.asarray(y, dtype=np.int8)
    source_rows = np.arange(labels.size, dtype=np.int64)
    outer_assignment = np.zeros(labels.size, dtype=np.int8)
    inner_frames: list[pd.DataFrame] = []
    outer_summary: list[dict] = []

    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=seed
    )
    for outer_fold, (outer_train, outer_test) in enumerate(
        outer_cv.split(source_rows, labels), start=1
    ):
        outer_assignment[outer_test] = outer_fold
        outer_summary.append(
            {
                "outer_fold": outer_fold,
                "train_rows": int(outer_train.size),
                "test_rows": int(outer_test.size),
                "train_positives": int(labels[outer_train].sum()),
                "test_positives": int(labels[outer_test].sum()),
                "inner_seed": seed + outer_fold,
            }
        )

        inner_assignment = np.zeros(outer_train.size, dtype=np.int8)
        inner_cv = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + outer_fold,
        )
        for inner_fold, (_, inner_validation_position) in enumerate(
            inner_cv.split(outer_train, labels[outer_train]), start=1
        ):
            inner_assignment[inner_validation_position] = inner_fold

        inner_frames.append(
            pd.DataFrame(
                {
                    "outer_fold": outer_fold,
                    "source_row_index": outer_train,
                    "target": labels[outer_train],
                    "inner_validation_fold": inner_assignment,
                }
            )
        )

    if np.any(outer_assignment == 0):
        raise RuntimeError("At least one row has no outer test-fold assignment.")

    outer_frame = pd.DataFrame(
        {
            "source_row_index": source_rows,
            "target": labels,
            "outer_test_fold": outer_assignment,
        }
    )
    inner_frame = (
        pd.concat(inner_frames, ignore_index=True)
        .sort_values(["outer_fold", "source_row_index"])
        .reset_index(drop=True)
    )
    summary = {
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "source_rows": int(labels.size),
        "positive_rows": int(labels.sum()),
        "outer_manifest_rows": int(len(outer_frame)),
        "inner_manifest_rows": int(len(inner_frame)),
        "outer_fold_summary": outer_summary,
    }
    return outer_frame, inner_frame, summary


def export_nested_cv_splits(
    y,
    output_dir: Path,
    seed: int = 42,
    source_csv: Path | None = None,
) -> dict:
    """Write shareable CSV manifests and their machine-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outer, inner, summary = build_nested_cv_splits(y, seed=seed)
    if source_csv is not None:
        summary["source_csv_sha256"] = sha256_file(source_csv)
    outer.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    inner.to_csv(output_dir / "inner_fold_assignments.csv", index=False)
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

