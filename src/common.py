"""Shared paths, split loading and metric helpers.

Everything is evaluated on the Rajpoot reproduction's saved patient-disjoint
80/20 split (79,398 development / 19,942 test encounters).  The reproduction
repo is used read-only: its code supplies the cohort definition and the 46
discharge fields, and its saved files supply the split and HGB predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

HERE = Path(__file__).resolve().parents[1]
RAJPOOT = Path(os.environ.get("RAJPOOT_REPO", HERE.parent / "Bhuvan_Rajpoot_SSRN")).resolve()
SALIM = Path(os.environ.get("SALIM_REPO", HERE.parent / "salim-ibrahim-2026-readmission")).resolve()
sys.path.insert(0, str(RAJPOOT / "src"))

from readmission.data import eligible_cohort, load_source, make_features  # noqa: E402
from readmission.metrics import calibration_parameters, choose_threshold  # noqa: E402

SOURCE = RAJPOOT / "data" / "raw" / "diabetic_data.csv"
IDS_MAPPING = RAJPOOT / "data" / "raw" / "IDS_mapping.csv"
PAPER = RAJPOOT / "results" / "full"
ASSIGNMENTS = PAPER / "splits" / "encounter_assignments.csv.gz"
PAPER_TEST = PAPER / "predictions" / "test_predictions.csv.gz"
PAPER_OOF = PAPER / "predictions" / "development_oof.csv.gz"
REFERENCE = RAJPOOT / "hgb_catboost" / "reference" / "46_fields"

OUT = HERE / "results"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PREDICTIONS = OUT / "predictions"
MODELS = HERE / "artifacts" / "models"
for _d in (TABLES, FIGURES, PREDICTIONS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 2026            # model seed used by the team
BOOTSTRAP_SEED = 2028  # patient-cluster bootstrap seed used by the team
THREADS = 4
KEYS = ["encounter_id", "patient_nbr", "target"]
TOP_FRACTION = 0.10    # fixed alert capacity used on slides 31, 32, 41, 42


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_split():
    """Return (cohort, features, manifest, assignment, dev, test) aligned on encounter_id."""
    assignment = pd.read_csv(ASSIGNMENTS)
    assert len(assignment) == 99340 and assignment.encounter_id.is_unique
    assert assignment.groupby("patient_nbr").partition.nunique().max() == 1
    cohort, _ = eligible_cohort(load_source(SOURCE))
    pd.testing.assert_frame_equal(cohort[KEYS], assignment[KEYS], check_dtype=False)
    features, manifest = make_features(cohort)
    features.index = assignment.encounter_id.to_numpy()
    cohort.index = features.index
    dev = assignment.loc[assignment.partition.eq("development")].reset_index(drop=True)
    test = assignment.loc[assignment.partition.eq("test")].reset_index(drop=True)
    assert (len(dev), int(dev.target.sum())) == (79398, 9017)
    assert (len(test), test.patient_nbr.nunique(), int(test.target.sum())) == (19942, 13998, 2297)
    return cohort, features, manifest["discharge"], assignment, dev, test


def top_k(n: int, fraction: float = TOP_FRACTION) -> int:
    return int(np.floor(n * fraction))


def capacity_flags(p, k: int) -> np.ndarray:
    """Flag exactly k highest scores (stable tie-break by position)."""
    order = np.argsort(-np.asarray(p), kind="mergesort")
    flags = np.zeros(len(p), dtype=bool)
    flags[order[:k]] = True
    return flags


def confusion_metrics(y, flags) -> dict:
    y = np.asarray(y).astype(int)
    flags = np.asarray(flags).astype(bool)
    tp = int(np.sum(flags & (y == 1)))
    fp = int(np.sum(flags & (y == 0)))
    fn = int(np.sum(~flags & (y == 1)))
    tn = int(np.sum(~flags & (y == 0)))

    def div(a, b):
        return float(a / b) if b else float("nan")

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {"alerts": tp + fp, "alert_rate": div(tp + fp, len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "specificity": div(tn, tn + fp), "npv": div(tn, tn + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "f2": div(5 * tp, 5 * tp + fp + 4 * fn),
            "balanced_accuracy": (recall + div(tn, tn + fp)) / 2, "accuracy": div(tp + tn, len(y))}


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Equal-frequency ECE (deciles), matching the decile plot on slide 39."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["bin"] = pd.qcut(frame.p.rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float(np.sum(grouped.n * (grouped.obs - grouped.pred).abs()) / len(frame))


def probability_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    intercept, slope = calibration_parameters(y, p)
    prevalence = y.mean()
    brier = brier_score_loss(y, p)
    return {"ap": average_precision_score(y, p), "auroc": roc_auc_score(y, p), "brier": brier,
            "scaled_brier": 1 - brier / (prevalence * (1 - prevalence)), "log_loss": log_loss(y, p),
            "calibration_intercept": intercept, "calibration_slope": slope,
            "mean_predicted": float(p.mean()), "observed_rate": float(prevalence),
            "ece_deciles": expected_calibration_error(y, p)}


def platt(p, coefficients: dict) -> np.ndarray:
    z = logit(np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12))
    return expit(coefficients["intercept"] + coefficients["slope"] * z)


def patient_weights(groups, replicates: int, seed: int = BOOTSTRAP_SEED):
    """Yield multiplicity weights for a patient-cluster bootstrap (same scheme as the team)."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n = inverse.max() + 1
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        yield counts[inverse]


def fit_dev_threshold(y_dev, p_dev, beta: float) -> float:
    return choose_threshold(np.asarray(y_dev), np.asarray(p_dev), beta)
