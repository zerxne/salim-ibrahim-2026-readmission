"""Paper-aligned feature engineering for the UCI diabetes readmission data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DROP_COLUMNS = ["encounter_id", "patient_nbr", "weight", "payer_code"]
DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# Supplementary Table S1 specifies these six binary diagnosis flags. The article
# body says 55 final predictors, but Table S1 says 54; applying every documented
# add/remove transformation gives 54, so the supplementary count is reproducible.
DIAGNOSIS_GROUPS = (
    "diabetes",
    "circulatory",
    "respiratory",
    "digestive",
    "injury",
    "cancer",
)


@dataclass(frozen=True)
class FeatureMetadata:
    rows: int
    predictors: int
    positive_count: int
    negative_count: int
    positive_rate: float
    top_specialties: Tuple[str, ...]
    numeric_features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "rows": self.rows,
            "predictors": self.predictors,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_rate": self.positive_rate,
            "top_specialties": list(self.top_specialties),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
        }


def age_midpoint(value: object) -> float:
    """Convert an age band such as '[60-70)' to its midpoint (65)."""
    match = re.fullmatch(r"\[(\d+)-(\d+)\)", str(value))
    if match is None:
        return np.nan
    lower, upper = map(int, match.groups())
    return (lower + upper) / 2.0


def diagnosis_group(value: object) -> str:
    """Map an ICD-9 code to one of the six Table-S1 diagnosis groups or other."""
    if pd.isna(value) or str(value).strip() in {"", "?", "Unknown"}:
        return "other"
    text = str(value).strip().upper()
    if text.startswith("250"):
        return "diabetes"
    try:
        code = float(text)
    except ValueError:  # V/E supplementary codes and malformed entries.
        return "other"

    if 140 <= code <= 239:
        return "cancer"
    if 390 <= code <= 459 or code == 785:
        return "circulatory"
    if 460 <= code <= 519 or code == 786:
        return "respiratory"
    if 520 <= code <= 579 or code == 787:
        return "digestive"
    if 800 <= code <= 999:
        return "injury"
    return "other"


def admission_type_group(value: object) -> str:
    return {1: "Emergency", 2: "Urgent", 3: "Elective"}.get(
        _safe_int(value), "Other"
    )


def discharge_group(value: object) -> str:
    code = _safe_int(value)
    if code in {1, 6, 8}:
        return "Home"
    if code in {11, 19, 20, 21}:
        return "Expired"
    if code in {2, 3, 4, 5, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30}:
        return "Transfer"
    return "Other"


def admission_source_group(value: object) -> str:
    code = _safe_int(value)
    if code == 7:
        return "Emergency Room"
    if code == 1:
        return "Physician Referral"
    if code in {4, 5, 6, 10, 18, 19, 22, 25, 26}:
        return "Transfer"
    return "Other"


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _has_group(values: Iterable[object], group: str) -> int:
    return int(any(diagnosis_group(value) == group for value in values))


def engineer_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, FeatureMetadata]:
    """Return the Table-S1-aligned 54-predictor table and binary target."""
    required = set(DROP_COLUMNS + DIAG_COLUMNS + ["readmitted", "age"])
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    data = raw.copy()
    data = data.replace("?", np.nan)
    y = data.pop("readmitted").eq("<30").astype(np.int8)
    data = data.drop(columns=DROP_COLUMNS)

    data["age"] = data["age"].map(age_midpoint)
    if data["age"].isna().any():
        raise ValueError("Age midpoint conversion produced missing values.")

    diag_groups = data[DIAG_COLUMNS].apply(
        lambda row: [diagnosis_group(value) for value in row], axis=1
    )
    for group in DIAGNOSIS_GROUPS:
        data[f"has_{group}"] = diag_groups.map(lambda groups, g=group: int(g in groups))
    data = data.drop(columns=DIAG_COLUMNS)

    data["admission_type"] = data.pop("admission_type_id").map(admission_type_group)
    data["discharge_group"] = data.pop("discharge_disposition_id").map(discharge_group)
    data["admission_source_group"] = data.pop("admission_source_id").map(
        admission_source_group
    )

    # Exact engineered variables listed in Supplementary Table S1.
    data["total_visits"] = (
        data["number_outpatient"]
        + data["number_emergency"]
        + data["number_inpatient"]
    )
    data["acute_utilization"] = data["number_emergency"] + data["number_inpatient"]
    data["disease_burden"] = data["number_diagnoses"] + data["num_procedures"]
    data["polypharmacy"] = (data["num_medications"] >= 5).astype(np.int8)
    data["long_stay"] = (data["time_in_hospital"] >= 7).astype(np.int8)
    data["hospital_pressure"] = (
        data["time_in_hospital"] * data["number_inpatient"]
    )

    categorical = list(data.select_dtypes(include=["object", "category"]).columns)
    data[categorical] = data[categorical].fillna("Unknown")

    # Ten retained values plus the catch-all "Other" gives 11 categories, as stated.
    specialty_counts = data["medical_specialty"].value_counts()
    top_specialties = tuple(specialty_counts.head(10).index.astype(str))
    data["medical_specialty"] = data["medical_specialty"].where(
        data["medical_specialty"].isin(top_specialties), "Other"
    )

    categorical = tuple(data.select_dtypes(include=["object", "category"]).columns)
    numeric = tuple(column for column in data.columns if column not in categorical)

    if data.isna().any().any():
        missing_cols = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Unexpected missing values after cleaning: {missing_cols}")
    if data.shape != (101_766, 54):
        raise ValueError(
            "Expected Supplementary Table S1's (101766, 54) predictor table, "
            f"got {data.shape}."
        )

    metadata = FeatureMetadata(
        rows=len(data),
        predictors=data.shape[1],
        positive_count=int(y.sum()),
        negative_count=int((1 - y).sum()),
        positive_rate=float(y.mean()),
        top_specialties=top_specialties,
        numeric_features=numeric,
        categorical_features=categorical,
    )
    return data, y, metadata

