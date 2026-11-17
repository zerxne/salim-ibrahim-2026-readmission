from pathlib import Path

import pandas as pd

from src.features import age_midpoint, diagnosis_group, engineer_features
from src.splits import build_nested_cv_splits


def test_age_midpoint():
    assert age_midpoint("[60-70)") == 65
    assert age_midpoint("[0-10)") == 5


def test_diagnosis_groups():
    assert diagnosis_group("250.83") == "diabetes"
    assert diagnosis_group("428") == "circulatory"
    assert diagnosis_group("174") == "cancer"
    assert diagnosis_group("V27") == "other"
    assert diagnosis_group("?") == "other"


def test_full_dataset_shape_if_available():
    data_path = Path(__file__).parents[1] / "data" / "raw" / "diabetic_data.csv"
    if not data_path.exists():
        return
    raw = pd.read_csv(data_path, low_memory=False)
    X, y, metadata = engineer_features(raw)
    assert X.shape == (101_766, 54)
    assert int(y.sum()) == 11_357
    assert metadata.negative_count == 90_409
    assert X.loc[0, "total_visits"] == 0
    assert X.loc[0, "hospital_pressure"] == 0


def test_nested_split_manifests_cover_every_role():
    y = pd.Series(([0, 1] * 15), dtype="int8")
    outer, inner, summary = build_nested_cv_splits(y, seed=42)
    assert len(outer) == 30
    assert set(outer.outer_test_fold) == {1, 2, 3, 4, 5}
    assert outer.groupby("source_row_index").size().eq(1).all()
    assert len(inner) == 30 * 4
    assert inner.groupby(["outer_fold", "source_row_index"]).size().eq(1).all()
    assert set(inner.inner_validation_fold) == {1, 2, 3}
    assert summary["outer_manifest_rows"] == 30
    assert summary["inner_manifest_rows"] == 120

