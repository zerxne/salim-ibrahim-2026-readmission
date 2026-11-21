"""Refit the deck's 46-field models from the team's locked settings and save test predictions.

* CatBoost (ICD-9 families) and CatBoost (raw ICD-9) use the exact parameters in
  hgb_catboost/reference/46_fields/selection_lock.json and must reproduce the
  team's reference test AP to 1e-9.
* The reproduced HGB is refit so SHAP has a model object; its predictions must
  match the saved Rajpoot reproduction predictions.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (KEYS, MODELS, PAPER_OOF, PAPER_TEST, PREDICTIONS, RAJPOOT, REFERENCE, SEED,  # noqa: E402
                    THREADS, load_split, platt, read_json)
from readmission.models import build_model  # noqa: E402

DIAGNOSIS = ["diag_1", "diag_2", "diag_3"]


def catboost_matrix(features, categorical):
    matrix = features.copy()
    matrix[categorical] = matrix[categorical].fillna("__MISSING__").astype(str)
    return matrix


def main():
    lock = read_json(REFERENCE / "selection_lock.json")
    reference = pd.read_csv(REFERENCE / "test_metrics.csv").set_index("method")
    cohort, features, manifest, assignment, dev, test = load_split()
    categorical = manifest["categorical"]

    paper_test = pd.read_csv(PAPER_TEST, float_precision="round_trip")
    pd.testing.assert_frame_equal(test[KEYS], paper_test[KEYS])
    out = test[KEYS + ["partition"]].copy()
    out["hgb"] = paper_test.discharge_hgb.to_numpy()

    grouped = catboost_matrix(features, categorical)
    raw = features.copy()
    raw[DIAGNOSIS] = cohort[DIAGNOSIS].to_numpy()
    raw = catboost_matrix(raw, categorical)
    params = dict(iterations=lock["trees"], depth=lock["depth"],
                  l2_leaf_reg={4: 5., 6: 15.}[lock["depth"]], learning_rate=.035, task_type="CPU",
                  loss_function="Logloss", random_seed=SEED, thread_count=THREADS,
                  allow_writing_files=False, verbose=False)
    for name, matrix in [("catboost", grouped), ("catboost_raw_icd9", raw)]:
        model = CatBoostClassifier(**params)
        model.fit(Pool(matrix.loc[dev.encounter_id], dev.target, cat_features=categorical))
        model.save_model(str(MODELS / f"{name}.cbm"))
        out[name] = model.predict_proba(Pool(matrix.loc[test.encounter_id], cat_features=categorical),
                                        thread_count=THREADS)[:, 1]
        ap = average_precision_score(out.target, out[name])
        print(f"{name}: AP {ap:.10f} (reference {reference.loc[name, 'ap']:.10f})", flush=True)
        assert abs(ap - reference.loc[name, "ap"]) < 1e-9, f"{name} does not reproduce the reference AP"

    weight = lock["blend_catboost_weight"]
    out["blend"] = (1 - weight) * out.hgb + weight * out.catboost
    for name, coefficients in lock["platt"].items():
        out[f"{name}_platt"] = platt(out[name], coefficients)

    # Refit HGB to obtain a model object for SHAP; predictions must match the saved ones.
    config = read_json(RAJPOOT / "configs" / "paper.json")
    hgb = build_model("hgb", manifest, config)
    hgb.fit(features.loc[dev.encounter_id, manifest["columns"]], dev.target)
    refit = hgb.predict_proba(features.loc[test.encounter_id, manifest["columns"]])[:, 1]
    gap = float(np.max(np.abs(refit - out.hgb)))
    print(f"HGB refit max |diff| vs saved predictions: {gap:.2e}")
    assert gap < 1e-6
    joblib.dump(hgb, MODELS / "hgb_pipeline.joblib", compress=3)

    out.to_csv(PREDICTIONS / "test_46_fields.csv.gz", index=False)
    oof = pd.read_csv(PAPER_OOF, float_precision="round_trip")
    oof[KEYS + ["cv_validation_fold", "discharge_hgb"]].rename(columns={"discharge_hgb": "hgb"}).to_csv(
        PREDICTIONS / "dev_oof_hgb.csv.gz", index=False)
    print(out.drop(columns=KEYS + ["partition"]).describe().T[["mean", "min", "max"]])


if __name__ == "__main__":
    with threadpool_limits(limits=THREADS):
        main()
