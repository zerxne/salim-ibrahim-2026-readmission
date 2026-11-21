"""Re-run the RQ1 imbalance arms and the RQ2 CDC arms on the same 80/20 split.

The deck reports only AP (and top-10% true positives) for these arms, and the
scripts that produced them are not in the shared repos.  This script rebuilds
them from the slide descriptions with the reproduced HGB configuration, so the
missing metrics (AUROC, Brier, calibration, threshold metrics, paired CIs) can
be filled.  Every arm uses the saved patient-disjoint folds:

* 3-fold development OOF predictions  -> development AP, F1/F2 thresholds, Platt
* refit on all development records    -> held-out test predictions

Arms (training-only treatments; the test set always keeps its observed prevalence):
  original        reproduced HGB, observed prevalence (slide 31)
  balanced        positive weights so both classes carry equal total weight (slide 31)
  duplicate_1to1  repeat observed positives until positives == negatives (slide 32)
  smotenc_1to1    SMOTE-NC synthetic positives to 1:1 (slide 32)
  agesex_control  46 fields + explicit age-sex group key (slide 34 control)
  agesex_cdc      control + six CDC group means + log(1 + group size) (slide 34)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import HERE, KEYS, PREDICTIONS, RAJPOOT, SEED, THREADS, load_split, read_json  # noqa: E402
from readmission.models import build_model  # noqa: E402

CDC_CSV = HERE / "data" / "external" / "cdc" / "cdc_diabetes_health_indicators.csv"
CDC_SHA256 = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
CDC_INDICATORS = ["GenHlth", "PhysHlth", "MentHlth", "DiffWalk", "BMI", "HighChol"]
# BRFSS _AGEG5YR codes -> UCI decade bands.  18-24 and 25-29 are pooled into 20-29;
# 80+ is matched to 80-89.  UCI 0-9, 10-19 and 90-99 have no CDC group and receive
# the overall CDC means with support 0, as described on slide 18.
CDC_AGE_BAND = {1: "[20-30)", 2: "[20-30)", 3: "[30-40)", 4: "[30-40)", 5: "[40-50)", 6: "[40-50)",
                7: "[50-60)", 8: "[50-60)", 9: "[60-70)", 10: "[60-70)", 11: "[70-80)", 12: "[70-80)",
                13: "[80-90)"}
CDC_SEX = {0: "Female", 1: "Male"}


def cdc_lookup():
    from common import sha256
    assert sha256(CDC_CSV) == CDC_SHA256, "Unexpected CDC file"
    cdc = pd.read_csv(CDC_CSV)
    assert len(cdc) == 253680
    cdc["age"] = cdc.Age.map(CDC_AGE_BAND)
    cdc["gender"] = cdc.Sex.map(CDC_SEX)
    lookup = cdc.groupby(["age", "gender"])[CDC_INDICATORS].mean()
    lookup["support"] = cdc.groupby(["age", "gender"]).size()
    overall = cdc[CDC_INDICATORS].mean()
    check = lookup.loc[("[60-70)", "Female")]
    assert int(check.support) == 36014 and round(check.DiffWalk, 4) == 0.2223  # slide 18 example
    return lookup, overall


def add_agesex(features, cohort, manifest, with_cdc):
    x = features.copy()
    x["age_sex_group"] = cohort["age"].astype(str) + "|" + cohort["gender"].astype(str)
    columns = list(manifest["columns"]) + ["age_sex_group"]
    numeric = list(manifest["numeric"])
    categorical = list(manifest["categorical"]) + ["age_sex_group"]
    if with_cdc:
        lookup, overall = cdc_lookup()
        keys = pd.MultiIndex.from_arrays([cohort["age"].astype(str), cohort["gender"].astype(str)])
        joined = lookup.reindex(keys)
        matched = joined.support.notna().to_numpy()
        for col in CDC_INDICATORS:
            x[f"cdc_{col}"] = np.where(matched, joined[col].to_numpy(), overall[col])
        x["cdc_log_support"] = np.log1p(np.where(matched, joined.support.to_numpy(), 0))
        new = [f"cdc_{c}" for c in CDC_INDICATORS] + ["cdc_log_support"]
        columns += new
        numeric += new
        print(f"CDC match rate: {matched.mean():.4%} of encounters", flush=True)
    return x, {"columns": columns, "numeric": numeric, "categorical": categorical}


def make_fit(arm, manifest, config):
    """Return fit(X, y) -> fitted model exposing predict_proba."""
    def fit(x, y):
        if arm == "balanced":
            model = build_model("hgb", manifest, config)
            return model.fit(x, y, model__sample_weight=compute_sample_weight("balanced", y))
        if arm == "duplicate_1to1":
            xr, yr = RandomOverSampler(sampling_strategy=1.0, random_state=SEED).fit_resample(x, y)
            return build_model("hgb", manifest, config).fit(xr, yr)
        if arm == "smotenc_1to1":
            base = build_model("hgb", manifest, config)
            categorical_idx = list(range(len(manifest["numeric"]),
                                         len(manifest["numeric"]) + len(manifest["categorical"])))
            model = ImbPipeline([("preprocess", base.named_steps["preprocess"]),
                                 ("smote", SMOTENC(categorical_features=categorical_idx, sampling_strategy=1.0,
                                                   random_state=SEED)),
                                 ("model", base.named_steps["model"])])
            return model.fit(x, y)
        return build_model("hgb", manifest, config).fit(x, y)
    return fit


def run_arm(arm, x, manifest, config, dev, test):
    fit = make_fit(arm, manifest, config)
    cols = manifest["columns"]
    oof = np.full(len(dev), np.nan)
    for fold in sorted(dev.cv_validation_fold.unique()):
        train, val = dev.cv_validation_fold.ne(fold), dev.cv_validation_fold.eq(fold)
        model = fit(x.loc[dev.encounter_id[train], cols], dev.target[train].to_numpy())
        oof[val.to_numpy()] = model.predict_proba(x.loc[dev.encounter_id[val], cols])[:, 1]
    model = fit(x.loc[dev.encounter_id, cols], dev.target.to_numpy())
    return oof, model.predict_proba(x.loc[test.encounter_id, cols])[:, 1]


def main():
    config = read_json(RAJPOOT / "configs" / "paper.json")
    cohort, features, manifest, assignment, dev, test = load_split()
    dev_out, test_out = dev[KEYS + ["cv_validation_fold"]].copy(), test[KEYS].copy()
    variants = {"original": (features, manifest), "balanced": (features, manifest),
                "duplicate_1to1": (features, manifest), "smotenc_1to1": (features, manifest),
                "agesex_control": add_agesex(features, cohort, manifest, with_cdc=False),
                "agesex_cdc": add_agesex(features, cohort, manifest, with_cdc=True)}
    for arm, (x, arm_manifest) in variants.items():
        start = time.time()
        dev_out[arm], test_out[arm] = run_arm(arm, x, arm_manifest, config, dev, test)
        print(f"{arm:15s} dev AP {average_precision_score(dev.target, dev_out[arm]):.5f}  "
              f"test AP {average_precision_score(test.target, test_out[arm]):.5f}  "
              f"({time.time() - start:.0f}s)", flush=True)
    saved = pd.read_csv(PREDICTIONS / "test_46_fields.csv.gz")
    gap = np.max(np.abs(saved.hgb.to_numpy() - test_out.original.to_numpy()))
    assert gap < 1e-6, f"original arm should equal reproduced HGB (gap {gap})"
    dev_out.to_csv(PREDICTIONS / "dev_oof_rq1_cdc.csv.gz", index=False)
    test_out.to_csv(PREDICTIONS / "test_rq1_cdc.csv.gz", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=THREADS):
        main()
