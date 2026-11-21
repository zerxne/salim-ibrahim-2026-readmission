"""Fill the metrics that the final deck does not report (all on the same 80/20 test set).

Outputs (results/tables):
  metrics_46_fields.csv            AP/AUROC/Brier/log-loss/calibration/ECE + patient-bootstrap 95% CIs
  operating_points_46_fields.csv   precision/recall/F1/F2/specificity/NPV at top-10%, dev-F1 and dev-F2 rules
  precision_at_recall.csv          slide 36 values as numbers
  calibration_deciles.csv          slide 39 values as numbers (before / after Platt)
  subgroup_metrics.csv             AUROC/AP/observed vs expected by sex, age, race, prior inpatient visits
  metrics_rq1_cdc.csv              same metric set for the re-run imbalance and CDC arms
  operating_points_rq1_cdc.csv     operating points for those arms (thresholds chosen on their own OOF)
  paired_differences.csv           paired patient-bootstrap CIs for every contrast shown in the deck
  literature_comparison.csv        our models next to the five reviewed papers on shared metrics
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (PREDICTIONS, REFERENCE, TABLES, capacity_flags, confusion_metrics, fit_dev_threshold,  # noqa: E402
                    load_split, patient_weights, platt, probability_metrics, read_json, top_k)

REPS = 1000
MAIN = ["hgb", "catboost", "blend", "catboost_raw_icd9", "hgb_platt", "catboost_platt", "blend_platt"]
ARMS = ["original", "balanced", "duplicate_1to1", "smotenc_1to1", "agesex_control", "agesex_cdc"]
DECK_AP = {"original": .23737, "balanced": .23458, "duplicate_1to1": .23241, "smotenc_1to1": .20900,
           "agesex_control": .23945, "agesex_cdc": .23927}
DECK_TOP10_TP = {"original": 565, "balanced": 580, "duplicate_1to1": 556, "smotenc_1to1": 501}


def boot_metrics(y, preds: dict, groups):
    rows = []
    for weights in patient_weights(groups, REPS):
        row = {}
        for name, p in preds.items():
            row[(name, "ap")] = average_precision_score(y, p, sample_weight=weights)
            row[(name, "auroc")] = roc_auc_score(y, p, sample_weight=weights)
            row[(name, "brier")] = np.average((y - p) ** 2, weights=weights)
        rows.append(row)
    return pd.DataFrame(rows)


def with_ci(y, preds, groups, boot):
    rows = []
    for name, p in preds.items():
        row = {"method": name, **probability_metrics(y, p)}
        for metric in ["ap", "auroc", "brier"]:
            lo, hi = np.quantile(boot[(name, metric)], [.025, .975])
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired(boot, y, preds, contrasts, family):
    rows = []
    for treatment, control in contrasts:
        for metric, fn in [("ap", average_precision_score), ("auroc", roc_auc_score),
                           ("brier", lambda t, p: np.mean((t - p) ** 2))]:
            diff = boot[(treatment, metric)] - boot[(control, metric)]
            rows.append({"family": family, "treatment": treatment, "control": control, "metric": metric,
                         "difference": fn(y, preds[treatment]) - fn(y, preds[control]),
                         "ci_low": diff.quantile(.025), "ci_high": diff.quantile(.975),
                         "ci_excludes_zero": bool(diff.quantile(.025) > 0 or diff.quantile(.975) < 0)})
    return rows


def operating(y, p, rules: dict, name):
    rows = []
    k = top_k(len(y))
    rows.append({"method": name, "rule": "top 10% capacity", "threshold": float(np.sort(p)[::-1][k - 1]),
                 **confusion_metrics(y, capacity_flags(p, k))})
    for rule, threshold in rules.items():
        rows.append({"method": name, "rule": rule, "threshold": threshold, **confusion_metrics(y, p >= threshold)})
    return rows


def precision_at_recall(y, p, levels=(.2, .4, .6, .8)):
    precision, recall, _ = precision_recall_curve(y, p)
    # highest threshold that reaches the recall level = largest precision index with recall >= level
    return {f"precision_at_recall_{int(r * 100)}": float(precision[np.where(recall >= r)[0][-1]]) for r in levels}


def main():
    cohort, features, manifest, assignment, dev, test = load_split()
    lock = read_json(REFERENCE / "selection_lock.json")
    main_preds = pd.read_csv(PREDICTIONS / "test_46_fields.csv.gz")
    y, groups = main_preds.target.to_numpy(), main_preds.patient_nbr.to_numpy()
    preds = {m: main_preds[m].to_numpy() for m in MAIN}

    # 1. 46-field models -------------------------------------------------------------------------
    boot = boot_metrics(y, preds, groups)
    table = with_ci(y, preds, groups, boot)
    reference = pd.read_csv(REFERENCE / "test_metrics.csv").set_index("method")
    table["matches_team_reference_ap"] = [abs(r.ap - reference.ap.get(r.method, np.nan)) < 1e-9 for r in table.itertuples()]
    table = table.join(pd.DataFrame([precision_at_recall(y, preds[m]) for m in MAIN]))
    table.to_csv(TABLES / "metrics_46_fields.csv", index=False)

    ops = []
    for name in ["hgb", "catboost", "blend", "catboost_raw_icd9"]:
        rules = {}
        if name in lock["thresholds"]:
            rules = {"dev-F1 threshold": lock["thresholds"][name]["f1"], "dev-F2 threshold": lock["thresholds"][name]["f2"]}
        ops += operating(y, preds[name], rules, name)
    ops = pd.DataFrame(ops)
    ops.to_csv(TABLES / "operating_points_46_fields.csv", index=False)

    pd.DataFrame([{"method": m, **precision_at_recall(y, preds[m])} for m in ["hgb", "catboost"]]).to_csv(
        TABLES / "precision_at_recall.csv", index=False)

    deciles = []
    for name in ["hgb", "catboost", "catboost_platt"]:
        frame = pd.DataFrame({"y": y, "p": preds[name]})
        frame["decile"] = pd.qcut(frame.p.rank(method="first"), 10, labels=range(1, 11))
        g = frame.groupby("decile", observed=True).agg(encounters=("y", "size"), mean_predicted=("p", "mean"),
                                                        observed_rate=("y", "mean"), positives=("y", "sum"))
        g["observed_minus_predicted"] = g.observed_rate - g.mean_predicted
        deciles.append(g.reset_index().assign(method=name))
    pd.concat(deciles).to_csv(TABLES / "calibration_deciles.csv", index=False)

    # 2. Subgroups (not reported in the deck) ------------------------------------------------------
    raw = cohort.loc[test.encounter_id]
    age = features.loc[test.encounter_id, "age"].to_numpy()
    inpatient = features.loc[test.encounter_id, "number_inpatient"].to_numpy()
    subgroups = {
        "sex": raw.gender.to_numpy(),
        "age_band": np.where(age < 50, "<50", np.where(age < 70, "50-69", "70+")),
        "race": raw.race.fillna("Missing").to_numpy(),
        "prior_inpatient": np.where(inpatient == 0, "0", np.where(inpatient <= 2, "1-2", "3+")),
    }
    rows = []
    for variable, values in subgroups.items():
        for value in pd.unique(values):
            mask = values == value
            if mask.sum() < 200 or y[mask].sum() < 20:
                continue
            for name in ["hgb", "catboost", "catboost_platt"]:
                p = preds[name][mask]
                rows.append({"variable": variable, "group": value, "method": name, "encounters": int(mask.sum()),
                             "positives": int(y[mask].sum()), "prevalence": y[mask].mean(),
                             "auroc": roc_auc_score(y[mask], p), "ap": average_precision_score(y[mask], p),
                             "ap_lift_over_prevalence": average_precision_score(y[mask], p) / y[mask].mean(),
                             "mean_predicted": p.mean(), "observed_to_expected": y[mask].mean() / p.mean(),
                             "share_flagged_top10": capacity_flags(preds[name], top_k(len(y)))[mask].mean(),
                             "recall_top10": capacity_flags(preds[name], top_k(len(y)))[mask & (y == 1)].mean()})
    pd.DataFrame(rows).to_csv(TABLES / "subgroup_metrics.csv", index=False)

    # 3. Re-run RQ1 / CDC arms -------------------------------------------------------------------
    arm_test = pd.read_csv(PREDICTIONS / "test_rq1_cdc.csv.gz")
    arm_dev = pd.read_csv(PREDICTIONS / "dev_oof_rq1_cdc.csv.gz")
    assert (arm_test.encounter_id.to_numpy() == main_preds.encounter_id.to_numpy()).all()
    y_dev = arm_dev.target.to_numpy()
    arm_preds = {a: arm_test[a].to_numpy() for a in ARMS}
    # Platt recalibration fitted on each arm's development OOF predictions (shows whether resampling
    # damage to probabilities is repairable without touching the test set).
    for a in ARMS:
        z = logit(np.clip(arm_dev[a].to_numpy(), 1e-12, 1 - 1e-12)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_dev)
        arm_preds[f"{a}_platt"] = platt(arm_test[a], {"intercept": lr.intercept_[0], "slope": lr.coef_[0, 0]})
    boot_arms = boot_metrics(y, arm_preds, groups)
    arms = with_ci(y, arm_preds, groups, boot_arms)
    arms["dev_ap"] = [average_precision_score(y_dev, arm_dev[m.replace("_platt", "")]) for m in arms.method]
    arms["deck_test_ap"] = arms.method.map(DECK_AP)
    arms["deck_top10_tp"] = arms.method.map(DECK_TOP10_TP)
    arms.to_csv(TABLES / "metrics_rq1_cdc.csv", index=False)

    ops = []
    for a in ARMS:
        p_dev = arm_dev[a].to_numpy()
        rules = {"dev-F1 threshold": fit_dev_threshold(y_dev, p_dev, 1.0),
                 "dev-F2 threshold": fit_dev_threshold(y_dev, p_dev, 2.0)}
        ops += operating(y, arm_preds[a], rules, a)
    pd.DataFrame(ops).to_csv(TABLES / "operating_points_rq1_cdc.csv", index=False)

    # 4. Paired contrasts --------------------------------------------------------------------------
    rows = paired(boot, y, preds, [("catboost", "hgb"), ("blend", "hgb"), ("blend", "catboost"),
                                   ("catboost_raw_icd9", "catboost"), ("catboost_platt", "catboost")], "46_fields")
    rows += paired(boot_arms, y, arm_preds, [("balanced", "original"), ("duplicate_1to1", "original"),
                                             ("smotenc_1to1", "original"), ("duplicate_1to1", "smotenc_1to1"),
                                             ("agesex_control", "original"), ("agesex_cdc", "agesex_control"),
                                             ("balanced_platt", "original"), ("duplicate_1to1_platt", "original"),
                                             ("smotenc_1to1_platt", "original")], "rq1_cdc_rerun")
    pd.DataFrame(rows).to_csv(TABLES / "paired_differences.csv", index=False)

    # 5. Literature comparison on shared metrics ---------------------------------------------------
    f1 = pd.read_csv(TABLES / "operating_points_46_fields.csv").set_index(["method", "rule"]).f1
    m = table.set_index("method")
    lit = pd.DataFrame([
        {"study": "Liu, Sue & Wu 2024 [7]", "split": "patient-grouped 5-fold CV", "auroc": .64, "ap": np.nan,
         "positive_f1": np.nan, "brier": np.nan, "note": "published F1 0.84 is not positive-class F1"},
        {"study": "Emi-Johnson & Nkrumah 2025 [8]", "split": "stratified 80/20 encounters", "auroc": .667,
         "ap": np.nan, "positive_f1": .032, "brier": np.nan, "note": "XGBoost; DNN reached F1 0.161 at AUROC 0.579"},
        {"study": "Salim & Ibrahim 2026 [9]", "split": "nested CV, not patient-disjoint", "auroc": .664, "ap": .215,
         "positive_f1": .271, "brier": np.nan, "note": "XGBoost"},
        {"study": "Rajpoot 2026 [10] (published)", "split": "patient-disjoint 80/20", "auroc": .677, "ap": .237,
         "positive_f1": np.nan, "brier": np.nan, "note": "HGB, discharge-time fields"},
        {"study": "Ours: reproduced HGB", "split": "same patient-disjoint 80/20", "auroc": m.auroc["hgb"],
         "ap": m.ap["hgb"], "positive_f1": f1["hgb", "dev-F1 threshold"], "brier": m.brier["hgb"],
         "note": "F1 at development-selected threshold"},
        {"study": "Ours: 46-field CatBoost", "split": "same patient-disjoint 80/20", "auroc": m.auroc["catboost"],
         "ap": m.ap["catboost"], "positive_f1": f1["catboost", "dev-F1 threshold"], "brier": m.brier["catboost"],
         "note": "F1 at development-selected threshold"},
    ])
    lit.to_csv(TABLES / "literature_comparison.csv", index=False)

    pd.set_option("display.width", 220)
    print(table[["method", "ap", "ap_ci_low", "ap_ci_high", "auroc", "brier", "ece_deciles",
                 "calibration_intercept", "calibration_slope"]].round(5).to_string(index=False))
    print(pd.read_csv(TABLES / "operating_points_46_fields.csv")[
        ["method", "rule", "threshold", "alerts", "tp", "precision", "recall", "f1", "f2", "specificity"]].round(4).to_string(index=False))
    print(arms[["method", "dev_ap", "ap", "deck_test_ap", "auroc", "brier", "mean_predicted", "calibration_slope", "ece_deciles"]].round(5).to_string(index=False))
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
