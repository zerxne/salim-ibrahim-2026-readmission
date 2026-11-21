"""SHAP follow-up for the selected 46-field CatBoost (slides 40 and 42).

The deck explains 384 sampled test encounters.  Here every test encounter
(19,942) is explained with two exact tree methods:

* log-odds TreeSHAP (path-dependent), which is additive in the model's raw score
* probability-scale interventional TreeSHAP against 200 development reference rows,
  which is in the same percentage-point unit as slide 40

Outputs (results/tables, results/figures):
  shap_global_importance.csv     mean |SHAP| per feature, both scales, patient-bootstrap CI and rank CI
  shap_sample384_variability.csv how much a 384-encounter sample moves the slide-40 ranking
  shap_reference_sensitivity.csv probability-SHAP stability across reference sets
  shap_category_effects.csv      mean SHAP per category (discharge disposition, admission source, ...)
  shap_numeric_effects.csv       mean SHAP by value for utilisation / stay features
  shap_missed_cases.csv          mean SHAP by alert outcome x prior-inpatient stratum at the top-10% budget
  shap_zero_visit_drivers.csv    what separates flagged from missed readmissions with no prior inpatient visit
  shap_model_agreement.csv       CatBoost vs reproduced HGB vs raw-ICD-9 CatBoost importance ranks
  shap_cross_study_salim.csv     comparison with the Salim & Ibrahim reproduction's SHAP ranking
"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from catboost import CatBoostClassifier, Pool  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import (FIGURES, IDS_MAPPING, MODELS, PREDICTIONS, SALIM, SEED, TABLES, THREADS,  # noqa: E402
                    capacity_flags, load_split, patient_weights, top_k)

REFERENCE_ROWS = 200
BOOTSTRAP = 500
LABELS = {
    "number_inpatient": "Prior inpatient visits", "discharge_disposition_id": "Discharge disposition",
    "number_emergency": "Prior emergency visits", "diag_1": "Primary diagnosis", "payer_code": "Payer code",
    "admission_source_id": "Admission source", "number_diagnoses": "Number of diagnoses",
    "medical_specialty": "Medical specialty", "time_in_hospital": "Length of stay (days)",
    "num_medications": "Number of medications", "number_outpatient": "Prior outpatient visits",
    "age": "Age", "diag_2": "Secondary diagnosis", "diag_3": "Tertiary diagnosis",
    "num_lab_procedures": "Lab procedures", "admission_type_id": "Admission type",
    "num_procedures": "Procedures", "diabetesMed": "Any diabetes medication", "insulin": "Insulin",
    "A1Cresult": "HbA1c result", "race": "Race", "metformin": "Metformin", "change": "Medication change",
    "gender": "Sex", "max_glu_serum": "Max glucose serum",
}
CATEGORY_FEATURES = ["discharge_disposition_id", "admission_source_id", "admission_type_id", "payer_code",
                     "medical_specialty", "diag_1", "diag_2", "diag_3", "A1Cresult", "insulin"]
NUMERIC_FEATURES = ["number_inpatient", "number_emergency", "number_outpatient", "number_diagnoses",
                    "time_in_hospital", "num_medications", "num_lab_procedures", "age"]
BLUE, ORANGE, GREY = "#2a6a9e", "#d9822b", "#8c8c8c"


def label(feature):
    return LABELS.get(feature, feature)


def id_descriptions():
    """Parse the three stacked tables in IDS_mapping.csv."""
    mapping, current = {}, None
    for line in IDS_MAPPING.read_text(encoding="utf-8").splitlines():
        if not line.strip(",").strip():
            current = None
            continue
        key, _, value = line.partition(",")
        if current is None:
            current = key
            mapping[current] = {}
        else:
            mapping[current][key] = value.strip('"')
    return mapping


def catboost_shap(model, matrix, y, categorical, reference):
    pool = Pool(matrix, y, cat_features=categorical)
    raw = model.get_feature_importance(pool, type="ShapValues", thread_count=THREADS)
    prob = model.get_feature_importance(pool, type="ShapValues", model_output="Probability",
                                        reference_data=reference, thread_count=THREADS)
    return raw, prob


def bootstrap_importance(values, groups):
    """Patient-cluster bootstrap of mean |SHAP| and of each feature's rank."""
    absolute = np.abs(values)
    means, ranks = [], []
    for weights in patient_weights(groups, BOOTSTRAP):
        m = weights @ absolute / weights.sum()
        means.append(m)
        ranks.append((-m).argsort().argsort() + 1)
    return np.array(means), np.array(ranks)


def main():
    cohort, features, manifest, assignment, dev, test = load_split()
    categorical, columns = manifest["categorical"], manifest["columns"]
    matrix = features[columns].copy()
    matrix[categorical] = matrix[categorical].fillna("__MISSING__").astype(str)
    x_test = matrix.loc[test.encounter_id]
    y = test.target.to_numpy()
    groups = test.patient_nbr.to_numpy()
    preds = pd.read_csv(PREDICTIONS / "test_46_fields.csv.gz")
    assert (preds.encounter_id.to_numpy() == test.encounter_id.to_numpy()).all()

    model = CatBoostClassifier().load_model(str(MODELS / "catboost.cbm"))
    rng = np.random.default_rng(SEED)
    ref_rows = dev.iloc[rng.choice(len(dev), REFERENCE_ROWS, replace=False)]
    reference = Pool(matrix.loc[ref_rows.encounter_id], ref_rows.target.to_numpy(), cat_features=categorical)
    raw, prob = catboost_shap(model, x_test, y, categorical, reference)
    p = preds.catboost.to_numpy()
    assert np.max(np.abs(raw.sum(1) - logit(p))) < 1e-8, "log-odds SHAP is not additive"
    assert np.max(np.abs(prob.sum(1) - p)) < 1e-8, "probability SHAP is not additive"
    phi, phi_p = raw[:, :-1], prob[:, :-1] * 100  # percentage points
    base_logit, base_prob = raw[0, -1], prob[0, -1]
    np.savez_compressed(PREDICTIONS / "catboost_test_shap.npz", logodds=phi, prob_pp=phi_p,
                        base_logit=base_logit, base_prob=base_prob, features=np.array(columns),
                        encounter_id=test.encounter_id.to_numpy())

    # 1. Global importance with uncertainty ---------------------------------------------------
    boot_p, rank_p = bootstrap_importance(phi_p, groups)
    boot_l, _ = bootstrap_importance(phi, groups)
    glob = pd.DataFrame({
        "feature": columns, "label": [label(c) for c in columns],
        "mean_abs_shap_pp": np.abs(phi_p).mean(0),
        "pp_ci_low": np.quantile(boot_p, .025, axis=0), "pp_ci_high": np.quantile(boot_p, .975, axis=0),
        "mean_abs_shap_logodds": np.abs(phi).mean(0),
        "logodds_ci_low": np.quantile(boot_l, .025, axis=0), "logodds_ci_high": np.quantile(boot_l, .975, axis=0),
        "mean_signed_shap_pp": phi_p.mean(0),
        "rank_ci_low": np.quantile(rank_p, .025, axis=0).astype(int),
        "rank_ci_high": np.quantile(rank_p, .975, axis=0).astype(int),
    }).sort_values("mean_abs_shap_pp", ascending=False)
    glob.insert(0, "rank", np.arange(1, len(glob) + 1))
    glob["share_of_total_pp"] = glob.mean_abs_shap_pp / glob.mean_abs_shap_pp.sum()
    glob["rank_logodds"] = glob.mean_abs_shap_logodds.rank(ascending=False).astype(int)
    glob.to_csv(TABLES / "shap_global_importance.csv", index=False)
    print(glob.head(12).to_string(index=False), flush=True)

    # 2. How variable is a 384-encounter estimate (slide 40)? ------------------------------------
    slide40 = {"number_inpatient": 3.17, "discharge_disposition_id": 1.81, "number_emergency": .65,
               "diag_1": .56, "payer_code": .53}
    top5 = list(glob.feature[:5])
    draws = []
    for _ in range(1000):
        idx = rng.choice(len(phi_p), 384, replace=False)
        m = np.abs(phi_p[idx]).mean(0)
        order = list(np.array(columns)[np.argsort(-m)[:5]])
        draws.append({**{c: m[columns.index(c)] for c in slide40}, "same_top5_set": set(order) == set(top5),
                      "same_top5_order": order == top5, "top2_same": order[:2] == top5[:2]})
    draws = pd.DataFrame(draws)
    variability = pd.DataFrame({
        "feature": list(slide40), "slide40_pp": list(slide40.values()),
        "full_test_pp": [glob.set_index("feature").mean_abs_shap_pp[c] for c in slide40],
        "sample384_p2.5": [draws[c].quantile(.025) for c in slide40],
        "sample384_p97.5": [draws[c].quantile(.975) for c in slide40]})
    variability.to_csv(TABLES / "shap_sample384_variability.csv", index=False)
    pd.Series({"share_same_top5_set": draws.same_top5_set.mean(),
               "share_same_top5_order": draws.same_top5_order.mean(),
               "share_same_top2_order": draws.top2_same.mean()}).to_csv(
        TABLES / "shap_sample384_rank_agreement.csv", header=["value"])

    # 3. Sensitivity to the reference set (first 2,000 test rows, three reference draws) --------
    subset = x_test.iloc[:2000]
    sens = {}
    for seed in [11, 22, 33]:
        rows = dev.iloc[np.random.default_rng(seed).choice(len(dev), REFERENCE_ROWS, replace=False)]
        ref = Pool(matrix.loc[rows.encounter_id], rows.target.to_numpy(), cat_features=categorical)
        values = model.get_feature_importance(Pool(subset, y[:2000], cat_features=categorical), type="ShapValues",
                                              model_output="Probability", reference_data=ref,
                                              thread_count=THREADS)[:, :-1] * 100
        sens[f"reference_seed_{seed}"] = np.abs(values).mean(0)
    sens["main_reference"] = np.abs(phi_p[:2000]).mean(0)
    sens = pd.DataFrame(sens, index=columns)
    sens["max_minus_min"] = sens.max(1) - sens.min(1)
    sens.sort_values("main_reference", ascending=False).to_csv(TABLES / "shap_reference_sensitivity.csv")

    # 4. Category-level effects ----------------------------------------------------------------
    names = id_descriptions()
    name_lookup = {"discharge_disposition_id": names.get("discharge_disposition_id", {}),
                   "admission_source_id": names.get("admission_source_id", {}),
                   "admission_type_id": names.get("admission_type_id", {})}
    rows = []
    for feature in CATEGORY_FEATURES:
        j = columns.index(feature)
        frame = pd.DataFrame({"value": x_test[feature].to_numpy(), "y": y, "pp": phi_p[:, j], "lo": phi[:, j]})
        for value, g in frame.groupby("value"):
            rows.append({"feature": feature, "value": value,
                         "description": name_lookup.get(feature, {}).get(value, ""),
                         "encounters": len(g), "observed_rate": g.y.mean(),
                         "mean_shap_pp": g.pp.mean(), "mean_shap_logodds": g.lo.mean()})
    cats = pd.DataFrame(rows).sort_values(["feature", "mean_shap_pp"], ascending=[True, False])
    cats.to_csv(TABLES / "shap_category_effects.csv", index=False)

    # 5. Numeric effects -------------------------------------------------------------------------
    rows = []
    for feature in NUMERIC_FEATURES:
        j = columns.index(feature)
        values = x_test[feature].to_numpy(dtype=float)
        caps = {"number_inpatient": 5, "number_emergency": 3, "number_outpatient": 3}  # last bin = cap or more
        if feature in caps:
            bins = np.minimum(values, caps[feature])
        elif feature in {"num_medications", "num_lab_procedures"}:
            bins = pd.qcut(values, 8, duplicates="drop")  # ordered intervals
        else:  # age midpoints, length of stay, number of diagnoses: one row per observed value
            bins = values
        frame = pd.DataFrame({"bin": bins, "y": y, "pp": phi_p[:, j], "lo": phi[:, j]})
        for value, g in frame.groupby("bin", sort=True, observed=True):
            rows.append({"feature": feature, "value": value, "encounters": len(g), "observed_rate": g.y.mean(),
                         "mean_shap_pp": g.pp.mean(), "mean_shap_logodds": g.lo.mean()})
    pd.DataFrame(rows).to_csv(TABLES / "shap_numeric_effects.csv", index=False)

    # 6. Missed cases at the top-10% alert budget (slide 42) ---------------------------------------
    k = top_k(len(y))
    flags = capacity_flags(p, k)
    cutoff = float(np.sort(p)[::-1][k - 1])
    inpatient = x_test["number_inpatient"].to_numpy()
    stratum = np.where(inpatient == 0, "0", np.where(inpatient == 1, "1", np.where(inpatient == 2, "2", "3+")))
    outcome = np.where(flags & (y == 1), "TP", np.where(~flags & (y == 1), "FN",
                       np.where(flags & (y == 0), "FP", "TN")))
    rows = []
    for (s, o), idx in pd.DataFrame({"s": stratum, "o": outcome}).groupby(["s", "o"]).groups.items():
        idx = np.asarray(idx)
        rows.append({"prior_inpatient": s, "outcome": o, "encounters": len(idx),
                     "mean_score": p[idx].mean(), "median_score": np.median(p[idx]),
                     **{f"shap_pp_{c}": phi_p[idx, columns.index(c)].mean() for c in glob.feature[:10]}})
    missed = pd.DataFrame(rows).sort_values(["prior_inpatient", "outcome"])
    missed.to_csv(TABLES / "shap_missed_cases.csv", index=False)
    summary = {"alert_budget": k, "score_cutoff": cutoff,
               "missed_positives": int(np.sum(outcome == "FN")),
               "missed_with_zero_prior_inpatient": int(np.sum((outcome == "FN") & (stratum == "0")))}
    zero = stratum == "0"
    summary.update({
        "zero_visit_encounters": int(zero.sum()), "zero_visit_prevalence": float(y[zero].mean()),
        "zero_visit_auroc": roc_auc_score(y[zero], p[zero]), "zero_visit_ap": average_precision_score(y[zero], p[zero]),
        "ge1_visit_auroc": roc_auc_score(y[~zero], p[~zero]), "ge1_visit_ap": average_precision_score(y[~zero], p[~zero]),
        "zero_visit_share_above_cutoff": float(np.mean(p[zero] >= cutoff)),
        "zero_visit_inpatient_shap_pp_mean": float(phi_p[zero, columns.index("number_inpatient")].mean()),
        "zero_visit_fn_median_score": float(np.median(p[zero & (outcome == "FN")])),
    })
    pd.Series(summary).to_csv(TABLES / "shap_missed_summary.csv", header=["value"])
    print(pd.Series(summary).to_string(), flush=True)

    # Which features separate flagged from missed readmissions with zero prior inpatient visits, and
    # readmitted from non-readmitted encounters inside that stratum?
    tp0, fn0, tn0 = zero & (outcome == "TP"), zero & (outcome == "FN"), zero & (outcome == "TN")
    drivers = pd.DataFrame({
        "feature": columns, "label": [label(c) for c in columns],
        "mean_pp_flagged_readmit": phi_p[tp0].mean(0), "mean_pp_missed_readmit": phi_p[fn0].mean(0),
        "mean_pp_true_negative": phi_p[tn0].mean(0),
        "mean_pp_all_readmitted_zero_visit": phi_p[zero & (y == 1)].mean(0),
        "mean_pp_all_not_readmitted_zero_visit": phi_p[zero & (y == 0)].mean(0)})
    drivers["flagged_minus_missed"] = drivers.mean_pp_flagged_readmit - drivers.mean_pp_missed_readmit
    drivers["readmitted_minus_not"] = (drivers.mean_pp_all_readmitted_zero_visit
                                       - drivers.mean_pp_all_not_readmitted_zero_visit)
    drivers = drivers.sort_values("flagged_minus_missed", ascending=False)
    drivers.to_csv(TABLES / "shap_zero_visit_drivers.csv", index=False)

    # 7. Model agreement: reproduced HGB and raw-ICD-9 CatBoost -----------------------------------
    hgb = joblib.load(MODELS / "hgb_pipeline.joblib")
    transformed = hgb.named_steps["preprocess"].transform(features.loc[test.encounter_id, columns])
    hgb_order = manifest["numeric"] + manifest["categorical"]
    explainer = shap.TreeExplainer(hgb.named_steps["model"])
    hgb_phi = np.asarray(explainer.shap_values(transformed))
    assert np.max(np.abs(hgb_phi.sum(1) + explainer.expected_value[0] - logit(preds.hgb))) < 1e-6
    hgb_imp = pd.Series(np.abs(hgb_phi).mean(0), index=hgb_order)[columns]
    raw_matrix = matrix.copy()
    raw_matrix[["diag_1", "diag_2", "diag_3"]] = cohort[["diag_1", "diag_2", "diag_3"]].fillna(
        "__MISSING__").astype(str).to_numpy()
    raw_model = CatBoostClassifier().load_model(str(MODELS / "catboost_raw_icd9.cbm"))
    raw_phi = raw_model.get_feature_importance(Pool(raw_matrix.loc[test.encounter_id], cat_features=categorical),
                                               type="ShapValues", thread_count=THREADS)[:, :-1]
    agree = pd.DataFrame({"feature": columns, "label": [label(c) for c in columns],
                          "catboost_logodds": np.abs(phi).mean(0), "hgb_logodds": hgb_imp.to_numpy(),
                          "catboost_raw_icd9_logodds": np.abs(raw_phi).mean(0)})
    for col in ["catboost_logodds", "hgb_logodds", "catboost_raw_icd9_logodds"]:
        agree[col.replace("_logodds", "_rank")] = agree[col].rank(ascending=False).astype(int)
    agree = agree.sort_values("catboost_rank")
    agree.to_csv(TABLES / "shap_model_agreement.csv", index=False)
    top10 = set(agree.feature[:10])
    stats = {"spearman_catboost_vs_hgb": spearmanr(agree.catboost_logodds, agree.hgb_logodds)[0],
             "spearman_catboost_vs_raw_icd9": spearmanr(agree.catboost_logodds, agree.catboost_raw_icd9_logodds)[0],
             "top10_overlap_hgb": len(top10 & set(agree.nsmallest(10, "hgb_rank").feature)),
             "top10_overlap_raw_icd9": len(top10 & set(agree.nsmallest(10, "catboost_raw_icd9_rank").feature))}
    pd.Series(stats).to_csv(TABLES / "shap_model_agreement_summary.csv", header=["value"])
    print(pd.Series(stats).to_string(), flush=True)

    # 8. Cross-study comparison with the Salim & Ibrahim reproduction ----------------------------
    salim_path = SALIM / "results" / "tables" / "shap_global_importance.csv"
    if salim_path.exists():
        salim = pd.read_csv(salim_path).head(15)
        salim.insert(0, "salim_rank", np.arange(1, len(salim) + 1))
        salim["note"] = np.where(salim.encoded_feature.str.contains("Expired"),
                                 "Death discharges are excluded from our cohort", "")
        salim.to_csv(TABLES / "shap_cross_study_salim.csv", index=False)

    # 9. Attribution sanity check -----------------------------------------------------------------
    # CatBoost builds CTR features from *combinations* of categorical columns and its SHAP splits a
    # combination's contribution equally among member columns.  A column can therefore receive
    # attribution that its own value does not explain (e.g. chlorpropamide, 'No' in 99.9% of rows).
    # (a) own-value R^2: share of the SHAP variance explained by the column's own value;
    # (b) a diagnostic CatBoost without combination CTRs (max_ctr_complexity=1), same settings otherwise.
    own_r2 = {}
    for j, c in enumerate(columns):
        s = pd.Series(phi_p[:, j])
        total = s.var()
        own_r2[c] = float(s.groupby(x_test[c].to_numpy()).transform("mean").var() / total) if total > 0 else np.nan
    single = CatBoostClassifier(iterations=750, depth=6, l2_leaf_reg=15., learning_rate=.035, task_type="CPU",
                                loss_function="Logloss", random_seed=SEED, thread_count=THREADS,
                                max_ctr_complexity=1, allow_writing_files=False, verbose=False)
    single.fit(Pool(matrix.loc[dev.encounter_id], dev.target, cat_features=categorical))
    single.save_model(str(MODELS / "catboost_no_ctr_combinations.cbm"))
    single_pool = Pool(x_test, y, cat_features=categorical)
    p_single = single.predict_proba(single_pool)[:, 1]
    single_phi = single.get_feature_importance(single_pool, type="ShapValues", model_output="Probability",
                                               reference_data=reference, thread_count=THREADS)[:, :-1] * 100
    check = glob[["rank", "feature", "label", "mean_abs_shap_pp"]].copy()
    check["own_value_r2"] = check.feature.map(own_r2)
    check["no_combo_mean_abs_shap_pp"] = check.feature.map(dict(zip(columns, np.abs(single_phi).mean(0))))
    check["no_combo_rank"] = check.no_combo_mean_abs_shap_pp.rank(ascending=False).astype(int)
    check["flag_borrowed_attribution"] = (check.own_value_r2 < .2) & (check["rank"] <= 15)
    check.to_csv(TABLES / "shap_attribution_check.csv", index=False)
    pd.Series({"no_combo_test_ap": average_precision_score(y, p_single),
               "no_combo_test_auroc": roc_auc_score(y, p_single),
               "main_test_ap": average_precision_score(y, p),
               "spearman_main_vs_no_combo": spearmanr(check.mean_abs_shap_pp, check.no_combo_mean_abs_shap_pp)[0],
               "top5_same_set": set(check.feature[:5]) == set(check.nsmallest(5, "no_combo_rank").feature)}).to_csv(
        TABLES / "shap_attribution_check_summary.csv", header=["value"])
    print(check.head(15).to_string(index=False), flush=True)

    flagged = set(check.feature[check.flag_borrowed_attribution])
    make_figures(glob[~glob.feature.isin(flagged)], phi_p, x_test, columns, categorical, cats, missed,
                 drivers[~drivers.feature.isin(flagged)], agree[~agree.feature.isin(flagged)], p, cutoff, y, flagged)


def make_figures(glob, phi_p, x_test, columns, categorical, cats, missed, drivers, agree, p, cutoff, y, flagged):
    note = ("Excluded (attribution borrowed from CatBoost CTR combinations): " + ", ".join(sorted(flagged))) if flagged else ""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Global importance with CI (probability scale)
    top = glob.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(top.label, top.mean_abs_shap_pp, color=BLUE,
            xerr=[top.mean_abs_shap_pp - top.pp_ci_low, top.pp_ci_high - top.mean_abs_shap_pp],
            error_kw={"ecolor": "#333", "capsize": 2, "lw": .8})
    for i, (v, hi) in enumerate(zip(top.mean_abs_shap_pp, top.pp_ci_high)):
        ax.text(hi + .06, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, top.pp_ci_high.max() * 1.12)
    ax.set_xlabel("Mean |SHAP| (percentage points of predicted risk)")
    ax.set_title("CatBoost feature importance, all 19,942 test encounters\n"
                 "error bars: 95% patient-bootstrap interval", fontsize=10)
    fig.text(.01, .01, note, fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(FIGURES / "shap_global_importance.png", dpi=180)
    plt.close(fig)

    # Beeswarm (categorical features are shown in grey: their codes have no order)
    order = list(glob.feature[:15])
    colour = x_test[order].copy()
    for c in order:
        colour[c] = np.nan if c in categorical else colour[c].astype(float)
    shap.summary_plot(phi_p[:, [columns.index(c) for c in order]], colour.astype(float).to_numpy(),
                      feature_names=[label(c) for c in order], show=False, max_display=15, plot_size=(8, 6))
    plt.xlabel("SHAP value (percentage points); grey = categorical feature")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_beeswarm.png", dpi=180)
    plt.close()

    # Discharge disposition categories
    dd = cats[(cats.feature == "discharge_disposition_id") & (cats.encounters >= 50)].copy()
    short = {"1": "Home", "2": "Another short-term hospital", "3": "Skilled nursing facility (SNF)",
             "4": "Intermediate care facility (ICF)", "5": "Another inpatient care institution",
             "6": "Home with home health service", "7": "Left against medical advice",
             "8": "Home under home IV provider", "18": "NULL", "22": "Rehab facility / rehab unit",
             "23": "Long-term care hospital", "25": "Not mapped",
             "28": "Psychiatric hospital / unit", "15": "Swing bed in this institution"}
    dd["name"] = dd.value.map(short).fillna(dd.description.str.slice(0, 30)) + " (ID " + dd.value \
        + ", n=" + dd.encounters.map("{:,}".format) + ")"
    dd = dd.sort_values("mean_shap_pp")
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(dd) + 1.4))
    ax.barh(dd.name, dd.mean_shap_pp, color=np.where(dd.mean_shap_pp > 0, ORANGE, BLUE))
    ax.axvline(0, color="#333", lw=.8)
    ax.set_xlabel("Mean SHAP (percentage points)")
    ax.set_title("Discharge disposition: mean SHAP per category\n(test encounters, categories with n ≥ 50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_discharge_disposition.png", dpi=180)
    plt.close(fig)

    # Dependence of prior inpatient / emergency visits
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, feature in zip(axes, ["number_inpatient", "number_emergency", "number_diagnoses"]):
        j = columns.index(feature)
        x = x_test[feature].to_numpy(dtype=float)
        jitter = np.random.default_rng(0).uniform(-.25, .25, len(x))
        ax.scatter(x + jitter, phi_p[:, j], s=2, alpha=.15, color=BLUE, rasterized=True)
        means = pd.Series(phi_p[:, j]).groupby(x).mean()
        ax.plot(means.index, means.values, color=ORANGE, lw=1.6, marker="o", ms=3)
        ax.axhline(0, color="#333", lw=.6)
        ax.set_xlabel(label(feature))
        ax.set_xlim(-.6, {"number_inpatient": 10, "number_emergency": 8, "number_diagnoses": 16.5}[feature])
    axes[0].set_ylabel("SHAP (percentage points)")
    fig.suptitle("Dependence plots (orange: mean SHAP per value)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_dependence_utilisation.png", dpi=180)
    plt.close(fig)

    # Missed vs flagged readmissions with zero prior inpatient visits
    d = drivers.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    yy = np.arange(len(d))
    ax.barh(yy + .2, d.mean_pp_flagged_readmit, .4, color=ORANGE, label="Flagged readmissions (TP)")
    ax.barh(yy - .2, d.mean_pp_missed_readmit, .4, color=BLUE, label="Missed readmissions (FN)")
    ax.set_yticks(yy, d.label)
    ax.axvline(0, color="#333", lw=.8)
    ax.set_xlabel("Mean SHAP (percentage points)")
    ax.set_title("Zero prior inpatient visits: what lifted the flagged cases above the cutoff", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_zero_visit_missed_vs_flagged.png", dpi=180)
    plt.close(fig)

    # Score distribution of readmissions by prior inpatient stratum relative to the cutoff
    inpatient = x_test["number_inpatient"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for s, mask, colour in [("0 prior inpatient", inpatient == 0, BLUE), ("1–2", (inpatient >= 1) & (inpatient <= 2), GREY),
                            ("3+", inpatient >= 3, ORANGE)]:
        ax.hist(p[mask & (y == 1)] * 100, bins=np.linspace(0, 80, 81), alpha=.6, color=colour, label=s)
    ax.axvline(cutoff * 100, color="#b00", lw=1.2, ls="--")
    ax.text(cutoff * 100 + .8, ax.get_ylim()[1] * .9, f"top-10% cutoff {cutoff:.1%}", color="#b00", fontsize=8)
    ax.set_xlabel("Predicted risk of readmitted encounters (%)")
    ax.set_ylabel("Readmissions")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "score_distribution_readmitted_by_prior_inpatient.png", dpi=180)
    plt.close(fig)

    # Rank agreement CatBoost vs HGB
    a = agree.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    yy = np.arange(len(a))
    ax.barh(yy + .2, a.catboost_logodds, .4, color=BLUE, label="CatBoost")
    ax.barh(yy - .2, a.hgb_logodds, .4, color=ORANGE, label="Reproduced HGB")
    ax.set_yticks(yy, a.label)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("Same 46 fields, two models: importance on the test set", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_catboost_vs_hgb.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
