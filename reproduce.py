#!/usr/bin/env python3
"""End-to-end reproduction of Salim and Ibrahim (Healthcare, 2026)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features import engineer_features
from src.modeling import (
    bootstrap_intervals,
    decision_curve,
    make_models,
    make_preprocessor,
    make_xgboost,
    overall_metrics,
    risk_tertiles,
    threshold_metrics,
)
from src.plotting import (
    calibration,
    decision_and_impact,
    discrimination,
    model_comparison,
    risk_stratification,
)
from src.splits import export_nested_cv_splits


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "artifacts" / "models"
SPLITS = ROOT / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--shap-stability-iterations", type=int, default=100)
    parser.add_argument("--shap-jobs", type=int, default=2)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-model-save", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 20k rows, two outer folds, one grid point, no SHAP stability.",
    )
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def select_final_params(best_params: list[dict]) -> dict:
    normalized = [
        tuple(sorted((key.replace("classifier__", ""), value) for key, value in params.items()))
        for params in best_params
    ]
    selected = Counter(normalized).most_common(1)[0][0]
    return dict(selected)


def nested_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    outer_cv,
    seed: int,
    threads: int,
    quick: bool,
):
    grid = (
        {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [3],
            "classifier__learning_rate": [0.1],
        }
        if quick
        else {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.03, 0.1],
        }
    )
    oof = np.full(len(y), np.nan, dtype=float)
    records, chosen_params = [], []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        started = time.time()
        pipeline = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("classifier", make_xgboost(seed + fold, threads)),
            ]
        )
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities = search.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = probabilities
        auc = roc_auc_score(y.iloc[test_idx], probabilities)
        f1 = f1_score(y.iloc[test_idx], probabilities >= 0.5)
        record = {
            "fold": fold,
            "roc_auc": auc,
            "f1_at_0.5": f1,
            "best_inner_auc": search.best_score_,
            "elapsed_seconds": time.time() - started,
            **{
                key.replace("classifier__", ""): value
                for key, value in search.best_params_.items()
            },
        }
        records.append(record)
        chosen_params.append(search.best_params_)
        print(
            f"XGBoost outer fold {fold}: AUC={auc:.4f}, "
            f"inner={search.best_score_:.4f}, params={search.best_params_}",
            flush=True,
        )
    if np.isnan(oof).any():
        raise RuntimeError("Nested CV did not generate every out-of-fold prediction.")
    return pd.DataFrame(records), oof, select_final_params(chosen_params)


def evaluate_comparators(X, y, preprocessor, outer_splits, seed, threads, xgb_params):
    models = make_models(preprocessor, seed, threads, xgb_params=xgb_params)
    fold_rows = []
    oof_by_model = {}
    for model_name in ["logistic_regression", "random_forest", "lightgbm", "stacking"]:
        print(f"Evaluating {model_name} ...", flush=True)
        oof = np.full(len(y), np.nan, dtype=float)
        for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            started = time.time()
            fitted = clone(models[model_name])
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            probability = fitted.predict_proba(X.iloc[test_idx])[:, 1]
            oof[test_idx] = probability
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "roc_auc": roc_auc_score(y.iloc[test_idx], probability),
                    "f1_at_0.5": f1_score(y.iloc[test_idx], probability >= 0.5),
                    "elapsed_seconds": time.time() - started,
                }
            )
        oof_by_model[model_name] = oof
    return pd.DataFrame(fold_rows), oof_by_model


def shap_analysis(final_pipeline, X, y, args):
    import shap

    preprocessor = final_pipeline.named_steps["preprocessor"]
    classifier = final_pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    rng = np.random.default_rng(args.seed)
    explain_idx = rng.choice(len(y), size=min(5_000, len(y)), replace=False)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(transformed[explain_idx])
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    importance = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(TABLES / "shap_global_importance.csv", index=False)

    import matplotlib.pyplot as plt

    shap.summary_plot(
        values,
        transformed[explain_idx],
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    iterations = 0 if args.quick else args.shap_stability_iterations
    if iterations <= 0:
        return

    def one_stability_run(iteration: int):
        local_rng = np.random.default_rng(args.seed + 10_000 + iteration)
        boot = local_rng.integers(0, len(y), size=len(y))
        local_threads = max(1, args.threads // max(1, args.shap_jobs))
        estimator = clone(classifier).set_params(n_jobs=local_threads)
        estimator.fit(transformed[boot], y.iloc[boot])
        explain = local_rng.choice(boot, size=min(1_000, len(boot)), replace=False)
        local_values = shap.TreeExplainer(estimator)(transformed[explain]).values
        local_values = np.asarray(local_values)
        if local_values.ndim == 3:
            local_values = local_values[:, :, 1]
        top = np.argsort(np.abs(local_values).mean(axis=0))[-10:]
        return feature_names[top].tolist()

    rankings = Parallel(n_jobs=args.shap_jobs, verbose=5)(
        delayed(one_stability_run)(iteration) for iteration in range(iterations)
    )
    counts = Counter(feature for ranking in rankings for feature in ranking)
    stability = pd.DataFrame(
        [
            {
                "encoded_feature": feature,
                "top_10_count": count,
                "top_10_frequency": count / iterations,
            }
            for feature, count in counts.items()
        ]
    ).sort_values(["top_10_frequency", "encoded_feature"], ascending=[False, True])
    stability.to_csv(TABLES / "shap_top10_stability.csv", index=False)


def main() -> None:
    args = parse_args()
    for directory in (TABLES, FIGURES, MODELS):
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data, low_memory=False)
    X, y, metadata = engineer_features(raw)
    if not args.quick:
        export_nested_cv_splits(y, SPLITS, seed=args.seed, source_csv=args.data)
    if args.quick:
        # Preserve prevalence while providing a fast end-to-end smoke test.
        negative = y[y == 0].sample(n=17_768, random_state=args.seed).index
        positive = y[y == 1].sample(n=2_232, random_state=args.seed).index
        sampled = np.concatenate([negative, positive])
        sampled = np.random.default_rng(args.seed).permutation(sampled)
        X, y = X.loc[sampled].reset_index(drop=True), y.loc[sampled].reset_index(drop=True)

    write_json(TABLES / "dataset_metadata.json", metadata.as_dict())
    write_json(
        TABLES / "runtime_versions.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "matplotlib": matplotlib.__version__,
            "seed": args.seed,
            "threads": args.threads,
            "quick_mode": args.quick,
        },
    )

    preprocessor = make_preprocessor(metadata.numeric_features, metadata.categorical_features)
    outer_folds = 2 if args.quick else 5
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=args.seed)
    outer_splits = list(outer_cv.split(X, y))

    xgb_folds, xgb_oof, final_params = nested_xgboost(
        X, y, preprocessor, outer_cv, args.seed, args.threads, args.quick
    )
    xgb_folds.insert(0, "model", "xgboost")
    comparator_folds, comparator_oof = evaluate_comparators(
        X, y, preprocessor, outer_splits, args.seed, args.threads, final_params
    )
    all_folds = pd.concat([xgb_folds, comparator_folds], ignore_index=True)
    all_folds.to_csv(TABLES / "outer_fold_metrics.csv", index=False)
    comparison = (
        all_folds.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_f1_at_05=("f1_at_0.5", "mean"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)
    model_comparison(comparison, FIGURES / "model_comparison.png")
    write_json(TABLES / "selected_xgboost_params.json", final_params)

    oof_metrics = overall_metrics(y, xgb_oof)
    write_json(TABLES / "nested_xgboost_oof_metrics.json", oof_metrics)
    oof_frame = pd.DataFrame({"target": y, "xgboost": xgb_oof})
    for name, probabilities in comparator_oof.items():
        oof_frame[name] = probabilities
    oof_frame.to_csv(TABLES / "out_of_fold_predictions.csv", index=False)

    final_pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                make_xgboost(args.seed, args.threads).set_params(**final_params),
            ),
        ]
    )
    final_pipeline.fit(X, y)
    calibrated = CalibratedClassifierCV(
        estimator=clone(final_pipeline), method="sigmoid", cv=5
    )
    calibrated.fit(X, y)
    calibrated_probability = calibrated.predict_proba(X)[:, 1]
    apparent_metrics = overall_metrics(y, calibrated_probability)
    write_json(TABLES / "calibrated_full_data_apparent_metrics.json", apparent_metrics)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    threshold_frame = pd.DataFrame(
        [threshold_metrics(y, calibrated_probability, value) for value in thresholds]
    )
    threshold_frame.to_csv(TABLES / "threshold_metrics.csv", index=False)

    tertiles = risk_tertiles(y, calibrated_probability)
    tertiles.to_csv(TABLES / "risk_tertiles.csv", index=False)
    risk_stratification(tertiles, FIGURES / "risk_stratification.png")

    bootstrap_samples, bootstrap_summary = bootstrap_intervals(
        y, calibrated_probability, args.bootstrap_iterations, args.seed
    )
    bootstrap_samples.to_csv(TABLES / "bootstrap_samples.csv", index=False)
    bootstrap_summary.to_csv(TABLES / "bootstrap_summary.csv", index=False)

    curve = decision_curve(y, calibrated_probability, np.linspace(0.01, 0.30, 100))
    curve.to_csv(TABLES / "decision_and_clinical_impact_curve.csv", index=False)
    decision_and_impact(
        curve,
        FIGURES / "decision_curve.png",
        FIGURES / "clinical_impact_curve.png",
    )
    discrimination(
        y,
        calibrated_probability,
        FIGURES / "roc_curve.png",
        FIGURES / "precision_recall_curve.png",
    )
    calibration(y, calibrated_probability, FIGURES / "calibration_curve.png")

    if not args.skip_shap:
        shap_analysis(final_pipeline, X, y, args)
    if not args.skip_model_save:
        joblib.dump(final_pipeline, MODELS / "xgboost_uncalibrated.joblib", compress=3)
        joblib.dump(calibrated, MODELS / "xgboost_platt_calibrated.joblib", compress=3)

    print("\nReproduction completed.")
    print(comparison.to_string(index=False))
    print(f"Nested XGBoost OOF metrics: {oof_metrics}")
    print(f"Calibrated full-data apparent metrics: {apparent_metrics}")


if __name__ == "__main__":
    main()

