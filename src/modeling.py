"""Models and evaluation helpers matching the paper where details are available."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class FoldBalancedXGBClassifier(XGBClassifier):
    """XGBoost that recomputes scale_pos_weight for every training fold."""

    def fit(self, X, y, **kwargs):  # noqa: N803 - sklearn API uses X
        labels = np.asarray(y, dtype=int)
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        if positives == 0:
            raise ValueError("A training fold contains no positive examples.")
        self.set_params(scale_pos_weight=negatives / positives)
        return super().fit(X, y, **kwargs)


def make_preprocessor(numeric: Iterable[str], categorical: Iterable[str]):
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(categorical),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def make_xgboost(random_state: int, n_jobs: int) -> FoldBalancedXGBClassifier:
    return FoldBalancedXGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        subsample=1.0,
        colsample_bytree=1.0,
    )


def make_models(
    preprocessor,
    random_state: int,
    n_jobs: int,
    xgb_params: Dict[str, float] | None = None,
) -> Dict[str, Pipeline]:
    xgb_params = xgb_params or {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
    }
    logistic = LogisticRegression(
        class_weight="balanced", max_iter=2_000, random_state=random_state
    )
    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    xgboost = make_xgboost(random_state, n_jobs).set_params(**xgb_params)
    lightgbm = LGBMClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_estimators=100,
        verbosity=-1,
        n_jobs=n_jobs,
    )
    stacking = StackingClassifier(
        estimators=[
            ("logistic", logistic),
            ("xgboost", make_xgboost(random_state, n_jobs).set_params(**xgb_params)),
            (
                "lightgbm",
                LGBMClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                    n_estimators=100,
                    verbosity=-1,
                    n_jobs=n_jobs,
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=2_000, random_state=random_state),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
    )
    return {
        name: Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )
        for name, classifier in {
            "logistic_regression": logistic,
            "random_forest": random_forest,
            "xgboost": xgboost,
            "lightgbm": lightgbm,
            "stacking": stacking,
        }.items()
    }


def threshold_metrics(y_true, probabilities, threshold: float) -> Dict[str, float]:
    predicted = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "high_risk_per_1000": float((tp + fp) / n * 1000),
        "true_readmissions_per_1000": float(tp / n * 1000),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def overall_metrics(y_true, probabilities, classification_threshold=0.5):
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "f1": float(
            f1_score(y_true, np.asarray(probabilities) >= classification_threshold)
        ),
    }


def risk_tertiles(y_true, probabilities) -> pd.DataFrame:
    frame = pd.DataFrame({"target": np.asarray(y_true), "risk": probabilities})
    frame["risk_group"] = pd.qcut(
        frame["risk"], q=3, labels=["low", "medium", "high"], duplicates="drop"
    )
    grouped = frame.groupby("risk_group", observed=True).agg(
        encounters=("target", "size"),
        readmissions=("target", "sum"),
        observed_rate=("target", "mean"),
        mean_predicted_risk=("risk", "mean"),
    )
    return grouped.reset_index()


def bootstrap_intervals(y_true, probabilities, n_bootstrap: int, seed: int):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    records = []
    for bootstrap_id in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot, p_boot = y[idx], p[idx]
        if np.unique(y_boot).size < 2:
            continue
        groups = pd.qcut(p_boot, q=3, labels=False, duplicates="drop")
        low_rate = y_boot[groups == 0].mean()
        high_rate = y_boot[groups == groups.max()].mean()
        records.append(
            {
                "bootstrap_id": bootstrap_id,
                "roc_auc": roc_auc_score(y_boot, p_boot),
                "brier_score": brier_score_loss(y_boot, p_boot),
                "high_vs_low_relative_risk": high_rate / low_rate,
            }
        )
    samples = pd.DataFrame(records)
    summary = []
    for metric in ["roc_auc", "brier_score", "high_vs_low_relative_risk"]:
        summary.append(
            {
                "metric": metric,
                "estimate": float(
                    overall_metrics(y, p)[metric]
                    if metric in {"roc_auc", "brier_score"}
                    else risk_tertiles(y, p).iloc[-1].observed_rate
                    / risk_tertiles(y, p).iloc[0].observed_rate
                ),
                "ci_2.5%": float(samples[metric].quantile(0.025)),
                "ci_97.5%": float(samples[metric].quantile(0.975)),
            }
        )
    return samples, pd.DataFrame(summary)


def decision_curve(y_true, probabilities, thresholds):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    n = len(y)
    prevalence = y.mean()
    records = []
    for threshold in thresholds:
        predicted = p >= threshold
        tp = np.sum(predicted & (y == 1))
        fp = np.sum(predicted & (y == 0))
        odds = threshold / (1 - threshold)
        records.append(
            {
                "threshold": threshold,
                "model_net_benefit": tp / n - fp / n * odds,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
                "high_risk_per_1000": predicted.mean() * 1000,
                "true_readmissions_per_1000": np.sum(predicted & (y == 1))
                / n
                * 1000,
            }
        )
    return pd.DataFrame(records)

