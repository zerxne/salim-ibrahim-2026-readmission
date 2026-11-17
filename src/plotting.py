"""Publication-style plots for the reproduction outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def model_comparison(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("mean_roc_auc")
    plt.figure(figsize=(7, 4.5))
    plt.barh(ordered["model"], ordered["mean_roc_auc"], color="#3572A5")
    plt.errorbar(
        ordered["mean_roc_auc"],
        ordered["model"],
        xerr=ordered["std_roc_auc"],
        fmt="none",
        color="black",
        capsize=3,
    )
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlim(0.5, max(0.70, ordered["mean_roc_auc"].max() + 0.02))
    plt.xlabel("Outer-fold ROC-AUC (mean ± SD)")
    plt.title("Nested / outer cross-validation model comparison")
    _finish(path)


def discrimination(y_true, probabilities, roc_path: Path, pr_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, color="#16697A", linewidth=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve: calibrated full-data predictions")
    _finish(roc_path)

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall, precision, color="#D95F02", linewidth=2)
    plt.axhline(np.mean(y_true), color="gray", linestyle="--", label="Prevalence")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    _finish(pr_path)


def calibration(y_true, probabilities, path: Path) -> None:
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="quantile"
    )
    plt.figure(figsize=(5.5, 5))
    plt.plot(predicted, observed, marker="o", label="Calibrated XGBoost")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve (apparent full-data evaluation)")
    plt.legend()
    _finish(path)


def decision_and_impact(frame: pd.DataFrame, dca_path: Path, cic_path: Path) -> None:
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(frame.threshold, frame.model_net_benefit, label="Model", linewidth=2)
    plt.plot(frame.threshold, frame.treat_all_net_benefit, label="Treat all")
    plt.plot(frame.threshold, frame.treat_none_net_benefit, label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis")
    plt.legend()
    _finish(dca_path)

    plt.figure(figsize=(6.5, 4.8))
    plt.plot(
        frame.threshold,
        frame.high_risk_per_1000,
        label="Classified high risk / 1000",
    )
    plt.plot(
        frame.threshold,
        frame.true_readmissions_per_1000,
        label="True readmissions / 1000",
    )
    plt.xlabel("Threshold probability")
    plt.ylabel("Patients per 1000")
    plt.title("Clinical impact curve")
    plt.legend()
    _finish(cic_path)


def risk_stratification(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    x = np.arange(len(frame))
    plt.bar(x - 0.18, frame.observed_rate, 0.36, label="Observed")
    plt.bar(x + 0.18, frame.mean_predicted_risk, 0.36, label="Predicted")
    plt.xticks(x, frame.risk_group)
    plt.ylabel("30-day readmission rate")
    plt.title("Risk tertile stratification")
    plt.legend()
    _finish(path)

