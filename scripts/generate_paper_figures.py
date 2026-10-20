"""Generate locked test-set figures and a policy cost comparison for the paper.

Run after scripts/run_all.py. Uses the saved model and GA thresholds only; no
re-tuning happens here, so the test set stays a pure final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from fraud_agent.agent.cost_model import CostConfig, total_operational_cost
from fraud_agent.agent.decision_policy import Action, Thresholds, decide_actions
from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.models.evaluate import timed_predict_proba
from fraud_agent.models.train import stratified_train_valid_test_split
from fraud_agent.utils.config import load_config

PLOT_DPI = 300


def _binary_actions(probabilities: np.ndarray, cutoff: float) -> np.ndarray:
    return np.where(probabilities >= cutoff, Action.BLOCK_ESCALATE.value, Action.APPROVE.value).astype(object)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    ga = json.loads((metrics_dir / "ga_thresholds.json").read_text(encoding="utf-8"))
    model = joblib.load(Path(config["paths"]["models_dir"]) / f"{ga['model_name']}.joblib")

    df = load_creditcard_dataset(config["paths"]["raw_data"])
    split_cfg = config["data"]["stratified_split"]
    split = stratified_train_valid_test_split(
        df,
        train_size=float(split_cfg["train_size"]),
        validation_size=float(split_cfg["validation_size"]),
        test_size=float(split_cfg["test_size"]),
        random_seed=int(config["project"]["random_seed"]),
    )

    cost_cfg = CostConfig(**config["costs"])
    ga_thresholds = Thresholds(float(ga["review_threshold"]), float(ga["block_threshold"]))
    band_cfg = config["agent"]["risk_bands"]
    band_thresholds = Thresholds(float(band_cfg["low"]), float(band_cfg["high"]))

    comparison: dict[str, dict[str, dict[str, float | int]]] = {}
    test_probs = None
    test_ga_actions = None
    for name, X, y in (("validation", split.X_valid, split.y_valid), ("test", split.X_test, split.y_test)):
        probs, _ = timed_predict_proba(model, X)
        labels = y.to_numpy()
        amounts = X["Amount"].to_numpy()
        policies = {
            "approve_all": np.full(labels.shape, Action.APPROVE.value, dtype=object),
            "default_0.50_binary": _binary_actions(probs, 0.50),
            "risk_bands_0.05_0.90": decide_actions(probs, band_thresholds),
            "ga_optimized": decide_actions(probs, ga_thresholds),
        }
        comparison[name] = {}
        for policy_name, actions in policies.items():
            comparison[name][policy_name] = {
                "cost": round(total_operational_cost(labels, actions, amounts, cost_cfg), 2),
                "approve": int((actions == Action.APPROVE.value).sum()),
                "review": int((actions == Action.MANUAL_REVIEW.value).sum()),
                "block": int((actions == Action.BLOCK_ESCALATE.value).sum()),
                "fraud_blocked_or_reviewed": int(((labels == 1) & (actions != Action.APPROVE.value)).sum()),
                "fraud_approved": int(((labels == 1) & (actions == Action.APPROVE.value)).sum()),
                "legit_blocked": int(((labels == 0) & (actions == Action.BLOCK_ESCALATE.value)).sum()),
            }
        if name == "test":
            test_probs = probs
            test_ga_actions = policies["ga_optimized"]

    (metrics_dir / "policy_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    y_test = split.y_test.to_numpy()

    # Confusion matrix at the default 0.50 cutoff (matches locked_test_metrics.json).
    cm = confusion_matrix(y_test, (test_probs >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(figures_dir / "paper_confusion_matrix_test.png", dpi=PLOT_DPI)
    plt.close(fig)

    # ROC and PR curves on the test set.
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC (AUC = {roc_auc_score(y_test, test_probs):.3f})")
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR (AP = {average_precision_score(y_test, test_probs):.3f})")
    fig.tight_layout()
    fig.savefig(figures_dir / "paper_roc_pr_test.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Action breakdown by true class under the GA policy (log scale: classes are very imbalanced).
    order = [Action.APPROVE.value, Action.MANUAL_REVIEW.value, Action.BLOCK_ESCALATE.value]
    legit_counts = [int(((y_test == 0) & (test_ga_actions == a)).sum()) for a in order]
    fraud_counts = [int(((y_test == 1) & (test_ga_actions == a)).sum()) for a in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(x - 0.2, legit_counts, width=0.4, label="Legitimate")
    ax.bar(x + 0.2, fraud_counts, width=0.4, label="Fraud")
    ax.set_yscale("log")
    ax.set_xticks(x, ["Approve", "Manual review", "Block/escalate"])
    ax.set_ylabel("Transactions (log scale)")
    for xi, v in zip(x - 0.2, legit_counts):
        ax.text(xi, max(v, 1) * 1.15, f"{v:,}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, fraud_counts):
        ax.text(xi, max(v, 1) * 1.15, f"{v:,}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "paper_policy_actions_test.png", dpi=PLOT_DPI)
    plt.close(fig)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
