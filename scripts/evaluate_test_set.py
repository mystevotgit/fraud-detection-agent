from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from fraud_agent.agent.cost_model import CostConfig, total_operational_cost
from fraud_agent.agent.decision_policy import Thresholds, decide_actions
from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.data.validate_data import require_valid_creditcard_dataset
from fraud_agent.models.evaluate import binary_classification_metrics, timed_predict_proba
from fraud_agent.models.train import stratified_train_valid_test_split
from fraud_agent.utils.config import ensure_directories, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    thresholds_path = Path(config["paths"]["metrics_dir"]) / "ga_thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError("Run scripts/optimize_thresholds.py before locked test evaluation.")
    threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    model_path = Path(config["paths"]["models_dir"]) / f"{threshold_payload['model_name']}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    df = load_creditcard_dataset(config["paths"]["raw_data"])
    require_valid_creditcard_dataset(df)
    seed = int(config["project"]["random_seed"])
    split_cfg = config["data"]["stratified_split"]
    split = stratified_train_valid_test_split(
        df,
        train_size=float(split_cfg["train_size"]),
        validation_size=float(split_cfg["validation_size"]),
        test_size=float(split_cfg["test_size"]),
        random_seed=seed,
    )

    model = joblib.load(model_path)
    probabilities, predict_seconds = timed_predict_proba(model, split.X_test)
    thresholds = Thresholds(float(threshold_payload["review_threshold"]), float(threshold_payload["block_threshold"]))
    actions = decide_actions(probabilities, thresholds)
    cost = total_operational_cost(split.y_test.to_numpy(), actions, split.X_test["Amount"].to_numpy(), CostConfig(**config["costs"]))
    metrics = binary_classification_metrics(split.y_test, probabilities).to_dict()
    metrics.update(
        {
            "model_name": threshold_payload["model_name"],
            "ga_review_threshold": thresholds.review,
            "ga_block_threshold": thresholds.block,
            "ga_policy_test_cost": cost,
            "test_predict_seconds": predict_seconds,
            "review_count": int((actions == "MANUAL REVIEW").sum()),
            "block_count": int((actions == "BLOCK/ESCALATE").sum()),
            "approve_count": int((actions == "APPROVE").sum()),
            "note": "Locked test evaluation after model and thresholds were selected.",
        }
    )
    output_path = Path(config["paths"]["metrics_dir"]) / "locked_test_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote locked test metrics to {output_path}")


if __name__ == "__main__":
    main()

