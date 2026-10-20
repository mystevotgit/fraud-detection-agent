from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fraud_agent.agent.cost_model import CostConfig
from fraud_agent.agent.genetic_optimizer import GAConfig, GeneticThresholdOptimizer
from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.data.validate_data import require_valid_creditcard_dataset
from fraud_agent.models.evaluate import timed_predict_proba
from fraud_agent.models.train import stratified_train_valid_test_split
from fraud_agent.utils.config import ensure_directories, load_config
from fraud_agent.utils.reproducibility import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)

    metrics_path = Path(config["paths"]["metrics_dir"]) / "validation_model_comparison.csv"
    if args.model_name is None:
        if not metrics_path.exists():
            raise FileNotFoundError("Run scripts/train_models.py before threshold optimization.")
        comparison = pd.read_csv(metrics_path)
        model_name = str(comparison.sort_values(["pr_auc", "recall"], ascending=False).iloc[0]["model"])
    else:
        model_name = args.model_name

    model_path = Path(config["paths"]["models_dir"]) / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    df = load_creditcard_dataset(config["paths"]["raw_data"])
    require_valid_creditcard_dataset(df)
    split_cfg = config["data"]["stratified_split"]
    split = stratified_train_valid_test_split(
        df,
        train_size=float(split_cfg["train_size"]),
        validation_size=float(split_cfg["validation_size"]),
        test_size=float(split_cfg["test_size"]),
        random_seed=seed,
    )
    model = joblib.load(model_path)
    probabilities, _ = timed_predict_proba(model, split.X_valid)

    cost_cfg = CostConfig(**config["costs"])
    ga_cfg = GAConfig(random_seed=seed, **config["ga"])
    optimizer = GeneticThresholdOptimizer(ga_cfg, cost_cfg)
    result = optimizer.optimize(split.y_valid.to_numpy(), probabilities, split.X_valid["Amount"].to_numpy())

    payload = {
        "model_name": model_name,
        "review_threshold": result.thresholds.review,
        "block_threshold": result.thresholds.block,
        "validation_best_cost": result.best_cost,
        "history": result.history,
        "selection_note": "Thresholds optimized on validation predictions only; test set remains untouched.",
    }
    output_path = Path(config["paths"]["metrics_dir"]) / "ga_thresholds.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote GA thresholds to {output_path}")


if __name__ == "__main__":
    main()

