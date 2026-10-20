from __future__ import annotations

import argparse
from pathlib import Path

from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.data.validate_data import require_valid_creditcard_dataset
from fraud_agent.models.train import train_default_candidates
from fraud_agent.utils.config import ensure_directories, load_config
from fraud_agent.utils.reproducibility import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)

    df = load_creditcard_dataset(config["paths"]["raw_data"])
    require_valid_creditcard_dataset(df)
    metrics = train_default_candidates(
        df,
        random_seed=seed,
        model_output_dir=config["paths"]["models_dir"],
    )
    output_path = Path(config["paths"]["metrics_dir"]) / "validation_model_comparison.csv"
    metrics.to_csv(output_path, index=False)
    print(f"Wrote validation model comparison to {output_path}")


if __name__ == "__main__":
    main()

