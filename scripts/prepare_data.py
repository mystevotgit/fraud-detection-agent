from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_agent.data.load_data import load_creditcard_dataset
from fraud_agent.data.validate_data import validate_creditcard_schema
from fraud_agent.utils.config import ensure_directories, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    df = load_creditcard_dataset(config["paths"]["raw_data"])
    report = validate_creditcard_schema(df)

    output_path = Path(config["paths"]["metrics_dir"]) / "data_validation_report.json"
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote validation report to {output_path}")
    if not report.valid:
        raise SystemExit("Dataset validation failed. See report for details.")


if __name__ == "__main__":
    main()

