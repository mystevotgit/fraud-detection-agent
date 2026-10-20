from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    commands = [
        [sys.executable, "scripts/prepare_data.py", "--config", args.config],
        [sys.executable, "scripts/train_models.py", "--config", args.config],
        [sys.executable, "scripts/optimize_thresholds.py", "--config", args.config],
        [sys.executable, "scripts/evaluate_test_set.py", "--config", args.config],
    ]
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

