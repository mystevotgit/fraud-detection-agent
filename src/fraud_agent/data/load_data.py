from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_creditcard_dataset(path: str | Path) -> pd.DataFrame:
    """Load the official Kaggle/ULB credit card fraud CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Place the Kaggle/ULB creditcard.csv file at "
            f"{csv_path}. See data/raw/README.md for setup instructions."
        )
    return pd.read_csv(csv_path)

