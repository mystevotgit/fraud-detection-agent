from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_agent.features.preprocessing import FEATURE_COLUMNS
from fraud_agent.models.evaluate import binary_classification_metrics, timed_predict_proba
from fraud_agent.models.registry import build_supervised_models


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def stratified_train_valid_test_split(
    df: pd.DataFrame,
    train_size: float = 0.60,
    validation_size: float = 0.20,
    test_size: float = 0.20,
    random_seed: int = 582,
) -> SplitData:
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.")

    X = df[FEATURE_COLUMNS]
    y = df["Class"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=train_size,
        stratify=y,
        random_state=random_seed,
    )
    relative_valid = validation_size / (validation_size + test_size)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=relative_valid,
        stratify=y_temp,
        random_state=random_seed,
    )
    return SplitData(X_train, y_train, X_valid, y_valid, X_test, y_test)


def train_default_candidates(df: pd.DataFrame, random_seed: int = 582, model_output_dir: str | None = None) -> pd.DataFrame:
    """Train default candidate models and return validation metrics.

    This is intentionally modest; deeper hyperparameter search should be run only
    after proposal approval and with validation-only selection.
    """
    split = stratified_train_valid_test_split(df, random_seed=random_seed)
    rows: list[dict[str, object]] = []
    for name, model in build_supervised_models(random_seed=random_seed).items():
        start = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        train_seconds = time.perf_counter() - start
        probabilities, predict_seconds = timed_predict_proba(model, split.X_valid)
        metrics = binary_classification_metrics(split.y_valid, probabilities, threshold=0.5).to_dict()
        rows.append(
            {
                "model": name,
                "train_seconds": train_seconds,
                "validation_predict_seconds": predict_seconds,
                **metrics,
            }
        )
        if model_output_dir:
            joblib.dump(model, f"{model_output_dir}/{name}.joblib")
    return pd.DataFrame(rows)

