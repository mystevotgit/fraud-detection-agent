from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    column_count: int
    fraud_count: int
    legitimate_count: int
    fraud_rate: float
    duplicate_rows: int
    missing_values: int
    valid: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_creditcard_schema(df: pd.DataFrame) -> ValidationReport:
    """Validate expected schema and core data quality checks."""
    issues: list[str] = []

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    if extra_columns:
        issues.append(f"Unexpected columns present: {extra_columns}")

    if list(df.columns) != EXPECTED_COLUMNS and not missing_columns:
        issues.append("Columns are present but not in the expected dataset order.")

    missing_values = int(df.isna().sum().sum())
    if missing_values:
        issues.append(f"Dataset contains {missing_values} missing values.")

    duplicate_rows = int(df.duplicated().sum())

    target = df["Class"] if "Class" in df.columns else pd.Series(dtype=int)
    invalid_targets = sorted(set(target.dropna().unique()) - {0, 1})
    if invalid_targets:
        issues.append(f"Class contains values other than 0 and 1: {invalid_targets}")

    numeric_columns = [col for col in EXPECTED_COLUMNS if col in df.columns]
    non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        issues.append(f"Columns must be numeric: {non_numeric}")

    if "Amount" in df.columns and (df["Amount"] < 0).any():
        issues.append("Amount contains negative values.")
    if "Time" in df.columns and (df["Time"] < 0).any():
        issues.append("Time contains negative values.")

    fraud_count = int((target == 1).sum()) if "Class" in df.columns else 0
    legitimate_count = int((target == 0).sum()) if "Class" in df.columns else 0
    row_count = int(len(df))
    fraud_rate = float(fraud_count / row_count) if row_count else np.nan
    if row_count and fraud_count == 0:
        issues.append("Dataset contains no fraud examples.")
    if row_count and legitimate_count == 0:
        issues.append("Dataset contains no legitimate examples.")

    return ValidationReport(
        row_count=row_count,
        column_count=int(df.shape[1]),
        fraud_count=fraud_count,
        legitimate_count=legitimate_count,
        fraud_rate=fraud_rate,
        duplicate_rows=duplicate_rows,
        missing_values=missing_values,
        valid=not issues,
        issues=tuple(issues),
    )


def require_valid_creditcard_dataset(df: pd.DataFrame) -> ValidationReport:
    report = validate_creditcard_schema(df)
    if not report.valid:
        raise ValueError("; ".join(report.issues))
    return report

