from __future__ import annotations

from fraud_agent.data.validate_data import validate_creditcard_schema


def test_valid_dataset_schema(tiny_creditcard_df):
    report = validate_creditcard_schema(tiny_creditcard_df)
    assert report.valid
    assert report.fraud_count == 2
    assert report.legitimate_count == 10


def test_missing_column_is_invalid(tiny_creditcard_df):
    report = validate_creditcard_schema(tiny_creditcard_df.drop(columns=["V28"]))
    assert not report.valid
    assert "Missing required columns" in report.issues[0]

