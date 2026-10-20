from __future__ import annotations

from fraud_agent.features.preprocessing import FEATURE_COLUMNS, build_preprocessor


def test_preprocessor_output_shape(tiny_creditcard_df):
    X = tiny_creditcard_df[FEATURE_COLUMNS]
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape == (len(X), 30)

