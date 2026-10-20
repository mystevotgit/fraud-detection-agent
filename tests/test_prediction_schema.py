from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from fraud_agent.features.preprocessing import FEATURE_COLUMNS, build_scaled_pipeline


def test_end_to_end_prediction_on_sample_record(tiny_creditcard_df):
    X = tiny_creditcard_df[FEATURE_COLUMNS]
    y = tiny_creditcard_df["Class"]
    model = build_scaled_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(X, y)
    probability = model.predict_proba(X.iloc[[0]])[0, 1]
    assert 0.0 <= probability <= 1.0

