from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from fraud_agent.features.preprocessing import build_scaled_pipeline


def test_preprocessing_is_inside_pipeline():
    pipeline = build_scaled_pipeline(LogisticRegression())
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    assert not hasattr(pipeline.named_steps["preprocess"], "transformers_")

