from __future__ import annotations

import pytest

from fraud_agent.agent.decision_policy import Action, Thresholds, decide_action, decide_actions


def test_threshold_ordering_required():
    with pytest.raises(ValueError):
        Thresholds(review=0.8, block=0.2)


def test_boundary_decisions():
    thresholds = Thresholds(review=0.2, block=0.7)
    assert decide_action(0.199, thresholds) == Action.APPROVE
    assert decide_action(0.2, thresholds) == Action.MANUAL_REVIEW
    assert decide_action(0.699, thresholds) == Action.MANUAL_REVIEW
    assert decide_action(0.7, thresholds) == Action.BLOCK_ESCALATE


def test_vectorized_decisions():
    actions = decide_actions([0.1, 0.3, 0.9], Thresholds(review=0.2, block=0.7))
    assert actions.tolist() == ["APPROVE", "MANUAL REVIEW", "BLOCK/ESCALATE"]

