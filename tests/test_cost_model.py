from __future__ import annotations

from fraud_agent.agent.cost_model import CostConfig, review_capacity_penalty, total_operational_cost, transaction_costs


def test_amount_weighted_false_negative_cost():
    costs = transaction_costs(
        y_true=[1],
        actions=["APPROVE"],
        amounts=[100.0],
        config=CostConfig(false_negative_base=5.0, false_negative_amount_multiplier=2.0),
    )
    assert costs[0] == 205.0


def test_legitimate_block_and_review_cost():
    total = total_operational_cost(
        y_true=[0, 0],
        actions=["BLOCK/ESCALATE", "MANUAL REVIEW"],
        amounts=[20.0, 30.0],
        config=CostConfig(false_block_legitimate=25.0, manual_review=3.0, max_review_rate=None),
    )
    assert total == 28.0


def test_review_capacity_penalty():
    penalty = review_capacity_penalty(
        actions=["MANUAL REVIEW", "MANUAL REVIEW", "APPROVE", "APPROVE"],
        config=CostConfig(max_review_rate=0.25, review_capacity_penalty=100.0),
    )
    assert penalty == 25.0

