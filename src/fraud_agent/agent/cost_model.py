from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_agent.agent.decision_policy import Action


@dataclass(frozen=True)
class CostConfig:
    false_negative_base: float = 5.0
    false_negative_amount_multiplier: float = 1.0
    false_block_legitimate: float = 25.0
    manual_review: float = 3.0
    correct_block_fraud_benefit: float = 0.0
    max_review_rate: float | None = 0.10
    review_capacity_penalty: float = 10000.0


def transaction_costs(y_true, actions, amounts, config: CostConfig) -> np.ndarray:
    """Compute per-transaction operational costs under experimental assumptions."""
    labels = np.asarray(y_true, dtype=int)
    action_values = np.asarray(actions, dtype=object)
    amount_values = np.asarray(amounts, dtype=float)
    if not (labels.shape == action_values.shape == amount_values.shape):
        raise ValueError("y_true, actions, and amounts must have the same shape.")
    if np.any(amount_values < 0):
        raise ValueError("Amounts must be non-negative.")

    costs = np.zeros(labels.shape, dtype=float)
    fraud = labels == 1
    legitimate = labels == 0
    approved = action_values == Action.APPROVE.value
    reviewed = action_values == Action.MANUAL_REVIEW.value
    blocked = action_values == Action.BLOCK_ESCALATE.value

    costs[fraud & approved] += (
        config.false_negative_base
        + config.false_negative_amount_multiplier * amount_values[fraud & approved]
    )
    costs[legitimate & blocked] += config.false_block_legitimate
    costs[reviewed] += config.manual_review
    costs[fraud & blocked] -= config.correct_block_fraud_benefit
    return costs


def review_rate(actions) -> float:
    action_values = np.asarray(actions, dtype=object)
    if action_values.size == 0:
        return 0.0
    return float(np.mean(action_values == Action.MANUAL_REVIEW.value))


def review_capacity_penalty(actions, config: CostConfig) -> float:
    if config.max_review_rate is None:
        return 0.0
    overage = max(0.0, review_rate(actions) - config.max_review_rate)
    return float(overage * config.review_capacity_penalty)


def total_operational_cost(y_true, actions, amounts, config: CostConfig) -> float:
    return float(transaction_costs(y_true, actions, amounts, config).sum() + review_capacity_penalty(actions, config))

