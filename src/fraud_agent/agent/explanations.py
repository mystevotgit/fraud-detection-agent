from __future__ import annotations

from dataclasses import dataclass

from fraud_agent.agent.decision_policy import Action, Thresholds, decide_action


@dataclass(frozen=True)
class DecisionExplanation:
    probability: float
    action: str
    risk_level: str
    threshold_reason: str
    note: str


def probability_risk_level(probability: float) -> str:
    if probability < 0.05:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.90:
        return "high"
    return "very high"


def explain_decision(probability: float, thresholds: Thresholds) -> DecisionExplanation:
    action = decide_action(probability, thresholds)
    if action == Action.APPROVE:
        reason = f"Probability is below the review threshold ({thresholds.review:.3f})."
    elif action == Action.MANUAL_REVIEW:
        reason = (
            f"Probability is between the review threshold ({thresholds.review:.3f}) "
            f"and block threshold ({thresholds.block:.3f})."
        )
    else:
        reason = f"Probability is at or above the block threshold ({thresholds.block:.3f})."
    return DecisionExplanation(
        probability=float(probability),
        action=action.value,
        risk_level=probability_risk_level(probability),
        threshold_reason=reason,
        note="PCA-anonymized features limit semantic interpretation; this is predictive, not causal.",
    )

