from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Action(str, Enum):
    APPROVE = "APPROVE"
    MANUAL_REVIEW = "MANUAL REVIEW"
    BLOCK_ESCALATE = "BLOCK/ESCALATE"


@dataclass(frozen=True)
class Thresholds:
    review: float
    block: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.review < self.block <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= review < block <= 1.")


def decide_action(probability: float, thresholds: Thresholds) -> Action:
    """Map a calibrated fraud probability to an operational action."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be within [0, 1].")
    if probability < thresholds.review:
        return Action.APPROVE
    if probability < thresholds.block:
        return Action.MANUAL_REVIEW
    return Action.BLOCK_ESCALATE


def decide_actions(probabilities, thresholds: Thresholds) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("All probabilities must be within [0, 1].")
    actions = np.full(probs.shape, Action.MANUAL_REVIEW.value, dtype=object)
    actions[probs < thresholds.review] = Action.APPROVE.value
    actions[probs >= thresholds.block] = Action.BLOCK_ESCALATE.value
    return actions
