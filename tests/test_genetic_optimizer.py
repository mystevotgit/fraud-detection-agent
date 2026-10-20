from __future__ import annotations

import numpy as np

from fraud_agent.agent.cost_model import CostConfig
from fraud_agent.agent.genetic_optimizer import GAConfig, GeneticThresholdOptimizer


def test_ga_constraints_and_reproducibility():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.01, 0.15, 0.55, 0.95, 0.30, 0.80])
    amounts = np.array([10.0, 15.0, 100.0, 200.0, 20.0, 150.0])
    ga_config = GAConfig(population_size=12, generations=8, random_seed=7)
    cost_config = CostConfig(max_review_rate=0.50)

    result_a = GeneticThresholdOptimizer(ga_config, cost_config).optimize(y_true, probabilities, amounts)
    result_b = GeneticThresholdOptimizer(ga_config, cost_config).optimize(y_true, probabilities, amounts)

    assert 0.0 <= result_a.thresholds.review < result_a.thresholds.block <= 1.0
    assert result_a.thresholds == result_b.thresholds
    assert result_a.best_cost == result_b.best_cost
    assert len(result_a.history) == ga_config.generations

