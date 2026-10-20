from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_agent.agent.cost_model import CostConfig, total_operational_cost
from fraud_agent.agent.decision_policy import Thresholds, decide_actions


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 40
    generations: int = 60
    mutation_rate: float = 0.20
    mutation_scale: float = 0.08
    crossover_rate: float = 0.80
    tournament_size: int = 3
    elitism: int = 2
    min_gap: float = 0.001
    random_seed: int = 582


@dataclass(frozen=True)
class GAResult:
    thresholds: Thresholds
    best_cost: float
    history: list[dict[str, float]]


class GeneticThresholdOptimizer:
    """Genetic Algorithm for validation-only threshold optimization."""

    def __init__(self, ga_config: GAConfig, cost_config: CostConfig):
        self.ga_config = ga_config
        self.cost_config = cost_config
        self.rng = np.random.default_rng(ga_config.random_seed)

    def optimize(self, y_true, probabilities, amounts) -> GAResult:
        labels = np.asarray(y_true, dtype=int)
        probs = np.asarray(probabilities, dtype=float)
        amount_values = np.asarray(amounts, dtype=float)
        if not (labels.shape == probs.shape == amount_values.shape):
            raise ValueError("y_true, probabilities, and amounts must have the same shape.")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("Probabilities must be within [0, 1].")

        population = self._initialize_population()
        history: list[dict[str, float]] = []

        for generation in range(self.ga_config.generations):
            costs = np.array([self._fitness(chrom, labels, probs, amount_values) for chrom in population])
            best_idx = int(np.argmin(costs))
            history.append(
                {
                    "generation": float(generation),
                    "best_cost": float(costs[best_idx]),
                    "mean_cost": float(np.mean(costs)),
                    "review_threshold": float(population[best_idx, 0]),
                    "block_threshold": float(population[best_idx, 1]),
                }
            )
            population = self._next_generation(population, costs)

        costs = np.array([self._fitness(chrom, labels, probs, amount_values) for chrom in population])
        best_idx = int(np.argmin(costs))
        best = self._repair(population[best_idx])
        return GAResult(Thresholds(float(best[0]), float(best[1])), float(costs[best_idx]), history)

    def _initialize_population(self) -> np.ndarray:
        population = self.rng.uniform(0.0, 1.0, size=(self.ga_config.population_size, 2))
        return np.array([self._repair(chrom) for chrom in population])

    def _fitness(self, chromosome: np.ndarray, labels: np.ndarray, probs: np.ndarray, amounts: np.ndarray) -> float:
        repaired = self._repair(chromosome)
        actions = decide_actions(probs, Thresholds(float(repaired[0]), float(repaired[1])))
        return total_operational_cost(labels, actions, amounts, self.cost_config)

    def _next_generation(self, population: np.ndarray, costs: np.ndarray) -> np.ndarray:
        elite_count = min(self.ga_config.elitism, len(population))
        elite_indices = np.argsort(costs)[:elite_count]
        next_population = [population[idx].copy() for idx in elite_indices]

        while len(next_population) < self.ga_config.population_size:
            parent_a = self._tournament(population, costs)
            parent_b = self._tournament(population, costs)
            child_a, child_b = self._crossover(parent_a, parent_b)
            next_population.append(self._mutate(child_a))
            if len(next_population) < self.ga_config.population_size:
                next_population.append(self._mutate(child_b))

        return np.array(next_population)

    def _tournament(self, population: np.ndarray, costs: np.ndarray) -> np.ndarray:
        size = min(self.ga_config.tournament_size, len(population))
        indices = self.rng.choice(len(population), size=size, replace=False)
        return population[indices[int(np.argmin(costs[indices]))]].copy()

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.ga_config.crossover_rate:
            return parent_a.copy(), parent_b.copy()
        weight = self.rng.random()
        child_a = weight * parent_a + (1.0 - weight) * parent_b
        child_b = weight * parent_b + (1.0 - weight) * parent_a
        return child_a, child_b

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        child = chromosome.copy()
        mask = self.rng.random(2) < self.ga_config.mutation_rate
        child[mask] += self.rng.normal(0.0, self.ga_config.mutation_scale, size=int(mask.sum()))
        return self._repair(child)

    def _repair(self, chromosome: np.ndarray) -> np.ndarray:
        values = np.clip(np.sort(np.asarray(chromosome, dtype=float)), 0.0, 1.0)
        if values[1] - values[0] < self.ga_config.min_gap:
            center = float(np.mean(values))
            values[0] = max(0.0, center - self.ga_config.min_gap / 2)
            values[1] = min(1.0, values[0] + self.ga_config.min_gap)
            if values[1] > 1.0:
                values[1] = 1.0
                values[0] = 1.0 - self.ga_config.min_gap
        return values

