from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.config import GeneticSelectorConfig

FitnessFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_mask: np.ndarray


@dataclass(frozen=True)
class GeneticSelectionResult:
    best_mask: np.ndarray
    best_fitness: float
    history: list[GenerationRecord] = field(default_factory=list)


def _random_mask(n_features: int, min_features: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random boolean feature mask with at least min_features active."""
    if min_features > n_features:
        raise ValueError(
            f"min_features={min_features} exceeds available features ({n_features})"
        )
    while True:
        mask = rng.random(n_features) < 0.5
        if mask.sum() >= min_features:
            return mask


def _tournament_select(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select highest fitness individual from random tournament sample."""
    idx = rng.integers(0, len(population), size=tournament_size)
    best_idx = idx[np.argmax(fitnesses[idx])]
    return population[int(best_idx)].copy()


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perform uniform crossover between two parent masks."""
    take_a = rng.random(len(parent_a)) < 0.5
    return np.where(take_a, parent_a, parent_b)


def _mutate(
    individual: np.ndarray, mutation_rate: float, min_features: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply bit-flip mutation ensuring min_features constraint is preserved."""
    flip = rng.random(len(individual)) < mutation_rate
    mutated = np.where(flip, ~individual, individual)
    if mutated.sum() < min_features:
        return individual.copy()
    return mutated


def run_genetic_selection(
    X: np.ndarray,
    y: np.ndarray,
    fitness_fn: FitnessFn,
    config: GeneticSelectorConfig | None = None,
    *,
    seed: int | None = None,
) -> GeneticSelectionResult:
    """Execute genetic algorithm to find optimal feature subset mask."""
    config = config or GeneticSelectorConfig()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")

    n_features = X.shape[1]
    rng = np.random.default_rng(seed)

    population = [
        _random_mask(n_features, config.min_features, rng) for _ in range(config.population_size)
    ]
    history: list[GenerationRecord] = []
    best_mask: np.ndarray | None = None
    best_fitness = -np.inf

    for generation in range(config.n_generations):
        fitnesses = np.array([fitness_fn(X, y, mask) for mask in population], dtype=float)

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_mask = population[gen_best_idx].copy()

        history.append(
            GenerationRecord(
                generation=generation,
                best_fitness=float(fitnesses[gen_best_idx]),
                mean_fitness=float(fitnesses.mean()),
                best_mask=population[gen_best_idx].copy(),
            )
        )

        elite_idx = np.argsort(fitnesses)[::-1][: config.elitism_count]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < config.population_size:
            parent_a = _tournament_select(population, fitnesses, config.tournament_size, rng)
            if rng.random() < config.crossover_rate:
                parent_b = _tournament_select(population, fitnesses, config.tournament_size, rng)
                child = _crossover(parent_a, parent_b, rng)
            else:
                child = parent_a.copy()
            child = _mutate(child, config.mutation_rate, config.min_features, rng)
            new_population.append(child)

        population = new_population

    assert best_mask is not None
    return GeneticSelectionResult(best_mask=best_mask, best_fitness=best_fitness, history=history)
