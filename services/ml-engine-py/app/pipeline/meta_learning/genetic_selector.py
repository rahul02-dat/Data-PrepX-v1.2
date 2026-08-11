from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.config import GeneticSelectorConfig

# fitness_fn(X, y, mask) -> validation score; higher is better. `mask` is a boolean array of
# shape (n_features,) selecting which columns of X to use. CLAUDE.md §5.2: "population of
# feature subsets, fitness = validation metric, standard GA operators."
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


# Draw a random boolean mask with at least min_features True entries
def _random_mask(n_features: int, min_features: int, rng: np.random.Generator) -> np.ndarray:
    if min_features > n_features:
        raise ValueError(
            f"min_features={min_features} exceeds the number of available features "
            f"({n_features})"
        )
    while True:
        mask = rng.random(n_features) < 0.5
        if mask.sum() >= min_features:
            return mask


# Tournament selection: sample tournament_size individuals, return the fittest
def _tournament_select(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = rng.integers(0, len(population), size=tournament_size)
    best_idx = idx[np.argmax(fitnesses[idx])]
    return population[int(best_idx)].copy()


# Uniform crossover: each gene independently taken from one parent or the other
def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    take_a = rng.random(len(parent_a)) < 0.5
    return np.where(take_a, parent_a, parent_b)


# Bit-flip mutation, rejected (parent returned unchanged) if it would violate min_features
def _mutate(
    individual: np.ndarray, mutation_rate: float, min_features: int, rng: np.random.Generator
) -> np.ndarray:
    flip = rng.random(len(individual)) < mutation_rate
    mutated = np.where(flip, ~individual, individual)
    if mutated.sum() < min_features:
        return individual.copy()
    return mutated


# Run a genetic algorithm over boolean feature-subset masks, evaluated by fitness_fn against
# (X, y). Standard operators: tournament selection, uniform crossover, bit-flip mutation, and
# elitism (the top elitism_count individuals survive each generation unchanged).
def run_genetic_selection(
    X: np.ndarray,
    y: np.ndarray,
    fitness_fn: FitnessFn,
    config: GeneticSelectorConfig | None = None,
    *,
    seed: int | None = None,
) -> GeneticSelectionResult:
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

    assert best_mask is not None  # population_size >= 2 guarantees at least one evaluation
    return GeneticSelectionResult(best_mask=best_mask, best_fitness=best_fitness, history=history)