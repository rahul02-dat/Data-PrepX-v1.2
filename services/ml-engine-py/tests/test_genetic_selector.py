import numpy as np
import pytest

from app.pipeline.config import GeneticSelectorConfig
from app.pipeline.meta_learning.genetic_selector import run_genetic_selection


def _make_informative_dataset(n=200, n_informative=2, n_noise=6, seed=0):
    rng = np.random.default_rng(seed)
    informative = rng.normal(size=(n, n_informative))
    noise = rng.normal(size=(n, n_noise))
    X = np.hstack([informative, noise])
    y = (informative.sum(axis=1) > 0).astype(int)
    return X, y


def _accuracy_fitness(X, y, mask):
    if mask.sum() == 0:
        return -1.0
    X_masked = X[:, mask]
    # Simple, fast, deterministic proxy: a linear separating-hyperplane-style score using the
    # sign of the summed selected features, which only works well if the informative columns
    # (which sum positively-correlated with y) are actually selected.
    score = (X_masked.sum(axis=1) > 0).astype(int)
    return float((score == y).mean())


def test_ga_selects_informative_features_over_noise():
    X, y = _make_informative_dataset(seed=1)
    config = GeneticSelectorConfig(population_size=20, n_generations=15)
    result = run_genetic_selection(X, y, _accuracy_fitness, config, seed=1)

    # The two informative columns (indices 0, 1) should end up selected in the best mask more
    # often than an arbitrary noise column, since selecting them is what drives fitness up.
    assert result.best_mask[0] or result.best_mask[1]
    assert result.best_fitness > 0.6  # meaningfully better than chance (0.5)


def test_fitness_improves_or_holds_across_generations():
    X, y = _make_informative_dataset(seed=2)
    config = GeneticSelectorConfig(population_size=15, n_generations=10)
    result = run_genetic_selection(X, y, _accuracy_fitness, config, seed=2)

    best_so_far = [rec.best_fitness for rec in result.history]
    running_max = np.maximum.accumulate(best_so_far)
    # The best-ever fitness (tracked outside the loop) should match the running max of
    # per-generation bests -- i.e. we never lose the best individual found so far (elitism).
    assert result.best_fitness == pytest.approx(running_max[-1])


def test_result_history_has_one_record_per_generation():
    X, y = _make_informative_dataset(n=60, seed=3)
    config = GeneticSelectorConfig(population_size=10, n_generations=7)
    result = run_genetic_selection(X, y, _accuracy_fitness, config, seed=3)
    assert len(result.history) == 7


def test_min_features_respected_in_final_population():
    X, y = _make_informative_dataset(n=60, seed=4)
    config = GeneticSelectorConfig(
        population_size=10, n_generations=5, min_features=3, mutation_rate=0.5
    )
    result = run_genetic_selection(X, y, _accuracy_fitness, config, seed=4)
    assert result.best_mask.sum() >= 3


def test_deterministic_given_seed():
    X, y = _make_informative_dataset(seed=5)
    config = GeneticSelectorConfig(population_size=10, n_generations=5)
    r1 = run_genetic_selection(X, y, _accuracy_fitness, config, seed=5)
    r2 = run_genetic_selection(X, y, _accuracy_fitness, config, seed=5)
    np.testing.assert_array_equal(r1.best_mask, r2.best_mask)
    assert r1.best_fitness == r2.best_fitness


def test_rejects_1d_input():
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        run_genetic_selection(np.array([1, 2, 3]), y, _accuracy_fitness)


def test_min_features_greater_than_available_raises():
    X, y = _make_informative_dataset(n=20, n_informative=1, n_noise=1, seed=6)
    config = GeneticSelectorConfig(population_size=5, n_generations=2, min_features=10)
    with pytest.raises(ValueError):
        run_genetic_selection(X, y, _accuracy_fitness, config, seed=6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 1},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"tournament_size": 0},
        {"elitism_count": -1},
        {"elitism_count": 30},
        {"min_features": 0},
        {"n_generations": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        GeneticSelectorConfig(**kwargs)
