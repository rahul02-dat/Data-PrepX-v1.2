import numpy as np
import pandas as pd
import pytest

from app.pipeline.config import DriftGateConfig, GeneticSelectorConfig, MAMLConfig
from app.pipeline.meta_learning.adaptive_loop import AdaptiveState, run_adaptive_step
from app.pipeline.meta_learning.maml import MAMLLearner


def _initial_state(reference_df: pd.DataFrame, n_features: int) -> AdaptiveState:
    return AdaptiveState(
        feature_mask=np.ones(n_features, dtype=bool),
        adapted_params={},
        reference_df=reference_df,
    )


def _fast_configs():
    return (
        DriftGateConfig(method="psi", psi_threshold=0.25),
        GeneticSelectorConfig(population_size=8, n_generations=3),
    )


def test_no_drift_reuses_existing_state_without_adaptation():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    current = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (current["a"] + current["b"] > 0).to_numpy().astype(int)

    learner = MAMLLearner(input_dim=2, task_type="classification", config=MAMLConfig(seed=0))
    state = _initial_state(reference, n_features=2)
    drift_config, ga_config = _fast_configs()

    result = run_adaptive_step(
        current, y, state, learner, drift_config=drift_config, genetic_config=ga_config, seed=0
    )

    assert result.drift_detected is False
    assert result.adapted_this_step is False
    np.testing.assert_array_equal(result.state.feature_mask, state.feature_mask)
    assert result.state.adapted_params == {}
    assert result.state.n_batches_seen == 1
    assert result.state.n_adaptations == 0
    # Reference distribution must not be replaced when nothing adapted.
    pd.testing.assert_frame_equal(result.state.reference_df, reference)


def test_drift_triggers_adaptation_and_updates_state():
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"a": rng.normal(loc=0, size=200), "b": rng.normal(loc=0, size=200)})
    current = pd.DataFrame({"a": rng.normal(loc=6, size=200), "b": rng.normal(loc=6, size=200)})
    y = (current["a"] + current["b"] > 12).to_numpy().astype(int)

    learner = MAMLLearner(input_dim=2, task_type="classification", config=MAMLConfig(seed=1))
    state = _initial_state(reference, n_features=2)
    drift_config, ga_config = _fast_configs()

    result = run_adaptive_step(
        current, y, state, learner, drift_config=drift_config, genetic_config=ga_config, seed=1
    )

    assert result.drift_detected is True
    assert result.adapted_this_step is True
    assert result.state.n_adaptations == 1
    assert result.state.n_batches_seen == 1
    assert result.state.adapted_params  # non-empty: MAML actually adapted
    assert "ga_best_fitness" in result.diagnostics
    # The new reference distribution becomes the batch that triggered adaptation.
    pd.testing.assert_frame_equal(result.state.reference_df, current)


def test_state_threads_through_multiple_steps():
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)})
    learner = MAMLLearner(input_dim=2, task_type="classification", config=MAMLConfig(seed=2))
    state = _initial_state(reference, n_features=2)
    drift_config, ga_config = _fast_configs()

    # Step 1: no drift.
    batch_1 = pd.DataFrame({"a": rng.normal(size=150), "b": rng.normal(size=150)})
    y_1 = (batch_1["a"] > 0).to_numpy().astype(int)
    result_1 = run_adaptive_step(
        batch_1, y_1, state, learner, drift_config=drift_config, genetic_config=ga_config, seed=2
    )
    assert result_1.adapted_this_step is False

    # Step 2: strong drift relative to the (still-unchanged) reference.
    batch_2 = pd.DataFrame({"a": rng.normal(loc=8, size=150), "b": rng.normal(loc=8, size=150)})
    y_2 = (batch_2["a"] > 8).to_numpy().astype(int)
    result_2 = run_adaptive_step(
        batch_2,
        y_2,
        result_1.state,
        learner,
        drift_config=drift_config,
        genetic_config=ga_config,
        seed=2,
    )
    assert result_2.adapted_this_step is True
    assert result_2.state.n_batches_seen == 2
    assert result_2.state.n_adaptations == 1


def test_rejects_mismatched_row_counts():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(size=50)})
    current = pd.DataFrame({"a": rng.normal(size=50)})
    y = np.zeros(10)  # wrong length

    learner = MAMLLearner(input_dim=1, task_type="classification", config=MAMLConfig(seed=3))
    state = _initial_state(reference, n_features=1)

    with pytest.raises(ValueError):
        run_adaptive_step(current, y, state, learner)


def test_rejects_batch_with_no_numeric_columns():
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"cat": ["x", "y", "z"]})
    y = np.array([0, 1, 0])

    learner = MAMLLearner(input_dim=1, task_type="classification", config=MAMLConfig(seed=4))
    state = _initial_state(reference, n_features=1)

    with pytest.raises(ValueError):
        run_adaptive_step(current, y, state, learner)
