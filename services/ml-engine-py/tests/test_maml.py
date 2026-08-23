import numpy as np
import pytest
import torch

from app.pipeline.config import MAMLConfig
from app.pipeline.meta_learning.maml import MAMLLearner, Task


def _make_regression_tasks(n_tasks=40, n_support=20, n_query=20, input_dim=3, seed=0):
    """Generate synthetic regression task distribution."""
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_tasks):
        w = rng.normal(scale=1.5, size=input_dim)
        b = rng.normal(scale=0.5)

        X_support = rng.normal(size=(n_support, input_dim))
        y_support = X_support @ w + b + rng.normal(scale=0.05, size=n_support)

        X_query = rng.normal(size=(n_query, input_dim))
        y_query = X_query @ w + b + rng.normal(scale=0.05, size=n_query)

        tasks.append(Task(X_support, y_support, X_query, y_query))
    return tasks


def _make_sine_tasks(n_tasks, n_support, n_query, seed):
    """Generate synthetic sine wave few-shot regression tasks."""
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_tasks):
        amplitude = rng.uniform(0.1, 5.0)
        phase = rng.uniform(0, np.pi)
        X_support = rng.uniform(-5, 5, size=(n_support, 1))
        y_support = amplitude * np.sin(X_support[:, 0] - phase)
        X_query = rng.uniform(-5, 5, size=(n_query, 1))
        y_query = amplitude * np.sin(X_query[:, 0] - phase)
        tasks.append(Task(X_support, y_support, X_query, y_query))
    return tasks


def _mse(preds, y):
    return float(np.mean((preds - y) ** 2))


def test_init_params_shapes_linear_head():
    learner = MAMLLearner(input_dim=4, task_type="regression", config=MAMLConfig(hidden_dim=0))
    assert learner.meta_params["w1"].shape == (1, 4)
    assert learner.meta_params["b1"].shape == (1,)
    assert "w2" not in learner.meta_params


def test_init_params_shapes_shallow_mlp():
    learner = MAMLLearner(input_dim=4, task_type="regression", config=MAMLConfig(hidden_dim=8))
    assert learner.meta_params["w1"].shape == (8, 4)
    assert learner.meta_params["w2"].shape == (1, 8)


def test_adapt_reduces_loss_on_its_own_support_set():
    # A single, clean, single-task check: adapting on (X, y) should reduce the loss on that
    # same (X, y), independent of any meta-training.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true

    config = MAMLConfig(inner_lr=0.1, adapt_steps=20, seed=0)
    learner = MAMLLearner(input_dim=3, task_type="regression", config=config)

    preds_before = learner.predict(X)
    loss_before = _mse(preds_before, y)

    adapted = learner.adapt(X, y)
    preds_after = learner.predict(X, adapted)
    loss_after = _mse(preds_after, y)

    assert loss_after < loss_before


def test_meta_training_reduces_query_loss_history():
    tasks = _make_regression_tasks(seed=1)
    config = MAMLConfig(
        hidden_dim=0,
        inner_lr=0.05,
        outer_lr=0.01,
        inner_steps=3,
        n_outer_steps=60,
        meta_batch_size=4,
        seed=1,
    )
    learner = MAMLLearner(input_dim=3, task_type="regression", config=config)
    history = learner.meta_train(tasks)

    assert len(history) == 60
    # Mean of the second half should be lower than the mean of the first half: the outer loop
    # is actually reducing query loss over training, not flat or diverging.
    first_half = np.mean(history[:20])
    second_half = np.mean(history[-20:])
    assert second_half < first_half


def test_meta_trained_init_adapts_faster_than_random_init():
    # Sine-regression benchmark (see _make_sine_tasks docstring). Verified stable across
    # multiple seeds during development; this run uses a fixed seed for reproducibility.
    train_tasks = _make_sine_tasks(n_tasks=300, n_support=10, n_query=10, seed=1)
    held_out_tasks = _make_sine_tasks(n_tasks=30, n_support=10, n_query=10, seed=901)

    config = MAMLConfig(
        hidden_dim=40,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        n_outer_steps=700,
        adapt_steps=5,
        meta_batch_size=16,
        seed=1,
    )
    meta_learner = MAMLLearner(input_dim=1, task_type="regression", config=config)
    meta_learner.meta_train(train_tasks)

    random_learner = MAMLLearner(
        input_dim=1, task_type="regression", config=MAMLConfig(**{**config.__dict__, "seed": 7777})
    )

    meta_errors = []
    random_errors = []
    for task in held_out_tasks:
        meta_adapted = meta_learner.adapt(task.X_support, task.y_support)
        meta_errors.append(_mse(meta_learner.predict(task.X_query, meta_adapted), task.y_query))

        random_adapted = random_learner.adapt(task.X_support, task.y_support)
        random_errors.append(
            _mse(random_learner.predict(task.X_query, random_adapted), task.y_query)
        )

    # Meta-learned initialization generalizes better than random initialization
    assert np.mean(meta_errors) < np.mean(random_errors)


def test_classification_task_type_end_to_end():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    config = MAMLConfig(inner_lr=0.2, adapt_steps=15, seed=3)
    learner = MAMLLearner(input_dim=2, task_type="classification", config=config)

    adapted = learner.adapt(X, y)
    preds = learner.predict(X, adapted)
    accuracy = float((preds == y).mean())
    assert accuracy > 0.6


def test_adapt_does_not_mutate_meta_params():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([1.0, 2.0, -1.0])

    learner = MAMLLearner(input_dim=3, task_type="regression", config=MAMLConfig(seed=4))
    before = {k: v.clone() for k, v in learner.meta_params.items()}
    learner.adapt(X, y)
    for k, v in learner.meta_params.items():
        torch.testing.assert_close(v, before[k])


def test_meta_train_rejects_empty_task_list():
    learner = MAMLLearner(input_dim=3, task_type="regression")
    with pytest.raises(ValueError):
        learner.meta_train([])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": -1},
        {"inner_lr": 0.0},
        {"outer_lr": -0.1},
        {"inner_steps": 0},
        {"adapt_steps": 0},
        {"n_outer_steps": 0},
        {"meta_batch_size": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        MAMLConfig(**kwargs)
