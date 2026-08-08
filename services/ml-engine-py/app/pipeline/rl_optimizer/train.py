"""
Train the Phase 5 RL pipeline optimizer.

Usage:
    cd services/ml-engine-py
    python3 -m app.pipeline.rl_optimizer.train --n-episodes 200

By default this uses full_stack_reward_fn (per project decision -- see
docs/adr/0005-rl-reward-cost.md for the cost tradeoff this implies). Each episode samples one
dataset from the corpus, observes its meta-features as state, lets the agent pick a
preprocessing action, and scores it with a full Phase 4 Optuna+stacking run. This is expensive:
budget realistically for tens of minutes to hours per episode at production Optuna settings
(see the ADR for the actual math) -- start with --n-trials and --cv-folds well below production
defaults to get a training run to *finish* before scaling up.

Pass --fast-surrogate to use fast_surrogate_reward_fn instead (a single untuned RandomForest),
which is what makes it practical to test this script itself, or to do quick local RL-agent
iteration, in minutes rather than hours/days.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits, load_wine

from app.pipeline.estimation.optuna_search import OptunaSearchConfig
from app.pipeline.rl_optimizer.environment import PreprocessingEnv, build_action_space
from app.pipeline.rl_optimizer.meta_features import compute_meta_features
from app.pipeline.rl_optimizer.q_learning import QLearningAgent
from app.pipeline.rl_optimizer.reward_functions import (
    fast_surrogate_reward_fn,
    full_stack_reward_fn,
)
from app.pipeline.rl_optimizer.state_discretization import discretize_state


# Corpus of varied benchmark datasets (planner: "a corpus of varied benchmark datasets"),
# each returned with injected missingness so the imputation half of the action space actually
# matters. All are sklearn-bundled (no network access required).
def _build_corpus(seed: int = 0) -> list[tuple[str, pd.DataFrame, np.ndarray, str]]:
    rng = np.random.default_rng(seed)
    corpus = []

    digits = load_digits()
    X = pd.DataFrame(digits.data)
    X_missing = X.mask(rng.random(X.shape) < 0.1)
    corpus.append(("digits", X_missing, digits.target, "classification"))

    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    X_missing = X.mask(rng.random(X.shape) < 0.15)
    corpus.append(("wine", X_missing, wine.target, "classification"))

    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    X_missing = X.mask(rng.random(X.shape) < 0.1)
    corpus.append(("breast_cancer", X_missing, breast_cancer.target, "classification"))

    try:
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names).sample(
            n=500, random_state=seed
        )
        y = housing.target[X.index.to_numpy()]
        X = X.reset_index(drop=True)
        X_missing = X.mask(rng.random(X.shape) < 0.1)
        corpus.append(("california_housing_sample", X_missing, y, "regression"))
    except Exception as exc:  # network-dependent fetch; degrade gracefully without it
        print(f"skipping california_housing (fetch failed: {exc})", file=sys.stderr)

    return corpus


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--fast-surrogate", action="store_true")
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Optuna trials/family (full-stack reward only)."
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--stacking-cv-folds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--epsilon-decay", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    corpus = _build_corpus(seed=args.seed)
    print(f"Corpus: {[name for name, *_ in corpus]}")

    if args.fast_surrogate:
        reward_fn = fast_surrogate_reward_fn(cv_folds=args.cv_folds, seed=args.seed)
        print("Reward: fast_surrogate_reward_fn (RandomForest, single fast model)")
    else:
        config = OptunaSearchConfig(n_trials=args.n_trials, cv_folds=args.cv_folds, seed=args.seed)
        reward_fn = full_stack_reward_fn(
            config, stacking_cv_folds=args.stacking_cv_folds, seed=args.seed
        )
        print(
            f"Reward: full_stack_reward_fn (n_trials={args.n_trials}, cv_folds={args.cv_folds}, "
            f"stacking_cv_folds={args.stacking_cv_folds}) -- this is slow, see the ADR"
        )

    env = PreprocessingEnv(reward_fn, seed=args.seed)
    actions = build_action_space()
    agent = QLearningAgent(
        actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)

    for episode in range(args.n_episodes):
        name, X, y, task = corpus[rng.integers(len(corpus))]
        t0 = time.monotonic()

        state_features = env.reset(X, y, task)
        state_key = discretize_state(state_features)

        def step_fn(_state, action):
            result = env.step(action)
            return result.reward, result.info

        record = agent.run_episode(episode, state_key, step_fn)
        elapsed = time.monotonic() - t0

        action = actions[record.action_index]
        print(
            f"episode={episode:4d} dataset={name:24s} action={action} "
            f"reward={record.reward:+.4f} epsilon={record.epsilon_used:.3f} "
            f"elapsed={elapsed:.1f}s"
        )

    print(f"\nTrained on {len(agent.episodes)} episodes.")
    print(f"Distinct states visited: {len(agent.q_table)}")
    for name, X, y, task in corpus:
        state_features = compute_meta_features(X, target=y)
        state_key = discretize_state(state_features)
        best = agent.best_action(state_key)
        print(f"Learned best action for {name:24s} (state={state_key}): {best}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
