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


def _build_corpus(seed: int = 0) -> list[tuple[str, pd.DataFrame, np.ndarray, str]]:
    """Construct benchmark datasets with injected missing values for RL training."""
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
    except Exception as exc:
        print(f"skipping california_housing (fetch failed: {exc})", file=sys.stderr)

    return corpus


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL pipeline optimizer training CLI")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--fast-surrogate", action="store_true")
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Optuna trials per family for full-stack reward."
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
    """Run RL optimizer training loop over benchmark dataset corpus."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    corpus = _build_corpus(seed=args.seed)
    print(f"Corpus: {[name for name, *_ in corpus]}")

    if args.fast_surrogate:
        reward_fn = fast_surrogate_reward_fn(cv_folds=args.cv_folds, seed=args.seed)
        print("Reward: fast_surrogate_reward_fn (RandomForest)")
    else:
        config = OptunaSearchConfig(n_trials=args.n_trials, cv_folds=args.cv_folds, seed=args.seed)
        reward_fn = full_stack_reward_fn(
            config, stacking_cv_folds=args.stacking_cv_folds, seed=args.seed
        )
        print(f"Reward: full_stack_reward_fn (n_trials={args.n_trials}, cv_folds={args.cv_folds})")

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
