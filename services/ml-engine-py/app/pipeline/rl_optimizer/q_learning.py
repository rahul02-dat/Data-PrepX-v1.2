from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.pipeline.rl_optimizer.environment import Action
from app.pipeline.rl_optimizer.state_discretization import StateKey


@dataclass(frozen=True)
class EpisodeRecord:
    """One training episode, shaped to map directly onto the `rl_episodes` lineage table
    (CLAUDE.md §5.1: "every episode (state, action, reward, resulting run_id) is persisted")."""

    episode_number: int
    state: StateKey
    action_index: int
    reward: float
    epsilon_used: float


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration (CLAUDE.md §5.1):
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)].

    Our environment (PreprocessingEnv) is single-step per episode, so next_state is always
    None in this project's actual usage -- update() still accepts a next_state for
    correctness/generality (and testability against the textbook multi-step formula), but the
    terminal (next_state=None) path is what Phase 5 exercises end to end.
    """

    def __init__(
        self,
        actions: list[Action],
        *,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        seed: int | None = None,
    ):
        if not actions:
            raise ValueError("actions must be non-empty")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self._rng = np.random.default_rng(seed)

        self.q_table: dict[StateKey, dict[int, float]] = defaultdict(
            lambda: dict.fromkeys(range(len(actions)), 0.0)
        )
        self.episodes: list[EpisodeRecord] = []

    # Choose an action index via epsilon-greedy; ties broken uniformly at random, not by index
    def select_action(self, state: StateKey) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(self.actions)))

        q_values = self.q_table[state]
        best_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == best_value]
        return int(self._rng.choice(best_actions))

    # Apply the Q-learning update rule for one observed transition
    def update(
        self,
        state: StateKey,
        action_index: int,
        reward: float,
        next_state: StateKey | None = None,
    ) -> float:
        current_q = self.q_table[state][action_index]
        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state].values())

        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action_index] = new_q
        return new_q

    # Run one full single-step episode: select an action, apply it via step_fn, update Q,
    # record the episode, and decay epsilon. step_fn takes (state, action) -> (reward, info).
    def run_episode(self, episode_number: int, state: StateKey, step_fn) -> EpisodeRecord:
        action_index = self.select_action(state)
        action = self.actions[action_index]
        reward, _info = step_fn(state, action)

        self.update(state, action_index, reward, next_state=None)

        record = EpisodeRecord(
            episode_number=episode_number,
            state=state,
            action_index=action_index,
            reward=reward,
            epsilon_used=self.epsilon,
        )
        self.episodes.append(record)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return record

    # Greedy (no exploration) action for a state -- used to read out the learned policy
    def best_action(self, state: StateKey) -> Action:
        q_values = self.q_table[state]
        best_index = max(q_values, key=q_values.get)
        return self.actions[best_index]
