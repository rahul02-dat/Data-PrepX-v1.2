import pytest

from app.pipeline.rl_optimizer.environment import Action
from app.pipeline.rl_optimizer.q_learning import QLearningAgent

_ACTIONS = [
    Action("mice", "none", 0),
    Action("knn", "none", 0),
    Action("mice", "isolation_forest", 0),
]


def test_rejects_empty_action_list():
    with pytest.raises(ValueError):
        QLearningAgent([])


@pytest.mark.parametrize("alpha", [0.0, 1.5, -0.1])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        QLearningAgent(_ACTIONS, alpha=alpha)


@pytest.mark.parametrize("gamma", [-0.1, 1.1])
def test_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        QLearningAgent(_ACTIONS, gamma=gamma)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        QLearningAgent(_ACTIONS, epsilon=epsilon)


def test_new_state_has_zero_initialized_q_values():
    agent = QLearningAgent(_ACTIONS, seed=0)
    q_values = agent.q_table[(0, 0, 0, 0, 0)]
    assert q_values == {0: 0.0, 1: 0.0, 2: 0.0}


def test_select_action_always_explores_with_epsilon_one():
    agent = QLearningAgent(_ACTIONS, epsilon=1.0, seed=0)
    chosen = {agent.select_action((0, 0, 0, 0, 0)) for _ in range(200)}
    assert chosen == {0, 1, 2}


def test_select_action_always_exploits_with_epsilon_zero():
    agent = QLearningAgent(_ACTIONS, epsilon=0.0, seed=0)
    state = (0, 0, 0, 0, 0)
    agent.q_table[state][1] = 5.0  # make action 1 clearly best
    chosen = {agent.select_action(state) for _ in range(50)}
    assert chosen == {1}


def test_update_terminal_transition_moves_q_toward_reward():
    agent = QLearningAgent(_ACTIONS, alpha=0.5, seed=0)
    state = (0, 0, 0, 0, 0)
    new_q = agent.update(state, action_index=0, reward=1.0, next_state=None)
    # Q started at 0; alpha=0.5 -> Q = 0 + 0.5*(1.0 - 0) = 0.5
    assert new_q == pytest.approx(0.5)
    assert agent.q_table[state][0] == pytest.approx(0.5)


def test_update_non_terminal_transition_uses_bootstrapped_max_next_q():
    agent = QLearningAgent(_ACTIONS, alpha=1.0, gamma=0.9, seed=0)
    state_a = (0, 0, 0, 0, 0)
    state_b = (1, 0, 0, 0, 0)
    agent.q_table[state_b][2] = 10.0  # best next-state action

    new_q = agent.update(state_a, action_index=0, reward=1.0, next_state=state_b)
    # target = r + gamma * max_a' Q(s', a') = 1.0 + 0.9*10.0 = 10.0; alpha=1.0 -> Q = 10.0
    assert new_q == pytest.approx(10.0)


def test_update_converges_toward_true_reward_over_many_updates():
    agent = QLearningAgent(_ACTIONS, alpha=0.3, seed=0)
    state = (0, 0, 0, 0, 0)
    for _ in range(200):
        agent.update(state, action_index=0, reward=2.0, next_state=None)
    assert agent.q_table[state][0] == pytest.approx(2.0, abs=1e-3)


def test_best_action_returns_argmax_action():
    agent = QLearningAgent(_ACTIONS, seed=0)
    state = (0, 0, 0, 0, 0)
    agent.q_table[state][2] = 99.0
    assert agent.best_action(state) == _ACTIONS[2]


def test_run_episode_records_episode_and_decays_epsilon():
    agent = QLearningAgent(_ACTIONS, epsilon=0.5, epsilon_decay=0.9, seed=0)
    state = (0, 0, 0, 0, 0)

    def step_fn(s, a):
        return 1.0, {}

    record = agent.run_episode(0, state, step_fn)

    assert record.episode_number == 0
    assert record.state == state
    assert record.reward == 1.0
    assert len(agent.episodes) == 1
    assert agent.epsilon == pytest.approx(0.45)


def test_epsilon_never_decays_below_min_epsilon():
    agent = QLearningAgent(_ACTIONS, epsilon=0.1, epsilon_decay=0.1, min_epsilon=0.05, seed=0)
    state = (0, 0, 0, 0, 0)

    def step_fn(s, a):
        return 0.0, {}

    for i in range(10):
        agent.run_episode(i, state, step_fn)

    assert agent.epsilon >= 0.05


def test_agent_is_deterministic_given_seed():
    def step_fn(s, a):
        return float(a.threshold_bin), {}

    agent1 = QLearningAgent(_ACTIONS, epsilon=0.5, seed=42)
    agent2 = QLearningAgent(_ACTIONS, epsilon=0.5, seed=42)
    state = (0, 0, 0, 0, 0)

    records1 = [agent1.run_episode(i, state, step_fn) for i in range(20)]
    records2 = [agent2.run_episode(i, state, step_fn) for i in range(20)]

    assert [r.action_index for r in records1] == [r.action_index for r in records2]
