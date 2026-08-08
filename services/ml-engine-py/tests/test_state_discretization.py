from app.pipeline.rl_optimizer.meta_features import MetaFeatures
from app.pipeline.rl_optimizer.state_discretization import discretize_state, n_possible_states


def _features(**overrides) -> MetaFeatures:
    base = dict(
        missingness_rate=0.0,
        mean_abs_skew=0.0,
        mean_cardinality_ratio=0.0,
        class_imbalance_ratio=1.0,
        drift_score=0.0,
    )
    base.update(overrides)
    return MetaFeatures(**base)


def test_discretize_state_returns_tuple_of_five_ints():
    key = discretize_state(_features())
    assert isinstance(key, tuple)
    assert len(key) == 5
    assert all(isinstance(v, int) for v in key)


def test_discretize_state_is_deterministic():
    features = _features(missingness_rate=0.3, mean_abs_skew=2.0)
    assert discretize_state(features) == discretize_state(features)


def test_discretize_state_low_values_map_to_bin_zero():
    key = discretize_state(_features())
    assert key == (0, 0, 0, 0, 0)


def test_discretize_state_high_values_map_to_highest_bin():
    key = discretize_state(
        _features(
            missingness_rate=0.9,
            mean_abs_skew=10.0,
            mean_cardinality_ratio=0.95,
            class_imbalance_ratio=50.0,
            drift_score=1.0,
        )
    )
    # 3 edges per feature -> bins 0..3, so the top bin is 3 for every feature here.
    assert key == (3, 3, 3, 3, 3)


def test_discretize_state_distinguishes_meaningfully_different_states():
    low = discretize_state(_features(missingness_rate=0.01))
    high = discretize_state(_features(missingness_rate=0.5))
    assert low != high


def test_n_possible_states_matches_bin_count_product():
    # 5 features, each with 3 edges -> 4 bins each -> 4^5 possible discrete states.
    assert n_possible_states() == 4**5
