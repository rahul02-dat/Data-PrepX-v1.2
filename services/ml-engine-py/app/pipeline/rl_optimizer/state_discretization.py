from __future__ import annotations

import numpy as np

from app.pipeline.rl_optimizer.meta_features import MetaFeatures

_FEATURE_BIN_EDGES: dict[str, tuple[float, ...]] = {
    "missingness_rate": (0.05, 0.2, 0.4),
    "mean_abs_skew": (0.5, 1.5, 3.0),
    "mean_cardinality_ratio": (0.05, 0.3, 0.7),
    "class_imbalance_ratio": (1.5, 3.0, 10.0),
    "drift_score": (0.1, 0.25, 0.5),
}

StateKey = tuple[int, ...]


def discretize_state(features: MetaFeatures) -> StateKey:
    """Discretize continuous MetaFeatures vector into discrete state bin tuple."""
    values = features.as_dict()
    return tuple(
        int(np.digitize(values[name], edges)) for name, edges in _FEATURE_BIN_EDGES.items()
    )


def n_possible_states() -> int:
    """Compute total number of discrete states across all feature bins."""
    total = 1
    for edges in _FEATURE_BIN_EDGES.values():
        total *= len(edges) + 1
    return total
