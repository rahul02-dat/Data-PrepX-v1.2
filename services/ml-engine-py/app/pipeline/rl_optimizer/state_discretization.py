from __future__ import annotations

import numpy as np

from app.pipeline.rl_optimizer.meta_features import MetaFeatures

# Fixed bin edges per meta-feature, used to discretize the continuous state vector into a
# tuple of small integers for tabular Q-learning (CLAUDE.md §5.1: "tabular/linear-function-
# approximation Q-learning"). These are deliberately hand-chosen rather than fit from a corpus
# of observed states: at the start of training there is no such corpus yet, and re-fitting bin
# edges partway through training would silently invalidate every Q-value learned so far under
# the old binning. Revisit as an ADR if the fixed edges prove too coarse/fine in practice.
_FEATURE_BIN_EDGES: dict[str, tuple[float, ...]] = {
    "missingness_rate": (0.05, 0.2, 0.4),
    "mean_abs_skew": (0.5, 1.5, 3.0),
    "mean_cardinality_ratio": (0.05, 0.3, 0.7),
    "class_imbalance_ratio": (1.5, 3.0, 10.0),
    "drift_score": (0.1, 0.25, 0.5),
}

StateKey = tuple[int, ...]


# Discretize a continuous MetaFeatures vector into a hashable tuple of bin indices
def discretize_state(features: MetaFeatures) -> StateKey:
    values = features.as_dict()
    return tuple(
        int(np.digitize(values[name], edges)) for name, edges in _FEATURE_BIN_EDGES.items()
    )


# Number of distinct discrete states representable (product of per-feature bin counts) --
# useful for sizing exploration expectations, not required for the Q-table itself (which is a
# dict and only ever materializes states actually visited).
def n_possible_states() -> int:
    total = 1
    for edges in _FEATURE_BIN_EDGES.values():
        total *= len(edges) + 1
    return total
