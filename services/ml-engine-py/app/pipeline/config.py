from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "gates.yaml"
_VALID_TREE_FAMILIES = {"xgboost", "lightgbm", "random_forest"}


@dataclass(frozen=True)
class MaxNullRateGateConfig:
    enabled: bool = True
    threshold: float = 0.4
    per_column: bool = True


@dataclass(frozen=True)
class SchemaConformanceGateConfig:
    enabled: bool = True
    allow_extra_columns: bool = False
    strict_dtypes: bool = True


@dataclass(frozen=True)
class DriftGateConfig:
    enabled: bool = True
    method: str = "psi"
    psi_threshold: float = 0.25
    ks_p_value_threshold: float = 0.05
    psi_bins: int = 10

    def __post_init__(self) -> None:
        if self.method not in ("psi", "ks"):
            raise ValueError(f"drift_gate.method must be 'psi' or 'ks', got {self.method!r}")


@dataclass(frozen=True)
class ImputationConfig:
    enabled: bool = True
    method: str = "mice"
    mice_max_iter: int = 25
    mice_tol: float = 1e-2
    knn_n_neighbors: int = 5
    categorical_strategy: str = "most_frequent"

    def __post_init__(self) -> None:
        if self.method not in ("mice", "knn"):
            raise ValueError(f"imputation.method must be 'mice' or 'knn', got {self.method!r}")
        if self.categorical_strategy not in ("most_frequent",):
            raise ValueError(
                "imputation.categorical_strategy must be 'most_frequent', "
                f"got {self.categorical_strategy!r}"
            )


@dataclass(frozen=True)
class OutlierDetectionConfig:
    enabled: bool = True
    method: str = "isolation_forest"
    contamination: float = 0.05
    n_estimators: int = 100
    lof_n_neighbors: int = 20

    def __post_init__(self) -> None:
        if self.method not in ("isolation_forest", "lof", "none"):
            raise ValueError(
                "outlier_detection.method must be 'isolation_forest', 'lof', or 'none', "
                f"got {self.method!r}"
            )
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(
                f"outlier_detection.contamination must be in (0, 0.5), got {self.contamination}"
            )


@dataclass(frozen=True)
class EstimationConfig:
    enabled: bool = True
    n_trials: int = 30
    cv_folds: int = 5
    n_startup_trials: int = 5
    n_warmup_steps: int = 5
    tree_model_families: tuple[str, ...] = ("xgboost", "lightgbm", "random_forest")
    include_linear_family: bool = True
    stacking_cv_folds: int = 5

    def __post_init__(self) -> None:
        for fam in self.tree_model_families:
            if fam not in _VALID_TREE_FAMILIES:
                raise ValueError(
                    f"estimation.tree_model_families entries must be one of "
                    f"{sorted(_VALID_TREE_FAMILIES)}, got {fam!r}"
                )
        if self.n_trials < 1:
            raise ValueError(f"estimation.n_trials must be >= 1, got {self.n_trials}")
        if self.cv_folds < 2:
            raise ValueError(f"estimation.cv_folds must be >= 2, got {self.cv_folds}")

    def all_families(self) -> tuple[str, ...]:
        """Return full list of configured model families including linear model if enabled."""
        if self.include_linear_family:
            return (*self.tree_model_families, "linear")
        return self.tree_model_families


@dataclass(frozen=True)
class RLOptimizerConfig:
    enabled: bool = True
    n_episodes: int = 50
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.3
    epsilon_decay: float = 0.98
    min_epsilon: float = 0.01
    reward_mode: str = "full_stack"

    def __post_init__(self) -> None:
        if self.reward_mode not in ("full_stack", "fast_surrogate"):
            raise ValueError(
                "rl_optimizer.reward_mode must be 'full_stack' or 'fast_surrogate', "
                f"got {self.reward_mode!r}"
            )
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"rl_optimizer.alpha must be in (0, 1], got {self.alpha}")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"rl_optimizer.epsilon must be in [0, 1], got {self.epsilon}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"rl_optimizer.gamma must be in [0, 1], got {self.gamma}")


@dataclass(frozen=True)
class GeneticSelectorConfig:
    enabled: bool = True
    population_size: int = 30
    n_generations: int = 20
    crossover_rate: float = 0.7
    mutation_rate: float = 0.05
    tournament_size: int = 3
    elitism_count: int = 2
    min_features: int = 1

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError(
                f"genetic_selector.population_size must be >= 2, got {self.population_size}"
            )
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError(
                f"genetic_selector.crossover_rate must be in [0, 1], got {self.crossover_rate}"
            )
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError(
                f"genetic_selector.mutation_rate must be in [0, 1], got {self.mutation_rate}"
            )
        if self.tournament_size < 1:
            raise ValueError(
                f"genetic_selector.tournament_size must be >= 1, got {self.tournament_size}"
            )
        if self.elitism_count < 0 or self.elitism_count >= self.population_size:
            raise ValueError(
                "genetic_selector.elitism_count must be in [0, population_size), "
                f"got {self.elitism_count}"
            )
        if self.min_features < 1:
            raise ValueError(f"genetic_selector.min_features must be >= 1, got {self.min_features}")
        if self.n_generations < 1:
            raise ValueError(
                f"genetic_selector.n_generations must be >= 1, got {self.n_generations}"
            )


@dataclass(frozen=True)
class MAMLConfig:
    enabled: bool = True
    hidden_dim: int = 0
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    inner_steps: int = 5
    n_outer_steps: int = 100
    meta_batch_size: int = 4
    adapt_steps: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_dim < 0:
            raise ValueError(f"maml.hidden_dim must be >= 0, got {self.hidden_dim}")
        if self.inner_lr <= 0:
            raise ValueError(f"maml.inner_lr must be > 0, got {self.inner_lr}")
        if self.outer_lr <= 0:
            raise ValueError(f"maml.outer_lr must be > 0, got {self.outer_lr}")
        if self.inner_steps < 1:
            raise ValueError(f"maml.inner_steps must be >= 1, got {self.inner_steps}")
        if self.adapt_steps < 1:
            raise ValueError(f"maml.adapt_steps must be >= 1, got {self.adapt_steps}")
        if self.n_outer_steps < 1:
            raise ValueError(f"maml.n_outer_steps must be >= 1, got {self.n_outer_steps}")
        if self.meta_batch_size < 1:
            raise ValueError(f"maml.meta_batch_size must be >= 1, got {self.meta_batch_size}")


@dataclass(frozen=True)
class MetaLearningConfig:
    genetic_selector: GeneticSelectorConfig = field(default_factory=GeneticSelectorConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)


@dataclass(frozen=True)
class PipelineConfig:
    max_null_rate_gate: MaxNullRateGateConfig = field(default_factory=MaxNullRateGateConfig)
    schema_conformance_gate: SchemaConformanceGateConfig = field(
        default_factory=SchemaConformanceGateConfig
    )
    drift_gate: DriftGateConfig = field(default_factory=DriftGateConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    outlier_detection: OutlierDetectionConfig = field(default_factory=OutlierDetectionConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    rl_optimizer: RLOptimizerConfig = field(default_factory=RLOptimizerConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)

    def as_dict(self) -> dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        return asdict(self)


def config_path() -> Path:
    """Resolve path to pipeline configuration file."""
    override = os.environ.get("GATE_CONFIG_PATH")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    """Load and validate pipeline configuration YAML."""
    resolved = path or config_path()
    if not resolved.exists():
        raise FileNotFoundError(
            "Gate config not found. Set GATE_CONFIG_PATH or restore " "config/gates.yaml."
        )
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    estimation_raw = dict(raw.get("estimation", {}))
    if "tree_model_families" in estimation_raw:
        estimation_raw["tree_model_families"] = tuple(estimation_raw["tree_model_families"])

    return PipelineConfig(
        max_null_rate_gate=MaxNullRateGateConfig(**raw.get("max_null_rate_gate", {})),
        schema_conformance_gate=SchemaConformanceGateConfig(
            **raw.get("schema_conformance_gate", {})
        ),
        drift_gate=DriftGateConfig(**raw.get("drift_gate", {})),
        imputation=ImputationConfig(**raw.get("imputation", {})),
        outlier_detection=OutlierDetectionConfig(**raw.get("outlier_detection", {})),
        estimation=EstimationConfig(**estimation_raw),
        rl_optimizer=RLOptimizerConfig(**raw.get("rl_optimizer", {})),
        meta_learning=MetaLearningConfig(
            genetic_selector=GeneticSelectorConfig(
                **raw.get("meta_learning", {}).get("genetic_selector", {})
            ),
            maml=MAMLConfig(**raw.get("meta_learning", {}).get("maml", {})),
        ),
    )
