from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from app.pipeline.config import MAMLConfig

TaskType = Literal["classification", "regression"]
ParamDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Task:
    """One meta-learning task: a historical data batch split into a support set (used for
    inner-loop adaptation) and a query set (used to score that adaptation for the outer-loop
    meta-update). CLAUDE.md §5.2: "outer loop meta-learns an initialization ... across a
    distribution of historical data-batch 'tasks'"."""

    X_support: np.ndarray
    y_support: np.ndarray
    X_query: np.ndarray
    y_query: np.ndarray


def _to_tensor(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr), dtype=dtype)


class MAMLLearner:
    """MAML over a small/linear or shallow-MLP head only (CLAUDE.md §5.2: "Scope the target
    model small ... MAML over gradient-boosted trees is not standard and is out of scope").

    Parameters are a plain dict of leaf tensors rather than an nn.Module, so the inner loop can
    be expressed functionally (forward(x, params)) and differentiated through with
    create_graph=True. That is what lets the outer loop's query loss backpropagate through the
    inner-loop adaptation steps themselves -- the actual "meta" part of MAML, as opposed to
    ordinary multi-task pretraining.
    """

    def __init__(
        self,
        input_dim: int,
        task_type: TaskType,
        config: MAMLConfig | None = None,
        *,
        output_dim: int | None = None,
    ):
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")

        self.config = config or MAMLConfig()
        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim or (1 if task_type == "regression" else 2)

        self._generator = torch.Generator().manual_seed(self.config.seed)
        self.meta_params: ParamDict = self._init_params()
        self._outer_optimizer = torch.optim.Adam(
            list(self.meta_params.values()), lr=self.config.outer_lr
        )

    # Initialize meta-parameters: a linear head (hidden_dim=0) or one ReLU hidden layer
    def _init_params(self) -> ParamDict:
        params: ParamDict = {}
        if self.config.hidden_dim > 0:
            params["w1"] = self._init_weight(self.config.hidden_dim, self.input_dim)
            params["b1"] = torch.zeros(self.config.hidden_dim, requires_grad=True)
            params["w2"] = self._init_weight(self.output_dim, self.config.hidden_dim)
            params["b2"] = torch.zeros(self.output_dim, requires_grad=True)
        else:
            params["w1"] = self._init_weight(self.output_dim, self.input_dim)
            params["b1"] = torch.zeros(self.output_dim, requires_grad=True)
        return params

    # Kaiming-uniform-initialized weight matrix, seeded via self._generator for determinism
    def _init_weight(self, out_features: int, in_features: int) -> torch.Tensor:
        w = torch.empty(out_features, in_features)
        bound = (1.0 / in_features) ** 0.5
        with torch.no_grad():
            w.uniform_(-bound, bound, generator=self._generator)
        w.requires_grad_(True)
        return w

    # Functional forward pass through an arbitrary parameter dict (not necessarily
    # self.meta_params) -- this is what lets the inner loop evaluate "what would the
    # currently-adapted model predict" without mutating the meta-parameters in place.
    def forward(self, x: torch.Tensor, params: ParamDict) -> torch.Tensor:
        if "w2" in params:
            hidden = F.relu(F.linear(x, params["w1"], params["b1"]))
            return F.linear(hidden, params["w2"], params["b2"])
        return F.linear(x, params["w1"], params["b1"])

    # Task-appropriate loss: cross-entropy for classification, MSE for regression
    def _loss(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return F.cross_entropy(preds, y.long())
        return F.mse_loss(preds.squeeze(-1), y.float())

    # Run n_steps of inner-loop gradient descent from `params`, returning the adapted dict.
    # create_graph=True keeps the adaptation differentiable w.r.t. the starting params (needed
    # by the outer loop); adapt() sets create_graph=False since nothing backprops through a
    # post-meta-training fast-adaptation call.
    def _inner_adapt(
        self,
        params: ParamDict,
        X: torch.Tensor,
        y: torch.Tensor,
        n_steps: int,
        *,
        create_graph: bool,
    ) -> ParamDict:
        adapted = dict(params)
        for _ in range(n_steps):
            preds = self.forward(X, adapted)
            loss = self._loss(preds, y)
            grads = torch.autograd.grad(loss, list(adapted.values()), create_graph=create_graph)
            adapted = {
                name: value - self.config.inner_lr * grad
                for (name, value), grad in zip(adapted.items(), grads, strict=True)
            }
        return adapted

    # Outer-loop meta-training over a distribution of historical tasks (CLAUDE.md §5.2). Each
    # outer step samples meta_batch_size tasks, inner-adapts the current meta_params to each
    # task's support set, scores that adaptation on the task's query set, and updates
    # meta_params to minimize the mean query loss -- this optimizes for fast adaptability, not
    # just average performance across tasks. Returns the per-step mean query loss for
    # diagnostics/convergence checks.
    def meta_train(self, tasks: list[Task]) -> list[float]:
        if not tasks:
            raise ValueError("meta_train requires at least one task")

        rng = np.random.default_rng(self.config.seed)
        query_loss_history: list[float] = []
        batch_size = min(self.config.meta_batch_size, len(tasks))

        for _ in range(self.config.n_outer_steps):
            batch_idx = rng.integers(0, len(tasks), size=batch_size)
            self._outer_optimizer.zero_grad()

            total_query_loss = torch.zeros(())
            for i in batch_idx:
                task = tasks[int(i)]
                X_s, y_s = _to_tensor(task.X_support), _to_tensor(task.y_support)
                X_q, y_q = _to_tensor(task.X_query), _to_tensor(task.y_query)

                adapted = self._inner_adapt(
                    self.meta_params, X_s, y_s, self.config.inner_steps, create_graph=True
                )
                query_preds = self.forward(X_q, adapted)
                total_query_loss = total_query_loss + self._loss(query_preds, y_q)

            mean_query_loss = total_query_loss / batch_size
            mean_query_loss.backward()
            self._outer_optimizer.step()
            query_loss_history.append(float(mean_query_loss.detach()))

        return query_loss_history

    # Fast-adapt the current meta-parameters to a brand-new data batch (CLAUDE.md §5.2: "inner
    # loop does a few gradient steps to adapt to each new batch instead of retraining from
    # scratch"). This is what adaptive_loop.py calls on drift-flagged batches. Detached from
    # self.meta_params first so this call never mutates the meta-parameters themselves.
    def adapt(self, X: np.ndarray, y: np.ndarray, *, n_steps: int | None = None) -> ParamDict:
        steps = n_steps if n_steps is not None else self.config.adapt_steps
        X_t, y_t = _to_tensor(X), _to_tensor(y)
        start_params = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in self.meta_params.items()
        }
        adapted = self._inner_adapt(start_params, X_t, y_t, steps, create_graph=False)
        return {name: value.detach() for name, value in adapted.items()}

    # Predict with a given (typically adapted) parameter dict, defaulting to the meta-params
    def predict(self, X: np.ndarray, params: ParamDict | None = None) -> np.ndarray:
        params = params if params is not None else self.meta_params
        X_t = _to_tensor(X)
        with torch.no_grad():
            preds = self.forward(X_t, params)
        if self.task_type == "classification":
            return preds.argmax(dim=-1).numpy()
        return preds.squeeze(-1).numpy()
