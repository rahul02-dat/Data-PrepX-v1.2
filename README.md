# DataPrepX v2

Research-grade, autonomous data preparation and modeling platform for
tabular data. RL-selected preprocessing, meta-learned feature adaptation,
hard validation gates with immutable lineage, Bayesian-tuned stacked
ensembles, and a bounded (statistics-verified, never free-form) LLM
summarizer.

## Status

Phase 0 (foundations & monorepo scaffold) — complete.
Phase 1 (data contracts & job model) — complete: shared JSON Schema
contract (`contracts/job.schema.json`), full Postgres lineage schema
(`infra/postgres/migrations/0002_schema.*.sql`), and gateway-go's job
submit/poll/WebSocket-stream implementation
(`services/gateway-go/internal/jobs`, `internal/ws`).
Phase 2 (validation gates & immutable lineage) — complete:
pluggable, fail-closed gates (`MaxNullRateGate`, `SchemaConformanceGate`,
`DriftGate`), content-addressed lineage recording and replay
(`services/ml-engine-py/app/pipeline/{validation_gates,lineage,hashing,config}.py`),
a deterministic `run_key` resolving the Phase 1/Phase 2 schema contradiction
around run identity (see `docs/adr/0003-run-id-determinism.md`), and a
user-supplied drift reference distribution (`docs/adr/0002-drift-reference-distribution.md`).
Gate thresholds are centrally configured in `config/gates.yaml`, never hardcoded.
Phase 3 (advanced imputation & outlier detection) — complete: MICE
(`IterativeImputer`) and KNN imputation with per-column-type routing and
convergence diagnostics, Isolation Forest and LOF outlier detection that
scores every row instead of dropping it
(`services/ml-engine-py/app/pipeline/{imputation,outliers}.py`). Benchmarked
against mean-imputation/IQR baselines in
`docs/research/imputation_outlier_benchmark.md` — MICE/KNN beat mean-imputation
cleanly, but Isolation Forest underperforms the IQR baseline on real,
high-dimensional data and LOF's advantage is inconsistent; see that write-up's
Interpretation and Limitations sections before citing either result. This
mixed finding is exactly the kind of thing the Phase 5 RL agent's reward
signal needs to be sensitive to, not smoothed over.
Phase 4 (Bayesian HPO & stacked ensembles) — complete: one Optuna study
per model family (XGBoost, LightGBM, RandomForest, ElasticNet/LogisticRegression),
TPE sampler with median pruning, cross-validated objective
(`services/ml-engine-py/app/pipeline/estimation/optuna_search.py`), and a
`StackingClassifier`/`StackingRegressor` combining the tuned base models
(`.../estimation/stacking.py`). New `estimation` block in `config/gates.yaml`
(`EstimationConfig` in `config.py`) versions `n_trials`, `cv_folds`, and the
model-family list into `config_hash`, consistent with the rest of the pipeline.
Benchmarked in `docs/research/optuna_stacking_benchmark.md`: the stack reliably
beats an untuned default-hyperparameter baseline, but on this benchmark pass it
did **not** consistently beat the single best *tuned* model — the tuned linear
model won on most (dataset, seed) pairs, because two of the three benchmark
datasets are synthetic with a near-linear signal. Read that write-up's
Limitations section before treating "stacking beats tuning" as established;
what's established so far is narrower ("stacking beats not tuning at all").
Also added: `app/pipeline/estimation/dataset_loading.py` (loads arbitrary CSVs,
auto-detects target column/task type; `load_dataset_with_auto_preprocess` for
quick exploration — imputes/encodes features only, never the target, drops
rather than fabricates rows with a missing target) and `run_on_dataset.py`, a
CLI to run the Phase 4 stack against any file.
Phase 4/5 lineage wiring — complete: `LineageRecorder` now has
`record_hyperparameter_trial`/`record_study_trials` and `record_metric`
(writing into the `hyperparameters`/`metrics` tables Phase 1 already defined),
plus `record_rl_episode` writing into a new `rl_episodes` table
(`infra/postgres/migrations/0005_rl_episodes.*.sql`). This closes the gap Phase
4 originally shipped with (trial results existed in code but were never
persisted to Postgres) and gives Phase 5 somewhere real to log episodes.
Phase 5 (RL pipeline optimizer) — environment, agent, and reward wiring
complete and unit-tested (70 tests across `meta_features.py`,
`state_discretization.py`, `environment.py`, `q_learning.py`); **the actual
training run and convergence-curve research artifact have not been produced.**
`PreprocessingEnv` is a single-step (contextual-bandit) environment over an
18-action space ({MICE,KNN} x {IsolationForest,LOF,none} x threshold bin);
`QLearningAgent` is tabular epsilon-greedy Q-learning. The reward is
`full_stack_reward_fn` — the entire Phase 4 stack, run fresh every episode —
by explicit, informed project-owner decision against the planner's own
recommendation to use a cheap surrogate during training
(see `docs/adr/0005-rl-reward-cost-and-environment-design.md`). Measured
(not estimated) cost at minimal settings was 15-32s/episode; at production
Optuna settings this is realistically 10-30+ minutes per episode, meaning a
100-episode training run is on the order of 20-40+ hours of compute. That run
has not been executed here — `train.py` was smoke-tested for correctness (2
real full-stack episodes, 15 fast-surrogate episodes) but the real training
run, its convergence curve, and the brute-force-grid-search comparison the
planner's Phase 5 acceptance criterion requires are still outstanding.
Phase 6 (meta-learning for adaptive feature engineering) — core modules
complete and unit-tested (39 tests across `genetic_selector.py`, `maml.py`,
`adaptive_loop.py`): a genetic-algorithm feature selector with standard
operators (tournament selection, uniform crossover, bit-flip mutation,
elitism), a PyTorch-based `MAMLLearner` scoped to a linear or one-hidden-layer
head per CLAUDE.md §5.2 (never gradient-boosted trees — see
`docs/adr/0006-maml-target-model-scope.md` for why PyTorch was pulled in and
how the scope is enforced in code, not just documented), and a standalone
`adaptive_loop.py` that only re-runs GA reselection + MAML adaptation when the
Phase 2 `DriftGate` flags drift, otherwise reusing the existing feature
set/model. The MAML "fast adaptation beats training from scratch" claim is
verified with the canonical sine-regression few-shot benchmark (Finn et al.,
2017), not an easy linear task — an earlier linear-task version of that test
was flaky because linear regression converges fast from any initialization,
which doesn't isolate MAML's actual benefit. `adaptive_loop.py` is
deliberately a plain synchronous function, not a Celery task, since Phase 8
(async execution) hasn't landed yet; a future Celery task can call
`run_adaptive_step` directly. **Not yet done:** the Phase 6 research artifact
(simulated concept-drift stream comparing MAML-adapted vs. static vs.
oracle-retrained-every-batch) required by the planner's acceptance criterion
— this is explicitly deferred, the same way Phase 5's training run was, and
should not be read as complete.

## Local development

```bash
make setup     # installs local venvs / node_modules for host-side test runs
make up        # docker compose up --build: gateway-go, ml-engine-py,
               # agent-orchestrator, frontend-react, postgres, redis, ollama
make migrate   # applies Postgres migrations (via the migrate/migrate image)
make test      # Go + Python + frontend test suites (unit only; excludes db-marked tests)
make lint      # gofmt/vet, ruff/black, tsc
```

To exercise the lineage/gate code against a real Postgres instance (rather than
the mocked unit tests in `test_lineage.py`), after `make up` and `make migrate`:

```bash
cd services/ml-engine-py
DATABASE_URL=postgres://dataprepx:dataprepx@localhost:5432/dataprepx?sslmode=disable \
    .venv/bin/pytest -v -m db
```

To reproduce the Phase 3 imputation/outlier benchmark or the Phase 4
Optuna/stacking benchmark (both are excluded from the default `pytest` run
via the `research` marker, and are slow — the stacking one especially so
with production-scale `n_trials`):

```bash
cd services/ml-engine-py
python3 -m tests.research.benchmark_imputation_outliers
python3 -m tests.research.benchmark_hpo_stacking
```

Service ports (local): gateway-go `:8080`, ml-engine-py `:8000`,
agent-orchestrator `:8001`, frontend-react `:4173` (`:5173` under
`docker-compose.dev.yml`), postgres `:5432`, redis `:6379`, ollama `:11434`.

## Repository layout

```
services/
├── gateway-go/          Go: auth, job submit/poll, WebSocket status
├── ml-engine-py/        Python/FastAPI: pipeline core + Celery tasks
│   └── app/pipeline/    validation_gates.py, lineage.py, hashing.py, config.py,
│                        imputation.py, outliers.py, estimation/
│                        (optuna_search.py, stacking.py, dataset_loading.py,
│                        run_on_dataset.py), rl_optimizer/
│                        (environment.py, q_learning.py, meta_features.py,
│                        state_discretization.py, reward_functions.py, train.py),
│                        meta_learning/ (genetic_selector.py, maml.py,
│                        adaptive_loop.py)
├── agent-orchestrator/  Python/LangGraph: bounded summarizer, Ollama client
└── frontend-react/      React/TS SPA
contracts/               Shared JSON Schema (job model) used across services
config/                  Central, versioned pipeline config (config/gates.yaml)
infra/                   Postgres migrations, Redis config
docs/
├── adr/                 Architecture decision records (one per non-obvious call)
└── research/            Benchmark write-ups, ablations
tests/
├── integration/         cross-service contract validation
└── research/            reproducibility + benchmark harness
```

This is a from-scratch build. No module here is derived from or should be
reconciled against any prior codebase.