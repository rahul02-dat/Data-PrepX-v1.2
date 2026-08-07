# DataPrepX v2
<div align="justify">
Research-grade, autonomous data preparation and modeling platform for
tabular data. RL-selected preprocessing, meta-learned feature adaptation,
hard validation gates with immutable lineage, Bayesian-tuned stacked
ensembles, and a bounded (statistics-verified, never free-form) LLM
summarizer.
</div>

## Status

<div align="justify">
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
Trial-level lineage logging (`hyperparameters`/`metrics` tables) is not yet
wired up — `TrialRecord` is shaped for it but the DB write is a Phase 5/8
follow-up, needed before the RL agent's reward signal can read real trial
history.
</div>

## Local development

```bash
make setup     # installs local venvs / node_modules for host-side test runs
make up        # docker compose up --build: gateway-go, ml-engine-py,
               # agent-orchestrator, frontend-react, postgres, redis, ollama
make migrate   # applies Postgres migrations (via the migrate/migrate image)
make test      # Go + Python + frontend test suites (unit only; excludes db-marked tests)
make lint      # gofmt/vet, ruff/black, tsc
```

<div align="justify">
To exercise the lineage/gate code against a real Postgres instance (rather than
the mocked unit tests in `test_lineage.py`), after `make up` and `make migrate`:
</div>

```bash
cd services/ml-engine-py
DATABASE_URL=postgres://dataprepx:dataprepx@localhost:5432/dataprepx?sslmode=disable \
    .venv/bin/pytest -v -m db
```

<div align="justify">
To reproduce the Phase 3 imputation/outlier benchmark or the Phase 4
Optuna/stacking benchmark (both are excluded from the default `pytest` run
via the `research` marker, and are slow — the stacking one especially so
with production-scale `n_trials`):
</div>

```bash
cd services/ml-engine-py
python3 -m tests.research.benchmark_imputation_outliers
python3 -m tests.research.benchmark_hpo_stacking
```

<div align="justify">
Service ports (local): gateway-go `:8080`, ml-engine-py `:8000`,
agent-orchestrator `:8001`, frontend-react `:4173` (`:5173` under
`docker-compose.dev.yml`), postgres `:5432`, redis `:6379`, ollama `:11434`.
</div>

## Repository layout

```
services/
├── gateway-go/          Go: auth, job submit/poll, WebSocket status
├── ml-engine-py/        Python/FastAPI: pipeline core + Celery tasks
│   └── app/pipeline/    validation_gates.py, lineage.py, hashing.py, config.py,
│                        imputation.py, outliers.py, estimation/
│                        (optuna_search.py, stacking.py)
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

<div align="justify">
This is a from-scratch build. No module here is derived from or should be
reconciled against any prior codebase.
</div>
