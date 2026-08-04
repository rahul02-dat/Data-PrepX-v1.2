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

Service ports (local): gateway-go `:8080`, ml-engine-py `:8000`,
agent-orchestrator `:8001`, frontend-react `:4173` (`:5173` under
`docker-compose.dev.yml`), postgres `:5432`, redis `:6379`, ollama `:11434`.

## Repository layout

```
services/
├── gateway-go/          Go: auth, job submit/poll, WebSocket status
├── ml-engine-py/        Python/FastAPI: pipeline core + Celery tasks
│   └── app/pipeline/    validation_gates.py, lineage.py, hashing.py, config.py
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