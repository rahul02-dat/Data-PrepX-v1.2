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

## Local development

```bash
make setup     # installs local venvs / node_modules for host-side test runs
make up        # docker compose up --build: gateway-go, ml-engine-py,
               # agent-orchestrator, frontend-react, postgres, redis, ollama
make migrate   # applies Postgres migrations (via the migrate/migrate image)
make test      # Go + Python + frontend test suites
make lint      # gofmt/vet, ruff/black, tsc
```

Service ports (local): gateway-go `:8080`, ml-engine-py `:8000`,
agent-orchestrator `:8001`, frontend-react `:4173` (`:5173` under
`docker-compose.dev.yml`), postgres `:5432`, redis `:6379`, ollama `:11434`.

## Repository layout

```
services/
├── gateway-go/          Go: auth, job submit/poll, WebSocket status
├── ml-engine-py/        Python/FastAPI: pipeline core + Celery tasks
├── agent-orchestrator/  Python/LangGraph: bounded summarizer, Ollama client
└── frontend-react/      React/TS SPA
contracts/               Shared JSON Schema (job model) used across services
infra/                   Postgres migrations, Redis config
tests/
├── integration/         cross-service contract validation
└── research/            reproducibility + benchmark harness
```

This is a from-scratch build. No module here is derived from or should be
reconciled against any prior codebase.