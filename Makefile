.PHONY: setup up down migrate migrate-down test test-go test-py test-frontend lint

# Creates local venvs / installs node_modules for running tests/lint outside
# Docker. Not required for `make up`, which builds and runs inside containers.
setup:
	cd services/ml-engine-py && python3 -m venv .venv && .venv/bin/pip install -e .[dev]
	cd services/agent-orchestrator && python3 -m venv .venv && .venv/bin/pip install -e .[dev]
	cd services/frontend-react && npm install

# Boots gateway-go, ml-engine-py, agent-orchestrator, frontend-react,
# postgres, redis, ollama (CLAUDE.md §8).
up:
	docker compose up --build

down:
	docker compose down

# Applies Postgres migrations via the golang-migrate Docker image (see
# docs/adr/0001-migration-tool-choice.md for why the CLI isn't installed
# on the host / in CI images directly).
migrate:
	docker compose run --rm migrate

migrate-down:
	docker compose run --rm --entrypoint migrate migrate \
		-path=/migrations \
		-database=postgres://dataprepx:dataprepx@postgres:5432/dataprepx?sslmode=disable \
		down 1

# Runs Go + Python unit/integration suites (CLAUDE.md §8).
test: test-go test-py test-frontend

test-go:
	cd services/gateway-go && go vet ./... && go test ./... -v

test-py:
	cd services/ml-engine-py && .venv/bin/pytest -v
	cd services/agent-orchestrator && .venv/bin/pytest -v

test-frontend:
	cd services/frontend-react && npx vitest run

lint:
	cd services/gateway-go && gofmt -l . && go vet ./...
	cd services/ml-engine-py && .venv/bin/ruff check . && .venv/bin/black --check .
	cd services/agent-orchestrator && .venv/bin/ruff check . && .venv/bin/black --check .
	cd services/frontend-react && npx tsc -b
