.PHONY: setup up down migrate migrate-down test test-go test-py test-frontend lint

setup:
	cd services/ml-engine-py && python3 -m venv .venv && .venv/bin/pip install -e .[dev]
	cd services/agent-orchestrator && python3 -m venv .venv && .venv/bin/pip install -e .[dev]
	cd services/frontend-react && npm install
	cd tests/integration && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

up:
	docker compose up --build

down:
	docker compose down

migrate:
	docker compose run --rm migrate

migrate-down:
	docker compose run --rm --entrypoint migrate migrate \
		-path=/migrations \
		-database=postgres://dataprepx:dataprepx@postgres:5432/dataprepx?sslmode=disable \
		down 1

test: test-go test-py test-frontend test-contracts

test-go:
	cd services/gateway-go && go vet ./... && go test ./... -v

test-py:
	cd services/ml-engine-py && .venv/bin/pytest -v
	cd services/agent-orchestrator && .venv/bin/pytest -v

test-frontend:
	cd services/frontend-react && npx vitest run

test-contracts:
	cd tests/integration && .venv/bin/pytest -v

lint:
	cd services/gateway-go && gofmt -l . && go vet ./...
	cd services/ml-engine-py && .venv/bin/ruff check . && .venv/bin/black --check .
	cd services/agent-orchestrator && .venv/bin/ruff check . && .venv/bin/black --check .
	cd services/frontend-react && npx tsc -b
