CREATE TABLE datasets (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash  TEXT NOT NULL UNIQUE,
    schema_json   JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE runs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id    UUID REFERENCES datasets(id),
    git_sha       TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'running', 'gate-check', 'optimizing', 'done', 'failed')),
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_runs_dataset_id ON runs(dataset_id);
CREATE INDEX idx_runs_status ON runs(status);

CREATE TABLE pipeline_steps (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_type     TEXT NOT NULL,
    input_hash    TEXT NOT NULL,
    output_hash   TEXT,
    params_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
    seed          BIGINT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_pipeline_steps_run_id ON pipeline_steps(run_id);

CREATE TABLE transformations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    step_id             UUID NOT NULL REFERENCES pipeline_steps(id) ON DELETE CASCADE,
    transform_code_hash TEXT NOT NULL,
    description         TEXT
);

CREATE INDEX idx_transformations_step_id ON transformations(step_id);

CREATE TABLE hyperparameters (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model_family  TEXT NOT NULL,
    trial_number  INTEGER NOT NULL,
    params_json   JSONB NOT NULL,
    score         DOUBLE PRECISION
);

CREATE INDEX idx_hyperparameters_run_id ON hyperparameters(run_id);

CREATE TABLE metrics (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id    UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name      TEXT NOT NULL,
    value     DOUBLE PRECISION NOT NULL,
    ci_low    DOUBLE PRECISION,
    ci_high   DOUBLE PRECISION
);

CREATE INDEX idx_metrics_run_id ON metrics(run_id);

CREATE TABLE artifacts (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    storage_uri   TEXT NOT NULL,
    content_hash  TEXT NOT NULL
);

CREATE INDEX idx_artifacts_run_id ON artifacts(run_id);

CREATE TABLE audit_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID REFERENCES runs(id) ON DELETE CASCADE,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_log_run_id ON audit_log(run_id);
