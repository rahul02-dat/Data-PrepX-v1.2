ALTER TABLE runs ADD COLUMN run_key TEXT;
CREATE UNIQUE INDEX idx_runs_run_key ON runs(run_key) WHERE run_key IS NOT NULL;

CREATE TABLE gate_evaluations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    gate_name   TEXT NOT NULL,
    passed      BOOLEAN NOT NULL,
    reason      TEXT,
    details     JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_gate_evaluations_run_id ON gate_evaluations(run_id);

ALTER TABLE datasets ADD COLUMN reference_dataset_id UUID REFERENCES datasets(id);
