-- Phase 2: validation gates & immutable lineage (CLAUDE.md §5.3, planner §4
-- Phase 2). Two additions to the Phase 1 schema:
--
-- 1. runs.run_key -- CLAUDE.md §6 states "run_id should be derivable purely
--    from (dataset content_hash, config_hash, git_sha)". The Phase 1 schema
--    used a random gen_random_uuid() as runs.id, which cannot itself satisfy
--    that property (two identical inputs would get two different ids). We
--    keep `id` as the surrogate primary key (existing FKs already point at
--    it) and add `run_key`, a deterministic sha256 over
--    (dataset content_hash || config_hash || git_sha), with a unique
--    constraint. Run creation becomes idempotent on run_key: the same three
--    inputs resolve to the same row instead of minting a new one. This is
--    the actual mechanism behind "any run_id is derivable / replayable" --
--    see docs/adr/0002-drift-reference-distribution.md sibling ADR
--    0003-run-id-determinism.md for the full rationale.
--
-- 2. gate_evaluations -- CLAUDE.md §5.3 requires "rejection reason is
--    structured, not a free-text log line." audit_log.action is a bare TEXT
--    field, insufficient for a structured gate verdict (which gate, pass/
--    fail, numeric details for replay/debugging). This table is the
--    structured record; audit_log still gets a coarse-grained entry
--    ('gate_evaluation_run') for the human-readable timeline.

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

-- Every gated dataset needs a fixed reference to compare against for the
-- DriftGate. Per docs/adr/0002-drift-reference-distribution.md this project
-- uses a user-supplied reference dataset, not the first-run baseline -- so
-- that choice has to be stored explicitly per dataset, not inferred.
ALTER TABLE datasets ADD COLUMN reference_dataset_id UUID REFERENCES datasets(id);
