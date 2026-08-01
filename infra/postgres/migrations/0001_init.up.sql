-- Phase 0: prove the migration tool wiring works end to end.
-- The real schema (datasets, runs, pipeline_steps, transformations,
-- hyperparameters, metrics, artifacts, audit_log) is Phase 1 scope --
-- see CLAUDE.md §6 and docs/01_IMPLEMENTATION_PLANNER.md Phase 1.
--
-- pgcrypto is enabled here because Phase 1's tables use gen_random_uuid()
-- for primary keys.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
