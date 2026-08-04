ALTER TABLE datasets DROP COLUMN IF EXISTS reference_dataset_id;
DROP INDEX IF EXISTS idx_gate_evaluations_run_id;
DROP TABLE IF EXISTS gate_evaluations;
DROP INDEX IF EXISTS idx_runs_run_key;
ALTER TABLE runs DROP COLUMN IF EXISTS run_key;
