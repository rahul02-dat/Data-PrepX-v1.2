DROP INDEX IF EXISTS idx_runs_run_key;
CREATE UNIQUE INDEX idx_runs_run_key ON runs(run_key) WHERE run_key IS NOT NULL;