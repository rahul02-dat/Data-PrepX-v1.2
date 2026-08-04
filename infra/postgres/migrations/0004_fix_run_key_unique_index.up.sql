-- 0003_lineage.up.sql created idx_runs_run_key as a PARTIAL unique index
-- (`WHERE run_key IS NOT NULL`). Postgres's ON CONFLICT target inference only
-- matches a partial index when the INSERT's own ON CONFLICT clause repeats the
-- same WHERE predicate. lineage.py's get_or_create_run does a plain
-- `ON CONFLICT (run_key) DO NOTHING`, which fails with:
--   InvalidColumnReference: there is no unique or exclusion constraint
--   matching the ON CONFLICT specification
--
-- The partial index was unnecessary: Postgres unique indexes already treat
-- every NULL as distinct from every other NULL, so rows with run_key IS NULL
-- (e.g. gateway-go's CreateJob rows before lineage.py backfills them) were
-- never at risk of colliding under a full unique index either. Replace the
-- partial index with a full one so ON CONFLICT (run_key) works as written.

DROP INDEX IF EXISTS idx_runs_run_key;
CREATE UNIQUE INDEX idx_runs_run_key ON runs(run_key);