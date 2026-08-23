-- Phase 8: track the Celery chain task ID on the runs row so the Go gateway
-- can correlate async task state (from Redis/Celery) back to the lineage run.
ALTER TABLE runs ADD COLUMN celery_task_id TEXT;
