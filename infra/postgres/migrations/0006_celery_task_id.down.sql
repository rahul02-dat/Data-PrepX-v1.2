-- Rollback Phase 8 Celery task ID column.
ALTER TABLE runs DROP COLUMN IF EXISTS celery_task_id;
