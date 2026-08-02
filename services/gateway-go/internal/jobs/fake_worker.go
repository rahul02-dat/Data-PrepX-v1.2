package jobs

import (
	"context"
	"time"
)

// RunFakeWorker walks a newly created job through the fixed sequence
// queued -> running -> done, on the given delay between transitions. This
// stands in for the real Celery task graph (planner Phase 8) which does not
// exist yet -- there is no real pipeline work to report progress on for a
// synthetic no-op job. It intentionally never produces gate-check,
// optimizing, or failed: those require real pipeline stages to be
// meaningful and are introduced by the phases that build them.
//
// Call this in a new goroutine per job; it returns once the job reaches
// StatusDone or the context is cancelled.
func RunFakeWorker(ctx context.Context, store Store, jobID string, delay time.Duration) error {
	transitions := []Status{StatusRunning, StatusDone}

	for _, status := range transitions {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}

		if _, err := store.UpdateStatus(ctx, jobID, status); err != nil {
			return err
		}
	}

	return nil
}
