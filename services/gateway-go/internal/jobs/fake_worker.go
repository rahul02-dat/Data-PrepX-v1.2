package jobs

import (
	"context"
	"time"
)

// Execute synthetic job status transition sequence
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
