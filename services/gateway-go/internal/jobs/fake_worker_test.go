package jobs

import (
	"context"
	"testing"
	"time"
)

func TestRunFakeWorker_QueuedRunningDone(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	job, err := store.CreateJob(ctx, SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	if job.Status != StatusQueued {
		t.Fatalf("expected initial status queued, got %s", job.Status)
	}

	if err := RunFakeWorker(ctx, store, job.ID, time.Millisecond); err != nil {
		t.Fatalf("RunFakeWorker: %v", err)
	}

	final, err := store.GetJob(ctx, job.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if final.Status != StatusDone {
		t.Fatalf("expected final status done, got %s", final.Status)
	}
}

func TestRunFakeWorker_RespectsContextCancellation(t *testing.T) {
	store := NewMemoryStore()
	job, err := store.CreateJob(context.Background(), SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = RunFakeWorker(ctx, store, job.ID, time.Hour)
	if err == nil {
		t.Fatal("expected RunFakeWorker to return an error when the context is already cancelled")
	}
}
