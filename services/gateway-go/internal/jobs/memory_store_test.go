package jobs

import (
	"context"
	"testing"
	"time"
)

func TestMemoryStore_CreateAndGetJob(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	job, err := store.CreateJob(ctx, SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	if job.Status != StatusQueued {
		t.Fatalf("expected new job to be queued, got %s", job.Status)
	}
	if job.ConfigHash == "" {
		t.Fatal("expected a non-empty config hash")
	}

	got, err := store.GetJob(ctx, job.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if got.ID != job.ID {
		t.Fatalf("expected job id %s, got %s", job.ID, got.ID)
	}
}

func TestMemoryStore_GetJob_NotFound(t *testing.T) {
	store := NewMemoryStore()
	_, err := store.GetJob(context.Background(), "does-not-exist")
	if err != ErrNotFound {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestMemoryStore_UpdateStatus(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	job, err := store.CreateJob(ctx, SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	updated, err := store.UpdateStatus(ctx, job.ID, StatusRunning)
	if err != nil {
		t.Fatalf("UpdateStatus: %v", err)
	}
	if updated.Status != StatusRunning {
		t.Fatalf("expected status running, got %s", updated.Status)
	}
	if !updated.UpdatedAt.After(job.UpdatedAt) && updated.UpdatedAt != job.UpdatedAt {
		t.Fatal("expected updated_at to advance or stay equal, never go backwards")
	}
}

func TestMemoryStore_UpdateStatus_NotFound(t *testing.T) {
	store := NewMemoryStore()
	_, err := store.UpdateStatus(context.Background(), "does-not-exist", StatusRunning)
	if err != ErrNotFound {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestMemoryStore_Subscribe_ReceivesTransitions(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	job, err := store.CreateJob(ctx, SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	ch, unsubscribe := store.Subscribe(job.ID)
	defer unsubscribe()

	if _, err := store.UpdateStatus(ctx, job.ID, StatusRunning); err != nil {
		t.Fatalf("UpdateStatus: %v", err)
	}

	select {
	case got := <-ch:
		if got.Status != StatusRunning {
			t.Fatalf("expected running, got %s", got.Status)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for status transition on subscription channel")
	}
}

func TestMemoryStore_Unsubscribe_ClosesChannel(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	job, err := store.CreateJob(ctx, SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	ch, unsubscribe := store.Subscribe(job.ID)
	unsubscribe()

	if _, ok := <-ch; ok {
		t.Fatal("expected channel to be closed after unsubscribe")
	}
}

func TestHashConfig_StableAcrossKeyOrder(t *testing.T) {
	a, err := hashConfig(map[string]any{"a": 1, "b": 2})
	if err != nil {
		t.Fatalf("hashConfig: %v", err)
	}
	b, err := hashConfig(map[string]any{"b": 2, "a": 1})
	if err != nil {
		t.Fatalf("hashConfig: %v", err)
	}
	if a != b {
		t.Fatalf("expected identical hash regardless of key order, got %s vs %s", a, b)
	}
}

func TestHashConfig_DifferentValuesDifferentHash(t *testing.T) {
	a, _ := hashConfig(map[string]any{"a": 1})
	b, _ := hashConfig(map[string]any{"a": 2})
	if a == b {
		t.Fatal("expected different configs to hash differently")
	}
}
