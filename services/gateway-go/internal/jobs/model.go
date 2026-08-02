// Package jobs implements the Phase 1 job/run model: submit a job, poll its
// status, and stream status transitions over WebSocket. This replaces the
// Phase 0 stub handler.
//
// The Store interface is intentionally storage-agnostic. Two implementations
// exist: an in-memory store (memory_store.go, used by tests and available
// for local dev without Postgres) and a real Postgres-backed store
// (postgres_store.go). Only the in-memory store is exercised by this
// sandbox's test suite; the Postgres store's SQL was hand-verified against a
// real Postgres instance during development (see docs/adr) but is not
// covered by an automated integration test here.
package jobs

import (
	"context"
	"time"
)

// Status mirrors the Celery task-graph states in CLAUDE.md §5.7. Phase 1's
// stub worker only ever produces Queued -> Running -> Done|Failed;
// GateCheck and Optimizing are reserved for Phase 2+ once real pipeline
// stages exist to report them.
type Status string

const (
	StatusQueued     Status = "queued"
	StatusRunning    Status = "running"
	StatusGateCheck  Status = "gate-check"
	StatusOptimizing Status = "optimizing"
	StatusDone       Status = "done"
	StatusFailed     Status = "failed"
)

// SubmitRequest is the body of POST /v1/jobs, matching contracts/job.schema.json.
type SubmitRequest struct {
	DatasetID *string        `json:"dataset_id,omitempty"`
	Config    map[string]any `json:"config,omitempty"`
}

// Job is the representation returned by submit/poll and streamed over the
// WebSocket on every status transition, matching contracts/job.schema.json.
type Job struct {
	ID         string    `json:"id"`
	DatasetID  *string   `json:"dataset_id,omitempty"`
	Status     Status    `json:"status"`
	ConfigHash string    `json:"config_hash"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// Store persists jobs and notifies subscribers of status transitions.
// Implementations must be safe for concurrent use.
type Store interface {
	// CreateJob persists a new job in StatusQueued and returns it.
	CreateJob(ctx context.Context, req SubmitRequest) (Job, error)

	// GetJob returns the job with the given id, or ErrNotFound.
	GetJob(ctx context.Context, id string) (Job, error)

	// UpdateStatus transitions a job to a new status, persists it, and
	// notifies any active subscribers. Returns the updated job.
	UpdateStatus(ctx context.Context, id string, status Status) (Job, error)

	// Subscribe registers a listener for every subsequent status
	// transition of the given job. The returned channel receives a Job
	// snapshot on each transition and is closed by unsubscribe. Callers
	// must call unsubscribe when done to avoid leaking the channel.
	Subscribe(id string) (<-chan Job, func())
}

// ErrNotFound is returned by Store.GetJob when no job with the given id exists.
var ErrNotFound = &notFoundError{}

type notFoundError struct{}

func (*notFoundError) Error() string { return "job not found" }
