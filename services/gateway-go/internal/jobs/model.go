package jobs

import (
	"context"
	"time"

	"dataprepx/gateway-go/internal/proxy"
)

type Status string

const (
	StatusQueued     Status = "queued"
	StatusRunning    Status = "running"
	StatusGateCheck  Status = "gate-check"
	StatusOptimizing Status = "optimizing"
	StatusDone       Status = "done"
	StatusFailed     Status = "failed"
)

type SubmitRequest struct {
	DatasetID *string        `json:"dataset_id,omitempty"`
	Config    map[string]any `json:"config,omitempty"`
	proxy.DispatchRequest
}

type Job struct {
	ID           string    `json:"id"`
	DatasetID    *string   `json:"dataset_id,omitempty"`
	Status       Status    `json:"status"`
	ConfigHash   string    `json:"config_hash"`
	CeleryTaskID *string   `json:"celery_task_id,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

type Store interface {
	CreateJob(ctx context.Context, req SubmitRequest) (Job, error)
	GetJob(ctx context.Context, id string) (Job, error)
	UpdateStatus(ctx context.Context, id string, status Status) (Job, error)
	Subscribe(id string) (<-chan Job, func())
}

var ErrNotFound = &notFoundError{}

type notFoundError struct{}

// Return error message for missing job
func (*notFoundError) Error() string { return "job not found" }
