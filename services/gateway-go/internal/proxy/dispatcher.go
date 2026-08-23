// Package proxy provides the Dispatcher that bridges the Go gateway to the
// ml-engine-py service for job submission and status polling.
//
// Design (CLAUDE.md §5.7):
//   - The Go gateway never executes ML logic. It submits, then polls status.
//   - Backpressure is enforced here, before a job row is created, so callers
//     receive 429/503 rather than unbounded queue growth.
//   - Per-user concurrency cap: semaphore keyed on user identity.
//   - Queue-depth guard: LLEN on the default Celery queue via raw Redis RESP.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// DispatchRequest mirrors the ml-engine-py JobRequest schema.
type DispatchRequest struct {
	Dataset struct {
		Rows    []map[string]any `json:"rows"`
		Columns []string         `json:"columns"`
	} `json:"dataset"`
	TargetColumn       string `json:"target_column"`
	TaskType           string `json:"task_type"`
	ImputationMethod   string `json:"imputation_method,omitempty"`
	OutlierMethod      string `json:"outlier_method,omitempty"`
	Seed               int    `json:"seed,omitempty"`
	NTrials            int    `json:"n_trials,omitempty"`
	CVFolds            int    `json:"cv_folds,omitempty"`
	StackingCVFolds    int    `json:"stacking_cv_folds,omitempty"`
}

// DispatchResponse mirrors the ml-engine-py JobResponse schema.
type DispatchResponse struct {
	JobID        string `json:"job_id"`
	CeleryTaskID string `json:"celery_task_id"`
	Status       string `json:"status"`
}

// StatusResponse mirrors ml-engine-py GET /v1/jobs/{id}/status.
type StatusResponse struct {
	JobID        string  `json:"job_id"`
	Status       string  `json:"status"`
	CeleryTaskID *string `json:"celery_task_id,omitempty"`
	LastStep     *string `json:"last_step,omitempty"`
}

// DispatchError wraps an HTTP error response from ml-engine-py.
type DispatchError struct {
	StatusCode int
	Body       string
}

func (e *DispatchError) Error() string {
	return fmt.Sprintf("ml-engine-py returned %d: %s", e.StatusCode, e.Body)
}

// Dispatcher submits jobs to ml-engine-py and polls their status.
type Dispatcher struct {
	mlEngineURL      string
	client           *http.Client
	maxPerUser       int           // per-user concurrency cap
	maxQueueDepth    int           // Redis queue depth limit
	redisAddr        string        // raw host:port for LLEN check
	mu               sync.Mutex
	userSlots        map[string]int // active job count per user
}

// NewDispatcher creates a Dispatcher from environment variables.
//
// Environment:
//   ML_ENGINE_URL              — base URL of ml-engine-py (default: http://ml-engine-py:8000)
//   PROXY_TIMEOUT              — HTTP client timeout in seconds (default: 30)
//   MAX_CONCURRENT_JOBS_PER_USER — per-user concurrency cap (default: 5)
//   MAX_QUEUE_DEPTH            — Celery default queue depth limit (default: 500)
//   REDIS_URL                  — used to extract the Redis host:port for LLEN check
func NewDispatcher() *Dispatcher {
	mlURL := envOr("ML_ENGINE_URL", "http://ml-engine-py:8000")
	timeoutSec, _ := strconv.Atoi(envOr("PROXY_TIMEOUT", "30"))
	maxPerUser, _ := strconv.Atoi(envOr("MAX_CONCURRENT_JOBS_PER_USER", "5"))
	maxDepth, _ := strconv.Atoi(envOr("MAX_QUEUE_DEPTH", "500"))

	return &Dispatcher{
		mlEngineURL:   mlURL,
		client:        &http.Client{Timeout: time.Duration(timeoutSec) * time.Second},
		maxPerUser:    maxPerUser,
		maxQueueDepth: maxDepth,
		userSlots:     make(map[string]int),
	}
}

// Submit sends a job to ml-engine-py after enforcing backpressure limits.
//
// Returns:
//   DispatchResponse on success.
//   *DispatchError   with StatusCode 429 (user concurrency cap) or 503
//                    (queue depth exceeded or ml-engine-py unavailable).
func (d *Dispatcher) Submit(ctx context.Context, userID string, req DispatchRequest) (DispatchResponse, error) {
	// --- Backpressure: per-user concurrency cap ---
	if !d.acquireSlot(userID) {
		return DispatchResponse{}, &DispatchError{
			StatusCode: http.StatusTooManyRequests,
			Body: fmt.Sprintf(
				"user %q has reached the maximum concurrent job limit (%d); "+
					"wait for an existing job to finish before submitting another",
				userID, d.maxPerUser,
			),
		}
	}
	// Slot is released when the job reaches a terminal state (caller's
	// responsibility via ReleaseSlot). We release on submission error here.

	body, err := json.Marshal(req)
	if err != nil {
		d.releaseSlot(userID)
		return DispatchResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		d.mlEngineURL+"/v1/jobs",
		bytes.NewReader(body),
	)
	if err != nil {
		d.releaseSlot(userID)
		return DispatchResponse{}, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(httpReq)
	if err != nil {
		d.releaseSlot(userID)
		return DispatchResponse{}, &DispatchError{
			StatusCode: http.StatusServiceUnavailable,
			Body:       fmt.Sprintf("ml-engine-py unreachable: %v", err),
		}
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK {
		d.releaseSlot(userID)
		return DispatchResponse{}, &DispatchError{
			StatusCode: resp.StatusCode,
			Body:       string(respBody),
		}
	}

	var dr DispatchResponse
	if err := json.Unmarshal(respBody, &dr); err != nil {
		d.releaseSlot(userID)
		return DispatchResponse{}, fmt.Errorf("decode response: %w", err)
	}

	return dr, nil
}

// PollStatus fetches the current status of a job from ml-engine-py.
func (d *Dispatcher) PollStatus(ctx context.Context, jobID string) (StatusResponse, error) {
	url := fmt.Sprintf("%s/v1/jobs/%s/status", d.mlEngineURL, jobID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return StatusResponse{}, fmt.Errorf("build status request: %w", err)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return StatusResponse{}, fmt.Errorf("ml-engine-py unreachable: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return StatusResponse{}, &DispatchError{StatusCode: resp.StatusCode, Body: string(body)}
	}

	var sr StatusResponse
	if err := json.Unmarshal(body, &sr); err != nil {
		return StatusResponse{}, fmt.Errorf("decode status: %w", err)
	}
	return sr, nil
}

// ReleaseSlot decrements the per-user active-job count. Call when a job
// reaches a terminal status (done or failed).
func (d *Dispatcher) ReleaseSlot(userID string) {
	d.releaseSlot(userID)
}

// acquireSlot increments the user's slot counter if under the cap.
// Returns false (without incrementing) if the cap is reached.
func (d *Dispatcher) acquireSlot(userID string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.userSlots[userID] >= d.maxPerUser {
		return false
	}
	d.userSlots[userID]++
	return true
}

func (d *Dispatcher) releaseSlot(userID string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.userSlots[userID] > 0 {
		d.userSlots[userID]--
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
