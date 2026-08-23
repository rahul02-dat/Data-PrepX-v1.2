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

type DispatchRequest struct {
	Dataset struct {
		Rows    []map[string]any `json:"rows"`
		Columns []string         `json:"columns"`
	} `json:"dataset"`
	TargetColumn     string `json:"target_column"`
	TaskType         string `json:"task_type"`
	ImputationMethod string `json:"imputation_method,omitempty"`
	OutlierMethod    string `json:"outlier_method,omitempty"`
	Seed             int    `json:"seed,omitempty"`
	NTrials          int    `json:"n_trials,omitempty"`
	CVFolds          int    `json:"cv_folds,omitempty"`
	StackingCVFolds  int    `json:"stacking_cv_folds,omitempty"`
}

type DispatchResponse struct {
	JobID        string `json:"job_id"`
	CeleryTaskID string `json:"celery_task_id"`
	Status       string `json:"status"`
}

type StatusResponse struct {
	JobID        string  `json:"job_id"`
	Status       string  `json:"status"`
	CeleryTaskID *string `json:"celery_task_id,omitempty"`
	LastStep     *string `json:"last_step,omitempty"`
}

type DispatchError struct {
	StatusCode int
	Body       string
}

func (e *DispatchError) Error() string {
	return fmt.Sprintf("ml-engine-py returned %d: %s", e.StatusCode, e.Body)
}

type Dispatcher struct {
	mlEngineURL   string
	client        *http.Client
	maxPerUser    int
	maxQueueDepth int
	redisAddr     string
	mu            sync.Mutex
	userSlots     map[string]int
}

// NewDispatcher initializes a Dispatcher using environment configuration.
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

// Submit sends a job dispatch request to ml-engine-py under concurrency controls.
func (d *Dispatcher) Submit(ctx context.Context, userID string, req DispatchRequest) (DispatchResponse, error) {
	if !d.acquireSlot(userID) {
		return DispatchResponse{}, &DispatchError{
			StatusCode: http.StatusTooManyRequests,
			Body: fmt.Sprintf(
				"user %q has reached the maximum concurrent job limit (%d); wait for an existing job to finish",
				userID, d.maxPerUser,
			),
		}
	}

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

// PollStatus queries ml-engine-py for the current status of a job.
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

// ReleaseSlot releases a concurrency slot for the given user.
func (d *Dispatcher) ReleaseSlot(userID string) {
	d.releaseSlot(userID)
}

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
