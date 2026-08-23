package proxy_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"dataprepx/gateway-go/internal/proxy"
)

func makeDispatcherWithServer(t *testing.T, handler http.HandlerFunc) (*proxy.Dispatcher, *httptest.Server, func()) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Setenv("ML_ENGINE_URL", srv.URL)
	t.Setenv("PROXY_TIMEOUT", "5")
	t.Setenv("MAX_CONCURRENT_JOBS_PER_USER", "2")
	t.Setenv("MAX_QUEUE_DEPTH", "500")
	d := proxy.NewDispatcher()
	return d, srv, func() { srv.Close() }
}

func TestDispatcher_Submit_HappyPath(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/jobs" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id":         "run-abc",
			"celery_task_id": "task-xyz",
			"status":         "queued",
		})
	}

	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	resp, err := d.Submit(context.Background(), "user1", proxy.DispatchRequest{
		TargetColumn: "label",
		TaskType:     "classification",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.JobID != "run-abc" {
		t.Errorf("expected job_id=run-abc, got %q", resp.JobID)
	}
	if resp.CeleryTaskID != "task-xyz" {
		t.Errorf("expected celery_task_id=task-xyz, got %q", resp.CeleryTaskID)
	}
}

func TestDispatcher_Submit_ConcurrencyCap(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": "j", "celery_task_id": "t", "status": "queued",
		})
	}

	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	for i := 0; i < 2; i++ {
		_, err := d.Submit(context.Background(), "user-capped", proxy.DispatchRequest{TaskType: "classification"})
		if err != nil {
			t.Fatalf("job %d: expected success, got %v", i, err)
		}
	}

	_, err := d.Submit(context.Background(), "user-capped", proxy.DispatchRequest{TaskType: "classification"})
	if err == nil {
		t.Fatal("expected 429 error when cap exceeded, got nil")
	}
	de, ok := err.(*proxy.DispatchError)
	if !ok {
		t.Fatalf("expected *DispatchError, got %T", err)
	}
	if de.StatusCode != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", de.StatusCode)
	}
}

func TestDispatcher_ReleaseSlot_AllowsNextJob(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": "j", "celery_task_id": "t", "status": "queued",
		})
	}

	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	for i := 0; i < 2; i++ {
		_, err := d.Submit(context.Background(), "user-rel", proxy.DispatchRequest{TaskType: "classification"})
		if err != nil {
			t.Fatalf("slot %d: %v", i, err)
		}
	}

	d.ReleaseSlot("user-rel")

	_, err := d.Submit(context.Background(), "user-rel", proxy.DispatchRequest{TaskType: "classification"})
	if err != nil {
		t.Fatalf("expected success after ReleaseSlot, got %v", err)
	}
}

func TestDispatcher_PerUserIsolation(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": "j", "celery_task_id": "t", "status": "queued",
		})
	}

	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	for i := 0; i < 2; i++ {
		_, err := d.Submit(context.Background(), "user-A", proxy.DispatchRequest{TaskType: "classification"})
		if err != nil {
			t.Fatalf("user-A slot %d: %v", i, err)
		}
	}

	_, err := d.Submit(context.Background(), "user-B", proxy.DispatchRequest{TaskType: "classification"})
	if err != nil {
		t.Fatalf("user-B should succeed independently: %v", err)
	}
}

func TestDispatcher_PollStatus(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": "run-abc",
			"status": "optimizing",
		})
	}

	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	sr, err := d.PollStatus(context.Background(), "run-abc")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sr.Status != "optimizing" {
		t.Errorf("expected status=optimizing, got %q", sr.Status)
	}
}

func TestDispatcher_PollStatus_NotFound(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}

	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	_, err := d.PollStatus(context.Background(), "no-such-job")
	if err == nil {
		t.Fatal("expected error for 404, got nil")
	}
}

func TestDispatcher_Submit_Unreachable(t *testing.T) {
	t.Setenv("ML_ENGINE_URL", "http://127.0.0.1:19999")
	t.Setenv("PROXY_TIMEOUT", "1")
	t.Setenv("MAX_CONCURRENT_JOBS_PER_USER", "5")
	d := proxy.NewDispatcher()

	_, err := d.Submit(context.Background(), "user1", proxy.DispatchRequest{TaskType: "classification"})
	if err == nil {
		t.Fatal("expected error when ml-engine unreachable, got nil")
	}
	de, ok := err.(*proxy.DispatchError)
	if !ok {
		t.Fatalf("expected *DispatchError, got %T: %v", err, err)
	}
	if de.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", de.StatusCode)
	}
}

func TestDispatcher_ConcurrentSubmit_RaceFree(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"job_id": "j", "celery_task_id": "t", "status": "queued",
		})
	}

	t.Setenv("MAX_CONCURRENT_JOBS_PER_USER", "10")
	d, _, cleanup := makeDispatcherWithServer(t, handler)
	defer cleanup()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = d.Submit(context.Background(), "race-user", proxy.DispatchRequest{TaskType: "classification"})
		}()
	}
	wg.Wait()
}

