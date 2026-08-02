package jobs

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestMux(store Store) http.Handler {
	h := NewHandler(store)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/jobs", h.Submit)
	mux.HandleFunc("GET /v1/jobs/{id}", h.Poll)
	return mux
}

func TestHandler_Submit(t *testing.T) {
	store := NewMemoryStore()
	mux := newTestMux(store)

	body := bytes.NewBufferString(`{"config":{"foo":"bar"}}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", body)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rec.Code, rec.Body.String())
	}

	var job Job
	if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if job.ID == "" {
		t.Fatal("expected a non-empty job id")
	}
	if job.Status != StatusQueued {
		t.Fatalf("expected queued status, got %s", job.Status)
	}
}

func TestHandler_Submit_EmptyBody(t *testing.T) {
	store := NewMemoryStore()
	mux := newTestMux(store)

	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("expected 201 for a synthetic no-op job with no body, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestHandler_Submit_InvalidJSON(t *testing.T) {
	store := NewMemoryStore()
	mux := newTestMux(store)

	body := bytes.NewBufferString(`{not valid json`)
	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", body)
	req.ContentLength = int64(body.Len())
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid JSON, got %d", rec.Code)
	}
}

func TestHandler_Poll(t *testing.T) {
	store := NewMemoryStore()
	mux := newTestMux(store)

	created, err := store.CreateJob(context.Background(), SubmitRequest{})
	if err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/jobs/"+created.ID, nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var job Job
	if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if job.ID != created.ID {
		t.Fatalf("expected id %s, got %s", created.ID, job.ID)
	}
}

func TestHandler_Poll_NotFound(t *testing.T) {
	store := NewMemoryStore()
	mux := newTestMux(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/jobs/does-not-exist", nil)
	rec := httptest.NewRecorder()

	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rec.Code)
	}
}

func TestHandler_OnCreated_InvokedAfterResponse(t *testing.T) {
	store := NewMemoryStore()
	h := NewHandler(store)

	invoked := make(chan Job, 1)
	h.OnCreated(func(job Job) { invoked <- job })

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/jobs", h.Submit)

	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	select {
	case job := <-invoked:
		if job.ID == "" {
			t.Fatal("expected onCreated to receive a job with a populated id")
		}
	default:
		t.Fatal("expected onCreated to be invoked synchronously during Submit")
	}
}
