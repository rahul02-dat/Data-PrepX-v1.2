package jobs

import (
	"encoding/json"
	"errors"
	"net/http"
)

// Handler serves the job submission and polling REST surface. WebSocket
// streaming lives in the sibling internal/ws package, wired up in main.go
// on its own route since it needs both a Store and a running worker to
// drive transitions.
type Handler struct {
	store     Store
	onCreated func(Job)
}

// NewHandler returns a Handler backed by store. Register its methods
// directly against a Go 1.22+ ServeMux pattern (see main.go):
//
//	mux.HandleFunc("POST /v1/jobs", h.Submit)
//	mux.HandleFunc("GET /v1/jobs/{id}", h.Poll)
func NewHandler(store Store) *Handler {
	return &Handler{store: store}
}

// OnCreated registers a callback invoked synchronously, after the response
// has been written, with every job this Handler creates. main.go uses this
// to kick off the fake worker (RunFakeWorker) without Submit needing to
// know that worker exists.
func (h *Handler) OnCreated(fn func(Job)) {
	h.onCreated = fn
}

// Submit handles POST /v1/jobs: creates a new job and returns it as JSON
// with 201 Created.
func (h *Handler) Submit(w http.ResponseWriter, r *http.Request) {
	var req SubmitRequest
	if r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}
	}

	job, err := h.store.CreateJob(r.Context(), req)
	if err != nil {
		http.Error(w, "failed to create job: "+err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusCreated, job)

	if h.onCreated != nil {
		h.onCreated(job)
	}
}

// Poll handles GET /v1/jobs/{id}: returns the current job state as JSON.
func (h *Handler) Poll(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	job, err := h.store.GetJob(r.Context(), id)
	if errors.Is(err, ErrNotFound) {
		http.Error(w, "job not found", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, "failed to fetch job: "+err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, job)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
