package jobs

import (
	"encoding/json"
	"errors"
	"net/http"
)

type Handler struct {
	store     Store
	onCreated func(Job)
}

// Construct new job HTTP handler
func NewHandler(store Store) *Handler {
	return &Handler{store: store}
}

// Register job creation callback
func (h *Handler) OnCreated(fn func(Job)) {
	h.onCreated = fn
}

// Handle POST /v1/jobs endpoint
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

// Handle GET /v1/jobs/{id} endpoint
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

// Helper to write JSON response
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
