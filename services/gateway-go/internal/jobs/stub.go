// Package jobs will hold job submission, polling, and WebSocket streaming
// logic (CLAUDE.md §2, planner Phase 1). For Phase 0 this is a stub that
// only proves the route exists and the service boots; it deliberately does
// not talk to Postgres or Redis yet.
package jobs

import (
	"encoding/json"
	"net/http"
)

type stubHandler struct{}

// NewStubHandler returns a placeholder /v1/jobs handler for Phase 0.
// It responds to GET and POST with a fixed payload so the gateway's route
// table and the frontend's client code can be wired up before the real
// job model (Phase 1) exists. It is not the real implementation.
func NewStubHandler() http.Handler {
	return &stubHandler{}
}

func (h *stubHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet, http.MethodPost:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"status": "not_implemented",
			"note":   "job model lands in Phase 1; this route only proves wiring",
		})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
