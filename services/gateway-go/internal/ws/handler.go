// Package ws streams job status transitions to WebSocket clients. It is
// deliberately thin: it subscribes to a jobs.Store and forwards whatever
// that store publishes. The Go gateway must never block on compute
// (CLAUDE.md §5.7) -- this handler only relays state, it does not drive it.
package ws

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/gorilla/websocket"

	"dataprepx/gateway-go/internal/jobs"
)

var upgrader = websocket.Upgrader{
	// Phase 1 has no browser-origin restriction yet; the frontend and
	// gateway are same-origin in dev via the Vite proxy / direct port.
	// Revisit this once auth (Phase 8+ hardening, CLAUDE.md §11) lands.
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Handler upgrades GET /v1/jobs/{id}/ws to a WebSocket connection and
// streams every subsequent status transition for that job as a JSON-encoded
// jobs.Job, until the job reaches a terminal state (done/failed) or the
// client disconnects.
type Handler struct {
	store jobs.Store
}

// NewHandler returns a Handler backed by store. Register it against a
// Go 1.22+ ServeMux pattern:
//
//	mux.HandleFunc("GET /v1/jobs/{id}/ws", h.Stream)
func NewHandler(store jobs.Store) *Handler {
	return &Handler{store: store}
}

// Stream handles GET /v1/jobs/{id}/ws.
func (h *Handler) Stream(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	// Send the current state immediately so a client connecting after the
	// job already progressed isn't stuck waiting for the next transition.
	current, err := h.store.GetJob(r.Context(), id)
	if err != nil {
		http.Error(w, "job not found", http.StatusNotFound)
		return
	}

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws: upgrade failed for job %s: %v", id, err)
		return
	}
	defer conn.Close()

	ch, unsubscribe := h.store.Subscribe(id)
	defer unsubscribe()

	if err := writeJob(conn, current); err != nil {
		return
	}
	if current.Status == jobs.StatusDone || current.Status == jobs.StatusFailed {
		return
	}

	for job := range ch {
		if err := writeJob(conn, job); err != nil {
			return
		}
		if job.Status == jobs.StatusDone || job.Status == jobs.StatusFailed {
			return
		}
	}
}

func writeJob(conn *websocket.Conn, job jobs.Job) error {
	b, err := json.Marshal(job)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, b)
}
