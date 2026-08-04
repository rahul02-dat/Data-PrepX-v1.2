package ws

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/gorilla/websocket"

	"dataprepx/gateway-go/internal/jobs"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Handler struct {
	store jobs.Store
}

// Construct WebSocket status stream handler
func NewHandler(store jobs.Store) *Handler {
	return &Handler{store: store}
}

// Stream job status transitions over WebSocket
func (h *Handler) Stream(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

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

// Write JSON-encoded job snapshot to WebSocket
func writeJob(conn *websocket.Conn, job jobs.Job) error {
	b, err := json.Marshal(job)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, b)
}
