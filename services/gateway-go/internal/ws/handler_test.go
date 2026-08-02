package ws

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"dataprepx/gateway-go/internal/jobs"
)

// TestEndToEnd_SubmitAndStreamStatus mirrors Phase 1's acceptance criterion:
// a synthetic no-op job can be submitted via REST, recorded in the store,
// and its status streamed to a WebSocket client end to end.
func TestEndToEnd_SubmitAndStreamStatus(t *testing.T) {
	store := jobs.NewMemoryStore()

	jobsHandler := jobs.NewHandler(store)
	jobsHandler.OnCreated(func(job jobs.Job) {
		go func() {
			_ = jobs.RunFakeWorker(context.Background(), store, job.ID, 10*time.Millisecond)
		}()
	})
	wsHandler := NewHandler(store)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/jobs", jobsHandler.Submit)
	mux.HandleFunc("GET /v1/jobs/{id}", jobsHandler.Poll)
	mux.HandleFunc("GET /v1/jobs/{id}/ws", wsHandler.Stream)

	server := httptest.NewServer(mux)
	defer server.Close()

	// 1. Submit the job over REST.
	resp, err := http.Post(server.URL+"/v1/jobs", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("submit request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("expected 201, got %d", resp.StatusCode)
	}

	var created jobs.Job
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		t.Fatalf("failed to decode submit response: %v", err)
	}
	if created.Status != jobs.StatusQueued {
		t.Fatalf("expected initial status queued, got %s", created.Status)
	}

	// 2. Connect to the WebSocket stream for that job.
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/v1/jobs/" + created.ID + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("websocket dial failed: %v", err)
	}
	defer conn.Close()

	// 3. Read the full transition sequence: queued (current state on
	// connect) -> running -> done.
	var statuses []jobs.Status
	deadline := time.Now().Add(5 * time.Second)
	for {
		conn.SetReadDeadline(deadline)
		_, msg, err := conn.ReadMessage()
		if err != nil {
			t.Fatalf("websocket read failed after statuses %v: %v", statuses, err)
		}
		var job jobs.Job
		if err := json.Unmarshal(msg, &job); err != nil {
			t.Fatalf("failed to decode ws message: %v", err)
		}
		statuses = append(statuses, job.Status)
		if job.Status == jobs.StatusDone {
			break
		}
	}

	want := []jobs.Status{jobs.StatusQueued, jobs.StatusRunning, jobs.StatusDone}
	if len(statuses) != len(want) {
		t.Fatalf("expected status sequence %v, got %v", want, statuses)
	}
	for i, s := range want {
		if statuses[i] != s {
			t.Fatalf("expected status sequence %v, got %v", want, statuses)
		}
	}

	// 4. Poll confirms the same final state via REST.
	pollResp, err := http.Get(server.URL + "/v1/jobs/" + created.ID)
	if err != nil {
		t.Fatalf("poll request failed: %v", err)
	}
	defer pollResp.Body.Close()
	var polled jobs.Job
	if err := json.NewDecoder(pollResp.Body).Decode(&polled); err != nil {
		t.Fatalf("failed to decode poll response: %v", err)
	}
	if polled.Status != jobs.StatusDone {
		t.Fatalf("expected polled status done, got %s", polled.Status)
	}
}
