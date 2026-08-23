package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"time"

	"dataprepx/gateway-go/internal/jobs"
	"dataprepx/gateway-go/internal/proxy"
	"dataprepx/gateway-go/internal/ws"
)

// Main gateway entrypoint
func main() {
	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	store := newStore()
	dispatcher := proxy.NewDispatcher()

	jobsHandler := jobs.NewHandler(store)
	jobsHandler.OnCreated(func(job jobs.Job) {
		// Dispatch to ml-engine-py asynchronously; the gateway immediately
		// returns 202 to the caller while this goroutine handles the slow path.
		go dispatchJob(store, dispatcher, job)
	})

	wsHandler := ws.NewHandler(store)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", handleHealthz)
	mux.HandleFunc("POST /v1/jobs", jobsHandler.Submit)
	mux.HandleFunc("GET /v1/jobs/{id}", jobsHandler.Poll)
	mux.HandleFunc("GET /v1/jobs/{id}/ws", wsHandler.Stream)

	log.Printf("gateway-go listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("gateway-go: server exited: %v", err)
	}
}

// dispatchJob submits the job to ml-engine-py and then polls until the job
// reaches a terminal state, updating the local store (and therefore the
// WebSocket hub) on each status transition.
func dispatchJob(store jobs.Store, dispatcher *proxy.Dispatcher, job jobs.Job) {
	// "anonymous" as the user ID until Phase 9 adds auth. When auth lands,
	// replace this with the authenticated user identity from the JWT/session so
	// the per-user concurrency cap in the dispatcher is per-real-user.
	const userID = "anonymous"

	ctx := context.Background()

	// Build a minimal DispatchRequest from the job. In Phase 9 the full
	// dataset and config will be part of the submit payload; for now we forward
	// what the gateway received.
	req := proxy.DispatchRequest{}
	req.TaskType = "classification" // placeholder until Phase 9 adds typed submit
	req.TargetColumn = ""           // placeholder — ml-engine uses last column as default

	dr, err := dispatcher.Submit(ctx, userID, req)
	if err != nil {
		var de *proxy.DispatchError
		if errors.As(err, &de) {
			log.Printf("gateway-go: dispatch rejected job %s (%d): %s", job.ID, de.StatusCode, de.Body)
		} else {
			log.Printf("gateway-go: dispatch error job %s: %v", job.ID, err)
		}
		if _, updateErr := store.UpdateStatus(ctx, job.ID, jobs.StatusFailed); updateErr != nil {
			log.Printf("gateway-go: failed to mark job %s as failed: %v", job.ID, updateErr)
		}
		return
	}

	log.Printf("gateway-go: job %s dispatched to ml-engine-py as celery_task=%s", job.ID, dr.CeleryTaskID)

	// Poll ml-engine-py /status until the job is terminal.
	pollInterval := 1 * time.Second
	lastStatus := jobs.StatusQueued
	for {
		time.Sleep(pollInterval)

		sr, err := dispatcher.PollStatus(ctx, dr.JobID)
		if err != nil {
			log.Printf("gateway-go: poll error job %s: %v; retrying", job.ID, err)
			continue
		}

		newStatus := jobs.Status(sr.Status)
		if newStatus != lastStatus {
			if _, err := store.UpdateStatus(ctx, job.ID, newStatus); err != nil {
				log.Printf("gateway-go: UpdateStatus error job %s → %s: %v", job.ID, newStatus, err)
			}
			lastStatus = newStatus
		}

		if newStatus == jobs.StatusDone || newStatus == jobs.StatusFailed {
			dispatcher.ReleaseSlot(userID)
			log.Printf("gateway-go: job %s reached terminal status %s", job.ID, newStatus)
			return
		}
	}
}

// Select job store implementation
func newStore() jobs.Store {
	if os.Getenv("JOB_STORE") == "memory" {
		log.Print("gateway-go: using in-memory job store (JOB_STORE=memory)")
		return jobs.NewMemoryStore()
	}

	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Fatal("gateway-go: DATABASE_URL is required unless JOB_STORE=memory")
	}
	store, err := jobs.NewPostgresStore(dsn)
	if err != nil {
		log.Fatalf("gateway-go: failed to connect to Postgres: %v", err)
	}
	return store
}

// Service health check endpoint
func handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok","service":"gateway-go"}`))
}
