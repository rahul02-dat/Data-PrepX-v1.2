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

func main() {
	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	store := newStore()
	dispatcher := proxy.NewDispatcher()

	jobsHandler := jobs.NewHandler(store)
	jobsHandler.OnCreated(func(job jobs.Job, req jobs.SubmitRequest) {
		go dispatchJob(store, dispatcher, job, req.DispatchRequest)
	})

	wsHandler := ws.NewHandler(store)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", handleHealthz)
	mux.HandleFunc("POST /v1/jobs", jobsHandler.Submit)
	mux.HandleFunc("GET /v1/jobs/{id}", jobsHandler.Poll)
	mux.HandleFunc("GET /v1/jobs/{id}/ws", wsHandler.Stream)

	log.Printf("gateway-go listening on %s", addr)
	if err := http.ListenAndServe(addr, corsMiddleware(mux)); err != nil {
		log.Fatalf("gateway-go: server exited: %v", err)
	}
}

// corsMiddleware adds basic CORS headers.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// dispatchJob dispatches a job to ml-engine-py and polls for status updates.
func dispatchJob(store jobs.Store, dispatcher *proxy.Dispatcher, job jobs.Job, req proxy.DispatchRequest) {
	const userID = "anonymous"
	ctx := context.Background()

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

	log.Printf("gateway-go: job %s dispatched (task=%s)", job.ID, dr.CeleryTaskID)

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
				log.Printf("gateway-go: UpdateStatus error job %s -> %s: %v", job.ID, newStatus, err)
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

// newStore initializes the configured job store.
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

// handleHealthz responds with health status.
func handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok","service":"gateway-go"}`))
}
