// Command gateway is the DataPrepX v2 API gateway.
//
// Scope for this service, per CLAUDE.md §2: authentication, job submission
// and polling, and WebSocket status fan-out. It must never contain ML logic
// -- if pandas/sklearn-shaped decisions start showing up here, that belongs
// in ml-engine-py instead.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"dataprepx/gateway-go/internal/jobs"
	"dataprepx/gateway-go/internal/ws"
)

func main() {
	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	store := newStore()

	jobsHandler := jobs.NewHandler(store)
	jobsHandler.OnCreated(func(job jobs.Job) {
		// Stands in for the real Celery task graph (planner Phase 8),
		// which doesn't exist yet. Backgrounded so Submit's response
		// is never delayed by it -- the gateway must not block on
		// compute (CLAUDE.md §5.7).
		go func() {
			if err := jobs.RunFakeWorker(context.Background(), store, job.ID, 500*time.Millisecond); err != nil {
				log.Printf("gateway-go: fake worker for job %s exited: %v", job.ID, err)
			}
		}()
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

// newStore selects the job Store implementation based on JOB_STORE
// (default: postgres via DATABASE_URL; set JOB_STORE=memory for local dev
// or demoing without a running Postgres instance).
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

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok","service":"gateway-go"}`))
}
