// Command gateway is the DataPrepX v2 API gateway.
//
// Scope for this service, per CLAUDE.md §2: authentication, job submission
// and polling, and WebSocket status fan-out. It must never contain ML logic
// — if pandas/sklearn-shaped decisions start showing up here, that belongs
// in ml-engine-py instead.
package main

import (
	"log"
	"net/http"
	"os"

	"dataprepx/gateway-go/internal/jobs"
)

func main() {
	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", handleHealthz)
	mux.Handle("/v1/jobs", jobs.NewStubHandler())

	log.Printf("gateway-go listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("gateway-go: server exited: %v", err)
	}
}

func handleHealthz(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok","service":"gateway-go"}`))
}
