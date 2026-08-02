package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleHealthz(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()

	handleHealthz(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}
	got := rec.Body.String()
	want := `{"status":"ok","service":"gateway-go"}`
	if got != want {
		t.Fatalf("expected body %q, got %q", want, got)
	}
}

func TestNewStore_Memory(t *testing.T) {
	t.Setenv("JOB_STORE", "memory")
	store := newStore()
	if store == nil {
		t.Fatal("expected a non-nil store when JOB_STORE=memory")
	}
}