package jobs

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestStubHandler_GET(t *testing.T) {
	h := NewStubHandler()
	req := httptest.NewRequest(http.MethodGet, "/v1/jobs", nil)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("expected application/json content type, got %q", ct)
	}
}

func TestStubHandler_MethodNotAllowed(t *testing.T) {
	h := NewStubHandler()
	req := httptest.NewRequest(http.MethodDelete, "/v1/jobs", nil)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", rec.Code)
	}
}
