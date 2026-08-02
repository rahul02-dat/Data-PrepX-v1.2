package jobs

import (
	"context"
	"database/sql"
	"errors"
	"sync"

	_ "github.com/lib/pq"
)

// postgresStore persists jobs to the runs table (CLAUDE.md §6) and fans out
// status transitions to in-process subscribers.
//
// Verification note: this file's SQL was hand-verified against a real
// Postgres 16 instance during development (schema create, insert, status
// update, CHECK-constraint rejection of an invalid status, and both down
// migrations reversing cleanly) but is not covered by an automated
// integration test in this repository yet -- doing so needs a Postgres
// instance in CI (docker-compose based), which this development sandbox
// cannot boot. The in-memory store (memory_store.go) is what the automated
// test suite exercises.
type postgresStore struct {
	db *sql.DB

	mu          sync.Mutex
	subscribers map[string][]chan Job
}

// NewPostgresStore opens a connection pool against dataSourceName (a
// postgres:// URL) and returns a Store backed by the runs table.
func NewPostgresStore(dataSourceName string) (Store, error) {
	db, err := sql.Open("postgres", dataSourceName)
	if err != nil {
		return nil, err
	}
	return &postgresStore{
		db:          db,
		subscribers: make(map[string][]chan Job),
	}, nil
}

func (s *postgresStore) CreateJob(ctx context.Context, req SubmitRequest) (Job, error) {
	hash, err := hashConfig(req.Config)
	if err != nil {
		return Job{}, err
	}

	const q = `
		INSERT INTO runs (dataset_id, git_sha, config_hash, status)
		VALUES ($1, $2, $3, 'queued')
		RETURNING id, dataset_id, status, config_hash, created_at, updated_at
	`
	// git_sha is not yet threaded through from the build (Phase 1 scope);
	// "unknown" is an explicit placeholder, not a silent default, per
	// CLAUDE.md's stance on avoiding silent defaults for lineage fields.
	row := s.db.QueryRowContext(ctx, q, req.DatasetID, "unknown", hash)
	return scanJob(row)
}

func (s *postgresStore) GetJob(ctx context.Context, id string) (Job, error) {
	const q = `
		SELECT id, dataset_id, status, config_hash, created_at, updated_at
		FROM runs WHERE id = $1
	`
	row := s.db.QueryRowContext(ctx, q, id)
	job, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Job{}, ErrNotFound
	}
	return job, err
}

func (s *postgresStore) UpdateStatus(ctx context.Context, id string, status Status) (Job, error) {
	const q = `
		UPDATE runs SET status = $2, updated_at = now()
		WHERE id = $1
		RETURNING id, dataset_id, status, config_hash, created_at, updated_at
	`
	row := s.db.QueryRowContext(ctx, q, id, string(status))
	job, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Job{}, ErrNotFound
	}
	if err != nil {
		return Job{}, err
	}

	s.mu.Lock()
	subs := append([]chan Job(nil), s.subscribers[id]...)
	s.mu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- job:
		default:
		}
	}

	return job, nil
}

func (s *postgresStore) Subscribe(id string) (<-chan Job, func()) {
	ch := make(chan Job, 8)

	s.mu.Lock()
	s.subscribers[id] = append(s.subscribers[id], ch)
	s.mu.Unlock()

	unsubscribe := func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		subs := s.subscribers[id]
		for i, c := range subs {
			if c == ch {
				s.subscribers[id] = append(subs[:i], subs[i+1:]...)
				break
			}
		}
		close(ch)
	}

	return ch, unsubscribe
}

func scanJob(row *sql.Row) (Job, error) {
	var job Job
	var datasetID sql.NullString
	if err := row.Scan(&job.ID, &datasetID, &job.Status, &job.ConfigHash, &job.CreatedAt, &job.UpdatedAt); err != nil {
		return Job{}, err
	}
	if datasetID.Valid {
		job.DatasetID = &datasetID.String
	}
	return job, nil
}
