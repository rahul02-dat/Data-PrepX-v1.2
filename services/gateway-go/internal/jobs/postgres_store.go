package jobs

import (
	"context"
	"database/sql"
	"errors"
	"sync"

	_ "github.com/lib/pq"
)

type postgresStore struct {
	db *sql.DB

	mu          sync.Mutex
	subscribers map[string][]chan Job
}

// Construct Postgres-backed job store
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

// Persist new job in Postgres
func (s *postgresStore) CreateJob(ctx context.Context, req SubmitRequest) (Job, error) {
	hash, err := hashConfig(req.Config)
	if err != nil {
		return Job{}, err
	}

	const q = `
		INSERT INTO runs (dataset_id, git_sha, config_hash, status)
		VALUES ($1, $2, $3, 'queued')
		RETURNING id, dataset_id, status, config_hash, celery_task_id, created_at, updated_at
	`
	row := s.db.QueryRowContext(ctx, q, req.DatasetID, "unknown", hash)
	return scanJob(row)
}

// Retrieve job by ID from Postgres
func (s *postgresStore) GetJob(ctx context.Context, id string) (Job, error) {
	const q = `
		SELECT id, dataset_id, status, config_hash, celery_task_id, created_at, updated_at
		FROM runs WHERE id = $1
	`
	row := s.db.QueryRowContext(ctx, q, id)
	job, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return Job{}, ErrNotFound
	}
	return job, err
}

// Update job status in Postgres and notify subscribers
func (s *postgresStore) UpdateStatus(ctx context.Context, id string, status Status) (Job, error) {
	const q = `
		UPDATE runs SET status = $2, updated_at = now()
		WHERE id = $1
		RETURNING id, dataset_id, status, config_hash, celery_task_id, created_at, updated_at
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

// Register subscriber for job status transitions
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

// Scan SQL database row into Job struct
func scanJob(row *sql.Row) (Job, error) {
	var job Job
	var datasetID sql.NullString
	var celeryTaskID sql.NullString
	if err := row.Scan(
		&job.ID, &datasetID, &job.Status, &job.ConfigHash,
		&celeryTaskID, &job.CreatedAt, &job.UpdatedAt,
	); err != nil {
		return Job{}, err
	}
	if datasetID.Valid {
		job.DatasetID = &datasetID.String
	}
	if celeryTaskID.Valid {
		job.CeleryTaskID = &celeryTaskID.String
	}
	return job, nil
}
