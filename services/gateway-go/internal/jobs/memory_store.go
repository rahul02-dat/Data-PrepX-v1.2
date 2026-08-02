package jobs

import (
	"context"
	"sync"
	"time"

	"github.com/google/uuid"
)

// memoryStore is an in-memory Store. Safe for concurrent use. It is the
// store exercised by this package's automated tests, and can also be
// selected at runtime (JOB_STORE=memory) for local development without a
// running Postgres instance.
type memoryStore struct {
	mu          sync.Mutex
	jobs        map[string]Job
	subscribers map[string][]chan Job
}

// NewMemoryStore constructs an empty in-memory Store.
func NewMemoryStore() Store {
	return &memoryStore{
		jobs:        make(map[string]Job),
		subscribers: make(map[string][]chan Job),
	}
}

func (s *memoryStore) CreateJob(_ context.Context, req SubmitRequest) (Job, error) {
	hash, err := hashConfig(req.Config)
	if err != nil {
		return Job{}, err
	}

	now := time.Now().UTC()
	job := Job{
		ID:         uuid.NewString(),
		DatasetID:  req.DatasetID,
		Status:     StatusQueued,
		ConfigHash: hash,
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	s.mu.Lock()
	s.jobs[job.ID] = job
	s.mu.Unlock()

	return job, nil
}

func (s *memoryStore) GetJob(_ context.Context, id string) (Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[id]
	if !ok {
		return Job{}, ErrNotFound
	}
	return job, nil
}

func (s *memoryStore) UpdateStatus(_ context.Context, id string, status Status) (Job, error) {
	s.mu.Lock()
	job, ok := s.jobs[id]
	if !ok {
		s.mu.Unlock()
		return Job{}, ErrNotFound
	}
	job.Status = status
	job.UpdatedAt = time.Now().UTC()
	s.jobs[id] = job

	subs := append([]chan Job(nil), s.subscribers[id]...)
	s.mu.Unlock()

	for _, ch := range subs {
		// Non-blocking send: a slow/gone subscriber must never stall the
		// job's own status transition.
		select {
		case ch <- job:
		default:
		}
	}

	return job, nil
}

func (s *memoryStore) Subscribe(id string) (<-chan Job, func()) {
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
