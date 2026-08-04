package jobs

import (
	"context"
	"sync"
	"time"

	"github.com/google/uuid"
)

type memoryStore struct {
	mu          sync.Mutex
	jobs        map[string]Job
	subscribers map[string][]chan Job
}

// Construct empty in-memory job store
func NewMemoryStore() Store {
	return &memoryStore{
		jobs:        make(map[string]Job),
		subscribers: make(map[string][]chan Job),
	}
}

// Persist new job in memory
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

// Retrieve job by ID from memory
func (s *memoryStore) GetJob(_ context.Context, id string) (Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[id]
	if !ok {
		return Job{}, ErrNotFound
	}
	return job, nil
}

// Update job status in memory and notify subscribers
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
		select {
		case ch <- job:
		default:
		}
	}

	return job, nil
}

// Register subscriber for job status transitions
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
