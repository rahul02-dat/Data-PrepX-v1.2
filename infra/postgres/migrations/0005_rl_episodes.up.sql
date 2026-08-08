CREATE TABLE rl_episodes (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_number INTEGER NOT NULL,
    state_json     JSONB NOT NULL,
    action_json    JSONB NOT NULL,
    reward         DOUBLE PRECISION NOT NULL,
    run_id         UUID REFERENCES runs(id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_rl_episodes_episode_number ON rl_episodes(episode_number);
CREATE INDEX idx_rl_episodes_run_id ON rl_episodes(run_id);