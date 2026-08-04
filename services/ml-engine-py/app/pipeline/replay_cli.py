from __future__ import annotations

import json
import sys

from app.pipeline.db import get_connection
from app.pipeline.lineage import LineageRecorder


# CLI entrypoint for run replay
def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.pipeline.replay_cli <run_id>", file=sys.stderr)
        return 2

    run_id = argv[1]
    conn = get_connection()
    try:
        recorder = LineageRecorder(conn)
        try:
            record = recorder.replay_run(run_id)
        except LookupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        payload = {
            "run_id": record.run_id,
            "dataset_id": record.dataset_id,
            "git_sha": record.git_sha,
            "config_hash": record.config_hash,
            "run_key": record.run_key,
            "steps": [
                {
                    "id": step.id,
                    "step_type": step.step_type,
                    "input_hash": step.input_hash,
                    "output_hash": step.output_hash,
                    "params": step.params,
                    "seed": step.seed,
                }
                for step in record.steps
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
