"""Content-addressed hashing for lineage (CLAUDE.md §5.3).

`run_id` (well, `run_key` -- see docs/adr/0003-run-id-determinism.md) is composed of
hash(dataset) + hash(config) + git_sha. This module is the single place that
computes any of those hashes so the scheme cannot drift between callers.

`hash_config` intentionally mirrors services/gateway-go/internal/jobs/hash.go's
`hashConfig`: sort keys, build an ordered [key, value, key, value, ...] list, marshal
to compact JSON, sha256, prefix with "sha256:". This lets ml-engine-py and gateway-go
agree on a config_hash for the same config object.

Known limitation (documented, not silently ignored): Go's encoding/json escapes
`<`, `>`, and `&` in strings by default; Python's json module does not. For config
values containing those characters, the two services' hashes will diverge. No config
value in the current schema uses them, so this is a latent risk rather than an active
bug -- flagged here and in docs/adr/0003-run-id-determinism.md rather than assumed
fixed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def hash_bytes(data: bytes) -> str:
    """sha256 of raw bytes, hex-encoded, unprefixed."""
    return hashlib.sha256(data).hexdigest()


def hash_config(config: dict[str, Any] | None) -> str:
    """sha256 over a config mapping, order-independent, matching gateway-go's hashConfig."""
    config = config or {}
    ordered: list[Any] = []
    for key in sorted(config.keys()):
        ordered.append(key)
        ordered.append(config[key])
    encoded = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Deterministic content hash of a dataframe's data and column schema.

    Uses pandas' own row-wise hashing (hash_pandas_object) rather than a raw bytes
    dump so that hashing is robust to things like index dtype quirks; column order
    and dtypes are folded in explicitly since two dataframes with the same values
    but reordered/retyped columns are not the same dataset for lineage purposes.
    """
    schema_repr = json.dumps(
        [[str(c), str(dt)] for c, dt in zip(df.columns, df.dtypes)],
        separators=(",", ":"),
    )
    row_hashes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    hasher = hashlib.sha256()
    hasher.update(schema_repr.encode("utf-8"))
    hasher.update(row_hashes)
    return "sha256:" + hasher.hexdigest()


def compute_run_key(dataset_content_hash: str, config_hash: str, git_sha: str) -> str:
    """The deterministic run identity described in CLAUDE.md §6 / ADR 0003.

    Same three inputs always produce the same run_key -- this is what
    get_or_create_run (lineage.py) uses to make run creation idempotent.
    """
    encoded = "|".join([dataset_content_hash, config_hash, git_sha]).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
