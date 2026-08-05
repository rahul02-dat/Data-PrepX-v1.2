from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import pandas as pd


# Compute SHA256 hex hash of raw bytes
def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Compute stable SHA256 hash of configuration dictionary
def hash_config(config: dict[str, Any] | None) -> str:
    config = config or {}
    ordered: list[Any] = []
    for key in sorted(config.keys()):
        ordered.append(key)
        ordered.append(config[key])
    encoded = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# Compute content hash of dataframe data and schema
def hash_dataframe(df: pd.DataFrame) -> str:
    schema_repr = json.dumps(
        [[str(c), str(dt)] for c, dt in zip(df.columns, df.dtypes)],
        separators=(",", ":"),
    )
    row_hashes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    hasher = hashlib.sha256()
    hasher.update(schema_repr.encode("utf-8"))
    hasher.update(row_hashes)
    return "sha256:" + hasher.hexdigest()


# Compute deterministic run identity key
def compute_run_key(dataset_content_hash: str, config_hash: str, git_sha: str) -> str:
    encoded = "|".join([dataset_content_hash, config_hash, git_sha]).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# Compute a transform_code_hash from a function's own source. This is what lets a lineage entry
# distinguish "same params, different transform implementation" -- if imputation.py's logic
# changes, this hash changes even though params_json may not, so a stale replay is detectable.
def hash_source(fn: Callable) -> str:
    return "sha256:" + hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
