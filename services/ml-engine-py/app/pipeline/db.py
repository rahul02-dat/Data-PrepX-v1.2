"""Thin Postgres connection helper.

Deliberately not an ORM: lineage.py's queries are simple and the schema is the
source of truth (CLAUDE.md §6), so a query builder would add indirection without
value. Uses psycopg (v3) with a plain connection-per-call; a pool is a Phase 8
(async execution) concern once this runs inside real Celery workers, not before.
"""

from __future__ import annotations

import os

import psycopg


def get_connection() -> psycopg.Connection:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set; cannot connect to Postgres.")
    return psycopg.connect(dsn)
