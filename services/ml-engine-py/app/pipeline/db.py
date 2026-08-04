from __future__ import annotations

import os

import psycopg


# Connect to Postgres database
def get_connection() -> psycopg.Connection:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set; cannot connect to Postgres.")
    return psycopg.connect(dsn)
