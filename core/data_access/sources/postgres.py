"""Placeholder Postgres ingestion module."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PostgresConnectionInfo:
    host: str
    database: str
    user: str
    password: str
    port: int = 5432


def extract_to_duckdb(connection: PostgresConnectionInfo, table: str, duckdb_path: str) -> None:
    logger.info(
        "Extract from Postgres %s/%s table %s into %s (stub)",
        connection.host,
        connection.database,
        table,
        duckdb_path,
    )
