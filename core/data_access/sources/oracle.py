"""Placeholder Oracle ingestion module."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OracleConnectionInfo:
    dsn: str
    user: str
    password: str


def extract_to_duckdb(connection: OracleConnectionInfo, table: str, duckdb_path: str) -> None:
    logger.info(
        "Extract from Oracle %s table %s into %s (stub)",
        connection.dsn,
        table,
        duckdb_path,
    )
