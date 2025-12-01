"""Placeholder SQL Server ingestion module."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class SQLServerConnectionInfo:
    server: str
    database: str
    user: str
    password: str
    driver: str = "ODBC Driver 17 for SQL Server"


def extract_to_duckdb(connection: SQLServerConnectionInfo, table: str, duckdb_path: str) -> None:
    """Stub function for SQL Server to DuckDB ETL.

    In a production implementation this would leverage pyodbc or similar.
    """
    logger.info(
        "Extract from SQL Server %s/%s table %s into %s (stub)",
        connection.server,
        connection.database,
        table,
        duckdb_path,
    )
