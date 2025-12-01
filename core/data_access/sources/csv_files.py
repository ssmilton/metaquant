"""CSV/Parquet ingestion helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import duckdb

logger = logging.getLogger(__name__)


def ingest_csv_files(db_path: str | Path, csv_files: Iterable[str], table: str) -> None:
    """Ingest CSV files into DuckDB table using COPY."""
    conn = duckdb.connect(str(db_path))
    for file in csv_files:
        logger.info("Ingesting %s into %s", file, table)
        conn.execute(f"COPY {table} FROM '{file}' (AUTO_DETECT TRUE)")
    conn.close()
