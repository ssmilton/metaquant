"""Simple CSV ingestion script."""
from __future__ import annotations

import typer

from core.data_access.sources.csv_files import ingest_csv_files

app = typer.Typer()


@app.command()
def ingest(db_path: str, table: str, files: str) -> None:
    """Ingest comma-separated list of CSV files into DuckDB."""
    paths = [f.strip() for f in files.split(",") if f.strip()]
    ingest_csv_files(db_path, paths, table)


if __name__ == "__main__":
    app()
