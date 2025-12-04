"""Create model results schema for model-specific DuckDB files."""
from __future__ import annotations

import duckdb
from pathlib import Path

# Schema for model results tables only
DDL = [
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id UUID PRIMARY KEY,
        model_id VARCHAR,
        model_version VARCHAR,
        experiment_id VARCHAR,
        start_date DATE,
        end_date DATE,
        params_json JSON,
        created_at TIMESTAMP,
        notes VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_metrics (
        run_id UUID,
        metric_name VARCHAR,
        scope VARCHAR,
        metric_value DOUBLE,
        PRIMARY KEY (run_id, metric_name, scope)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_signals (
        run_id UUID,
        timestamp TIMESTAMP,
        security_id INTEGER,
        signal_type VARCHAR,
        strength DOUBLE,
        confidence DOUBLE,
        meta_json JSON
    );
    """,
    """
    CREATE SEQUENCE IF NOT EXISTS trade_id_seq START 1;
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        run_id UUID,
        trade_id BIGINT DEFAULT nextval('trade_id_seq'),
        timestamp TIMESTAMP,
        security_id INTEGER,
        side VARCHAR,
        quantity DOUBLE,
        price DOUBLE,
        fees DOUBLE,
        pnl DOUBLE,
        PRIMARY KEY (run_id, trade_id)
    );
    """,
]


def create_model_results_schema(db_path: str | Path) -> None:
    """Initialize schema for model results in a model-specific database.

    Args:
        db_path: Path to the model-specific DuckDB file
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    for stmt in DDL:
        con.execute(stmt)
    con.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python init_model_results_schema.py <db_path>")
        sys.exit(1)
    create_model_results_schema(sys.argv[1])
