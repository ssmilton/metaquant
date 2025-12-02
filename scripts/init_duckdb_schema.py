"""Create DuckDB schema for MetaQuant."""
from __future__ import annotations

import duckdb

DDL = [
    """
    CREATE TABLE IF NOT EXISTS securities (
        security_id INTEGER PRIMARY KEY,
        ticker VARCHAR,
        exchange VARCHAR,
        asset_type VARCHAR,
        sector VARCHAR,
        industry VARCHAR,
        is_option BOOLEAN,
        underlying_id INTEGER,
        currency VARCHAR,
        active BOOLEAN
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        security_id INTEGER,
        date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        adj_close DOUBLE,
        volume DOUBLE,
        PRIMARY KEY (security_id, date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS options_quotes (
        option_id INTEGER,
        date DATE,
        bid DOUBLE,
        ask DOUBLE,
        last DOUBLE,
        iv DOUBLE,
        delta DOUBLE,
        gamma DOUBLE,
        theta DOUBLE,
        vega DOUBLE,
        open_interest DOUBLE,
        underlying_price DOUBLE,
        PRIMARY KEY (option_id, date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamentals (
        security_id INTEGER,
        report_date DATE,
        metric_name VARCHAR,
        metric_value DOUBLE,
        PRIMARY KEY (security_id, report_date, metric_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS market_features (
        date DATE PRIMARY KEY,
        vix DOUBLE,
        spx_return_1d DOUBLE,
        realized_vol_30d DOUBLE,
        regime_label VARCHAR
    );
    """,
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


def create_schema(db_path: str) -> None:
    con = duckdb.connect(db_path)
    for stmt in DDL:
        con.execute(stmt)
    con.close()


if __name__ == "__main__":
    create_schema("data/metaquant.duckdb")
