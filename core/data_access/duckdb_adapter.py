"""DuckDB data access layer for MetaQuant."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBAdapter:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.connection = duckdb.connect(str(self.db_path))
        logger.debug("Connected to DuckDB at %s", self.db_path)

    def fetch_daily_prices(self, security_ids: Iterable[int], start_date: str, end_date: str) -> pd.DataFrame:
        ids = list(security_ids)
        if not ids:
            return pd.DataFrame()
        query = """
            SELECT *
            FROM daily_prices
            WHERE security_id IN ({placeholders})
              AND date BETWEEN ? AND ?
            ORDER BY security_id, date
        """.format(placeholders=",".join(["?"] * len(ids)))
        params = [*ids, start_date, end_date]
        logger.debug("Fetching prices with params %s", params)
        return self.connection.execute(query, params).fetch_df()

    def fetch_market_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        query = """
            SELECT *
            FROM market_features
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """
        logger.debug("Fetching market features between %s and %s", start_date, end_date)
        return self.connection.execute(query, [start_date, end_date]).fetch_df()

    def insert_model_run(self, row: dict) -> None:
        df = pd.DataFrame([row])
        logger.debug("Inserting model run: %s", row)
        self.connection.execute("INSERT INTO model_runs SELECT * FROM df")

    def insert_signals(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("No signals to insert.")
            return
        logger.debug("Inserting %d signals", len(df))
        self.connection.execute("INSERT INTO model_signals SELECT * FROM df")

    def insert_trades(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("No trades to insert.")
            return
        logger.debug("Inserting %d trades", len(df))
        self.connection.execute("INSERT INTO trades SELECT * FROM df")

    def insert_metrics(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("No metrics to insert.")
            return
        logger.debug("Inserting %d metrics", len(df))
        self.connection.execute("INSERT INTO model_metrics SELECT * FROM df")

    def close(self) -> None:  # pragma: no cover - passthrough
        self.connection.close()
