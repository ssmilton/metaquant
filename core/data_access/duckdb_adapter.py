"""DuckDB data access layer for MetaQuant."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBAdapter:
    def __init__(self, db_path: str | Path, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.connection = duckdb.connect(str(self.db_path), read_only=read_only)
        logger.debug("Connected to DuckDB at %s (read_only=%s)", self.db_path, read_only)

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

    def lookup_security_ids(self, tickers: Iterable[str]) -> List[int]:
        """Look up security IDs from ticker symbols (case-insensitive).

        Args:
            tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "aapl"])

        Returns:
            List of security IDs corresponding to the tickers

        Raises:
            ValueError: If any tickers are not found in the database
        """
        ticker_list = [t.upper() for t in tickers]
        if not ticker_list:
            return []

        query = """
            SELECT ticker, security_id
            FROM securities
            WHERE UPPER(ticker) IN ({placeholders})
            AND active = true
        """.format(placeholders=",".join(["?"] * len(ticker_list)))

        logger.debug("Looking up security IDs for tickers: %s", ticker_list)
        result = self.connection.execute(query, ticker_list).fetch_df()

        if result.empty:
            raise ValueError(f"No securities found for tickers: {ticker_list}")

        # Normalize result tickers to uppercase for matching
        result["ticker_upper"] = result["ticker"].str.upper()
        found_tickers = set(result["ticker_upper"].tolist())
        missing_tickers = set(ticker_list) - found_tickers

        if missing_tickers:
            raise ValueError(
                f"Tickers not found in database: {sorted(missing_tickers)}. "
                f"Available tickers: {sorted(found_tickers)}"
            )

        # Return IDs in the same order as input tickers
        ticker_to_id = dict(zip(result["ticker_upper"], result["security_id"]))
        return [int(ticker_to_id[ticker]) for ticker in ticker_list]

    def insert_model_run(self, row: dict) -> None:
        if self.read_only:
            raise RuntimeError("Cannot insert into read-only database")
        df = pd.DataFrame([row])
        logger.debug("Inserting model run: %s", row)
        self.connection.execute("INSERT INTO model_runs SELECT * FROM df")

    def insert_signals(self, df: pd.DataFrame) -> None:
        if self.read_only:
            raise RuntimeError("Cannot insert into read-only database")
        if df.empty:
            logger.warning("No signals to insert.")
            return
        logger.debug("Inserting %d signals", len(df))
        # Convert timestamp string to datetime and rename meta to meta_json
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'meta' in df.columns:
            df = df.rename(columns={'meta': 'meta_json'})
        # Reorder columns to match table schema
        df = df[['run_id', 'timestamp', 'security_id', 'signal_type', 'strength', 'confidence', 'meta_json']]
        self.connection.execute("INSERT INTO model_signals SELECT * FROM df")

    def insert_trades(self, df: pd.DataFrame) -> None:
        if self.read_only:
            raise RuntimeError("Cannot insert into read-only database")
        if df.empty:
            logger.warning("No trades to insert.")
            return
        logger.debug("Inserting %d trades", len(df))
        # Convert timestamp string to datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Reorder columns to match table schema (excluding trade_id which has DEFAULT)
        df = df[['run_id', 'timestamp', 'security_id', 'side', 'quantity', 'price', 'fees', 'pnl']]
        self.connection.execute("INSERT INTO trades (run_id, timestamp, security_id, side, quantity, price, fees, pnl) SELECT * FROM df")

    def insert_metrics(self, df: pd.DataFrame) -> None:
        if self.read_only:
            raise RuntimeError("Cannot insert into read-only database")
        if df.empty:
            logger.warning("No metrics to insert.")
            return
        logger.debug("Inserting %d metrics", len(df))
        self.connection.execute("INSERT INTO model_metrics SELECT * FROM df")

    def close(self) -> None:  # pragma: no cover - passthrough
        self.connection.close()
