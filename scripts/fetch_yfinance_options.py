"""Fetch historical options data from yfinance into the MetaQuant DuckDB schema."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time

import duckdb
import pandas as pd
import yfinance as yf

from scripts.init_duckdb_schema import create_schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest historical options data from yfinance into DuckDB")
    parser.add_argument("tickers", nargs="*", help="Underlying tickers to fetch options for (e.g., AAPL MSFT)")
    parser.add_argument("--tickers-file", type=Path, help="Path to newline-delimited tickers list")
    parser.add_argument("--db", default="data/metaquant.duckdb", help="DuckDB path (default: data/metaquant.duckdb)")
    parser.add_argument("--expiration-date", help="Specific option expiration date (YYYY-MM-DD). If not provided, fetches all available expirations")
    parser.add_argument("--days-to-expiry-min", type=int, default=0, help="Minimum days to expiration (default: 0)")
    parser.add_argument("--days-to-expiry-max", type=int, help="Maximum days to expiration (default: no limit)")
    parser.add_argument("--option-type", choices=["calls", "puts", "both"], default="both", help="Type of options to fetch")
    parser.add_argument("--rate-limit-delay", type=float, default=0.5, help="Delay in seconds between API requests (default: 0.5)")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Number of retry attempts for failed requests (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Delay in seconds between retry attempts (default: 5.0)")
    return parser.parse_args()


def load_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend([t.strip() for t in args.tickers_file.read_text().splitlines() if t.strip()])
    tickers.extend(args.tickers)
    return sorted({t.upper() for t in tickers})


def ensure_schema(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_schema(str(db_path))
    return duckdb.connect(str(db_path))


def get_or_create_security_id(con: duckdb.DuckDBPyConnection, ticker: str, is_option: bool = False, underlying_id: Optional[int] = None) -> int:
    """Get or create a security_id for the given ticker."""
    existing = con.execute("SELECT security_id FROM securities WHERE ticker = ?", [ticker]).fetchone()
    if existing:
        return int(existing[0])

    max_id = con.execute("SELECT COALESCE(MAX(security_id), 0) FROM securities").fetchone()[0]
    security_id = int(max_id) + 1

    row = {
        "security_id": security_id,
        "ticker": ticker,
        "exchange": None,
        "asset_type": "option" if is_option else "equity",
        "sector": None,
        "industry": None,
        "is_option": is_option,
        "underlying_id": underlying_id,
        "currency": "USD",
        "active": True,
    }
    df = pd.DataFrame([row])
    con.execute("INSERT INTO securities SELECT * FROM df")
    return security_id


def parse_option_symbol(option_symbol: str) -> Tuple[str, dt.date, str, float]:
    """Parse yfinance option symbol into components.

    Format: TICKER{6-char-date}{C|P}{8-digit-strike}
    Example: AAPL240119C00180000 -> ('AAPL', 2024-01-19, 'C', 180.0)
    """
    # Find the position where date starts (after ticker)
    # Option symbols have a standard format after the ticker
    import re

    # Match pattern: TICKER + YYMMDD + C/P + Strike (padded to 8 digits with 3 decimal places)
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', option_symbol)
    if not match:
        raise ValueError(f"Invalid option symbol format: {option_symbol}")

    ticker, date_str, option_type, strike_str = match.groups()

    # Parse date (YYMMDD)
    exp_date = dt.datetime.strptime(date_str, "%y%m%d").date()

    # Parse strike (8 digits with implied 3 decimal places)
    strike = float(strike_str) / 1000.0

    return ticker, exp_date, option_type, strike


def filter_expirations(
    expirations: List[dt.date],
    target_date: Optional[dt.date] = None,
    min_days: int = 0,
    max_days: Optional[int] = None,
    reference_date: Optional[dt.date] = None
) -> List[dt.date]:
    """Filter expiration dates based on criteria."""
    if target_date:
        return [target_date] if target_date in expirations else []

    if reference_date is None:
        reference_date = dt.date.today()

    filtered = []
    for exp_date in expirations:
        days_to_expiry = (exp_date - reference_date).days
        if days_to_expiry < min_days:
            continue
        if max_days is not None and days_to_expiry > max_days:
            continue
        filtered.append(exp_date)

    return filtered


def fetch_with_retry(fetch_func, max_retries: int = 3, retry_delay: float = 5.0):
    """Wrapper function to retry API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    return None


def fetch_options_chain(ticker: str, expiration: dt.date, option_type: str, retry_attempts: int = 3, retry_delay: float = 5.0) -> pd.DataFrame:
    """Fetch options chain for a specific expiration date with retry logic."""
    ticker_obj = yf.Ticker(ticker)

    def _fetch():
        opt = ticker_obj.option_chain(expiration.strftime("%Y-%m-%d"))

        if option_type == "calls":
            return opt.calls
        elif option_type == "puts":
            return opt.puts
        else:  # both
            return pd.concat([opt.calls, opt.puts], ignore_index=True)

    try:
        df = fetch_with_retry(_fetch, max_retries=retry_attempts, retry_delay=retry_delay)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to fetch options for {ticker} expiring {expiration} after {retry_attempts} attempts: {e}")
        return pd.DataFrame()


def process_options_data(
    con: duckdb.DuckDBPyConnection,
    underlying_ticker: str,
    underlying_id: int,
    options_df: pd.DataFrame,
    fetch_date: dt.date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process options data into securities and options_quotes format.

    Returns:
        Tuple of (securities_df, quotes_df)
    """
    if options_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    securities_rows = []
    quotes_rows = []

    for _, row in options_df.iterrows():
        contract_symbol = row.get("contractSymbol")
        if not contract_symbol:
            continue

        try:
            ticker, exp_date, opt_type, strike = parse_option_symbol(contract_symbol)
        except ValueError as e:
            logger.warning(f"Skipping invalid contract symbol: {e}")
            continue

        # Create or get option security_id
        option_id = get_or_create_security_id(
            con=con,
            ticker=contract_symbol,
            is_option=True,
            underlying_id=underlying_id
        )

        # Build quotes row
        quote = {
            "option_id": option_id,
            "date": fetch_date,
            "bid": row.get("bid"),
            "ask": row.get("ask"),
            "last": row.get("lastPrice"),
            "iv": row.get("impliedVolatility"),
            "delta": None,  # yfinance doesn't provide greeks by default
            "gamma": None,
            "theta": None,
            "vega": None,
            "open_interest": row.get("openInterest"),
            "underlying_price": row.get("lastPrice"),  # Can be improved if available
        }

        quotes_rows.append(quote)

    quotes_df = pd.DataFrame(quotes_rows) if quotes_rows else pd.DataFrame()

    return quotes_df


def upsert_options_quotes(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, fetch_date: dt.date) -> None:
    """Upsert options quotes into the database."""
    if df.empty:
        return

    # Get unique option_ids from this batch
    option_ids = df["option_id"].unique().tolist()

    # Delete existing records for these options on this date
    if option_ids:
        placeholders = ",".join(["?"] * len(option_ids))
        con.execute(
            f"DELETE FROM options_quotes WHERE date = ? AND option_id IN ({placeholders})",
            [fetch_date] + option_ids
        )

    # Insert new records
    con.execute("INSERT INTO options_quotes SELECT * FROM df")
    logger.info(f"  Inserted {len(df)} option quotes")


def ingest_options(args: argparse.Namespace) -> None:
    tickers = load_tickers(args)
    if not tickers:
        logger.error("No tickers provided.")
        return

    db_path = Path(args.db)
    con = ensure_schema(db_path)
    fetch_date = dt.date.today()

    target_expiration = dt.date.fromisoformat(args.expiration_date) if args.expiration_date else None

    total_options = 0

    for idx, ticker in enumerate(tickers, 1):
        logger.info(f"Processing options for {ticker} ({idx}/{len(tickers)})...")

        # Rate limiting delay between tickers
        if idx > 1:
            time.sleep(args.rate_limit_delay)

        # Ensure underlying security exists
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
            underlying_id = get_or_create_security_id(con, ticker, is_option=False)
        except Exception as e:
            logger.error(f"Failed to get ticker info for {ticker}: {e}")
            continue

        # Get available expirations
        try:
            expirations = [dt.datetime.strptime(exp, "%Y-%m-%d").date() for exp in ticker_obj.options]
        except Exception as e:
            logger.error(f"Failed to get expiration dates for {ticker}: {e}")
            continue

        if not expirations:
            logger.warning(f"No options available for {ticker}")
            continue

        # Filter expirations
        filtered_expirations = filter_expirations(
            expirations=expirations,
            target_date=target_expiration,
            min_days=args.days_to_expiry_min,
            max_days=args.days_to_expiry_max,
            reference_date=fetch_date
        )

        if not filtered_expirations:
            logger.warning(f"No expirations match the criteria for {ticker}")
            continue

        logger.info(f"  Found {len(filtered_expirations)} expiration dates to process")

        # Fetch and process each expiration
        for expiration in filtered_expirations:
            days_to_exp = (expiration - fetch_date).days
            logger.info(f"  Fetching options expiring {expiration} (DTE: {days_to_exp})...")

            # Rate limiting between expiration fetches
            time.sleep(args.rate_limit_delay)

            options_df = fetch_options_chain(
                ticker,
                expiration,
                args.option_type,
                retry_attempts=args.retry_attempts,
                retry_delay=args.retry_delay
            )

            if options_df.empty:
                logger.warning(f"  No data for expiration {expiration}")
                continue

            logger.info(f"  Retrieved {len(options_df)} option contracts")

            # Process and insert
            quotes_df = process_options_data(
                con=con,
                underlying_ticker=ticker,
                underlying_id=underlying_id,
                options_df=options_df,
                fetch_date=fetch_date
            )

            upsert_options_quotes(con, quotes_df, fetch_date)
            total_options += len(quotes_df)

    con.close()
    logger.info(f"\nCompleted! Ingested {total_options} option quotes for {len(tickers)} underlying tickers into {db_path}")


if __name__ == "__main__":
    ingest_options(parse_args())
