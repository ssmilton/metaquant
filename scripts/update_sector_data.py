"""Update sector and industry data for existing securities from yfinance."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import duckdb
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update sector and industry data for securities from yfinance")
    parser.add_argument("--db", default="data/metaquant.duckdb", help="DuckDB path (default: data/metaquant.duckdb)")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers to update (optional, defaults to all)")
    parser.add_argument("--tickers-file", type=Path, help="Path to newline-delimited tickers list")
    parser.add_argument("--rate-limit-delay", type=float, default=0.2, help="Delay in seconds between ticker fetches (default: 0.2)")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Number of retry attempts for failed requests (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Delay in seconds between retry attempts (default: 5.0)")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of tickers to process before committing (default: 100)")
    parser.add_argument("--only-missing", action="store_true", help="Only update securities missing sector data")
    return parser.parse_args()


def load_tickers(args: argparse.Namespace) -> Optional[List[str]]:
    """Load tickers from command line or file, or return None to process all."""
    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend([t.strip() for t in args.tickers_file.read_text().splitlines() if t.strip() and not t.strip().startswith('#')])
    if args.tickers:
        tickers.extend(args.tickers)
    return sorted({t.upper() for t in tickers}) if tickers else None


def fetch_with_retry(fetch_func, max_retries: int = 3, retry_delay: float = 5.0):
    """Wrapper function to retry API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"All retry attempts failed: {e}")
                return None
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    return None


def get_ticker_sector_info(ticker: str, retry_attempts: int = 3, retry_delay: float = 5.0) -> tuple[Optional[str], Optional[str]]:
    """Fetch sector and industry for a ticker from yfinance."""
    def _fetch():
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
        return info.get("sector"), info.get("industry")

    result = fetch_with_retry(_fetch, max_retries=retry_attempts, retry_delay=retry_delay)
    return result if result else (None, None)


def update_securities(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return

    con = duckdb.connect(str(db_path))

    # Get tickers to update
    specific_tickers = load_tickers(args)

    if specific_tickers:
        placeholders = ','.join(['?'] * len(specific_tickers))
        query = f"SELECT security_id, ticker FROM securities WHERE ticker IN ({placeholders}) AND is_option = FALSE"
        securities = con.execute(query, specific_tickers).fetchall()
        logger.info(f"Found {len(securities)} securities from provided ticker list")
    elif args.only_missing:
        securities = con.execute(
            "SELECT security_id, ticker FROM securities WHERE is_option = FALSE AND (sector IS NULL OR industry IS NULL)"
        ).fetchall()
        logger.info(f"Found {len(securities)} securities missing sector/industry data")
    else:
        securities = con.execute(
            "SELECT security_id, ticker FROM securities WHERE is_option = FALSE"
        ).fetchall()
        logger.info(f"Found {len(securities)} total securities to update")

    if not securities:
        logger.info("No securities to update")
        con.close()
        return

    total = len(securities)
    updated = 0
    skipped = 0
    failed = 0

    logger.info(f"Starting update for {total} securities...")

    for idx, (security_id, ticker) in enumerate(securities, 1):
        try:
            # Rate limiting
            if idx > 1:
                time.sleep(args.rate_limit_delay)

            # Fetch sector info
            sector, industry = get_ticker_sector_info(ticker, args.retry_attempts, args.retry_delay)

            if sector or industry:
                # Update the securities table
                con.execute(
                    """
                    UPDATE securities
                    SET sector = ?, industry = ?
                    WHERE security_id = ?
                    """,
                    [sector, industry, security_id]
                )
                updated += 1
                logger.info(f"[{idx}/{total}] {ticker:8} -> Sector: {sector or 'N/A':20} Industry: {industry or 'N/A'}")
            else:
                skipped += 1
                logger.warning(f"[{idx}/{total}] {ticker:8} -> No sector/industry data available")

            # Commit in batches
            if idx % args.batch_size == 0:
                con.commit()
                logger.info(f"Committed batch at {idx}/{total} tickers")

        except Exception as e:
            failed += 1
            logger.error(f"[{idx}/{total}] {ticker:8} -> Failed: {e}")

    # Final commit
    con.commit()
    con.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"Update complete!")
    logger.info(f"  Total securities: {total}")
    logger.info(f"  Updated: {updated}")
    logger.info(f"  Skipped (no data): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    update_securities(parse_args())
