"""Fetch equity data from yfinance into the MetaQuant DuckDB schema."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd
import yfinance as yf

# Reuse the schema helper so running this script is idempotent
from scripts.init_duckdb_schema import create_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest equity prices and fundamentals from yfinance into DuckDB")
    parser.add_argument("tickers", nargs="*", help="Tickers to fetch (e.g., AAPL MSFT)")
    parser.add_argument("--tickers-file", type=Path, help="Path to newline-delimited tickers list")
    parser.add_argument("--db", default="data/metaquant.duckdb", help="DuckDB path (default: data/metaquant.duckdb)")
    parser.add_argument("--start-date", default=(dt.date.today() - dt.timedelta(days=365)).isoformat(), help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=dt.date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Skip writing fundamentals metrics")
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


def get_or_create_security_id(con: duckdb.DuckDBPyConnection, ticker: str, info: Dict) -> int:
    existing = con.execute("SELECT security_id FROM securities WHERE ticker = ?", [ticker]).fetchone()
    if existing:
        return int(existing[0])
    max_id = con.execute("SELECT COALESCE(MAX(security_id), 0) FROM securities").fetchone()[0]
    security_id = int(max_id) + 1
    row = {
        "security_id": security_id,
        "ticker": ticker,
        "exchange": info.get("fullExchangeName"),
        "asset_type": info.get("quoteType", "equity"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "is_option": False,
        "underlying_id": None,
        "currency": info.get("currency", "USD"),
        "active": True,
    }
    df = pd.DataFrame([row])
    con.execute("INSERT INTO securities SELECT * FROM df")
    return security_id


def download_history(tickers: Iterable[str], start: str, end: str) -> Tuple[pd.DataFrame, bool]:
    tickers_list = list(tickers)
    multi = len(tickers_list) > 1
    hist = yf.download(
        tickers=tickers_list,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    return hist, multi


def history_to_rows(hist: pd.DataFrame, ticker: str, security_id: int, multi: bool) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame()
    if multi:
        try:
            df = hist[ticker].reset_index()
        except KeyError:
            return pd.DataFrame()
    else:
        df = hist.reset_index()
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close", "Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]].dropna(subset=["close"])
    df.insert(0, "security_id", security_id)
    return df


def fundamentals_rows(security_id: int, as_of: dt.date, info: Dict) -> pd.DataFrame:
    metrics = {
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "profit_margins": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "free_cash_flow": info.get("freeCashflow"),
        "debt_to_equity": info.get("debtToEquity"),
    }
    rows = [
        {
            "security_id": security_id,
            "report_date": as_of,
            "metric_name": name,
            "metric_value": float(value),
        }
        for name, value in metrics.items()
        if value is not None
    ]
    return pd.DataFrame(rows)


def upsert_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, start: str, end: str) -> None:
    if df.empty:
        return
    security_id = int(df.iloc[0]["security_id"])
    con.execute(
        "DELETE FROM daily_prices WHERE security_id = ? AND date BETWEEN ? AND ?",
        [security_id, start, end],
    )
    con.execute("INSERT INTO daily_prices SELECT * FROM df")


def upsert_fundamentals(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, security_id: int, as_of: str) -> None:
    if df.empty:
        return
    con.execute(
        "DELETE FROM fundamentals WHERE security_id = ? AND report_date = ?",
        [security_id, as_of],
    )
    con.execute("INSERT INTO fundamentals SELECT * FROM df")


def ingest(args: argparse.Namespace) -> None:
    tickers = load_tickers(args)
    if not tickers:
        print("No tickers provided.")
        return

    db_path = Path(args.db)
    con = ensure_schema(db_path)
    as_of = dt.date.fromisoformat(args.end_date)

    hist, multi = download_history(tickers, args.start_date, args.end_date)

    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
        security_id = get_or_create_security_id(con, ticker, info)

        price_df = history_to_rows(hist, ticker, security_id, multi)
        upsert_prices(con, price_df, args.start_date, args.end_date)

        if not args.skip_fundamentals:
            fundamentals_df = fundamentals_rows(security_id, as_of, info)
            upsert_fundamentals(con, fundamentals_df, security_id, args.end_date)

    con.close()
    print(f"Ingested {len(tickers)} tickers into {db_path}")


if __name__ == "__main__":
    ingest(parse_args())
