"""Fetch civilian unemployment rate data from FRED into DuckDB.

This script pulls the UNRATE series from the St. Louis Fed FRED API,
normalizes the column names, and stores the results in the shared
DuckDB database under the ``fred_series`` table.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

FRED_SERIES_ID = "UNRATE"
FRED_TITLE = "Civilian Unemployment Rate"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(series_id: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """Download a single FRED series as a DataFrame.

    Args:
        series_id: FRED series ID to fetch.
        start_date: Optional ISO date string to bound the query.

    Returns:
        DataFrame with columns [series_id, date, value, title].
    """

    params = ["id=" + series_id]
    if start_date:
        params.append("cosd=" + start_date)
    url = FRED_CSV_URL + "?" + "&".join(params)

    df = pd.read_csv(url)

    # Check what columns we actually got
    if df.empty:
        raise ValueError(f"No data returned for series {series_id}")

    # Handle different possible column names
    # FRED API returns 'observation_date' and the series ID as column names
    if "observation_date" in df.columns:
        df = df.rename(columns={"observation_date": "date"})
    elif "DATE" in df.columns:
        df = df.rename(columns={"DATE": "date"})
    elif "date" not in df.columns:
        # Assume first column is date
        df = df.rename(columns={df.columns[0]: "date"})

    # Handle value column - FRED returns data in a column with the series ID
    if series_id in df.columns:
        df = df.rename(columns={series_id: "value"})
    elif "value" not in df.columns:
        # Assume second column is value
        df = df.rename(columns={df.columns[1]: "value"})

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["series_id"] = series_id
    df["title"] = FRED_TITLE
    return df[["series_id", "date", "value", "title"]]


def write_to_duckdb(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fred_series (
            series_id VARCHAR,
            date DATE,
            value DOUBLE,
            title VARCHAR,
            PRIMARY KEY (series_id, date)
        );
        """
    )
    # Register the DataFrame and use it in the query
    con.register("df_temp", df)
    con.execute(
        "INSERT OR REPLACE INTO fred_series SELECT * FROM df_temp ORDER BY date"
    )
    con.unregister("df_temp")
    con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load FRED UNRATE data into DuckDB")
    parser.add_argument(
        "--db-path",
        default="data/metaquant.duckdb",
        help="Path to the DuckDB database that should receive the series.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional ISO start date to bound the download (e.g. 1990-01-01)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    start_date = args.start_date or date(1948, 1, 1).isoformat()

    df = fetch_fred_series(FRED_SERIES_ID, start_date=start_date)
    write_to_duckdb(df, db_path)
    print(
        f"Wrote {len(df)} rows of {FRED_SERIES_ID} data to {db_path} (dates {df['date'].min().date()} to {df['date'].max().date()})."
    )


if __name__ == "__main__":
    main()
