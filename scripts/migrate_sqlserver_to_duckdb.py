"""Migrate historical financial data from SQL Server to DuckDB."""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
import pyodbc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQL Server connection string (from .mcp.json)
SQLSERVER_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=stock_screener;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# DuckDB database path
DUCKDB_PATH = "data/metaquant.duckdb"

# Batch size for processing
BATCH_SIZE = 10000


def migrate_tickers_to_securities(sqlserver_conn: pyodbc.Connection, duckdb_conn: duckdb.DuckDBPyConnection) -> None:
    """Migrate tickers from SQL Server to securities table in DuckDB.

    Maps:
    - ticker -> ticker
    - active -> active
    - Creates a sequential security_id
    """
    logger.info("Migrating tickers to securities table...")

    # First, check if securities already has data
    existing_count = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
    if existing_count > 0:
        logger.warning(f"Securities table already has {existing_count} records. Clearing...")
        duckdb_conn.execute("DELETE FROM securities")

    # Fetch all tickers from SQL Server
    query = """
        SELECT
            ticker,
            active
        FROM dbo.tickers
        ORDER BY ticker
    """

    df = pd.read_sql(query, sqlserver_conn)
    logger.info(f"Fetched {len(df)} tickers from SQL Server")

    # Add security_id as sequential integer
    df['security_id'] = range(1, len(df) + 1)

    # Add default values for other columns
    df['exchange'] = None
    df['asset_type'] = 'equity'  # Assuming equities
    df['sector'] = None
    df['industry'] = None
    df['is_option'] = False
    df['underlying_id'] = None
    df['currency'] = 'USD'

    # Reorder columns to match schema
    df = df[[
        'security_id', 'ticker', 'exchange', 'asset_type', 'sector',
        'industry', 'is_option', 'underlying_id', 'currency', 'active'
    ]]

    # Insert into DuckDB
    duckdb_conn.execute("INSERT INTO securities SELECT * FROM df")
    logger.info(f"Inserted {len(df)} securities into DuckDB")

    return df[['security_id', 'ticker']]  # Return mapping for use in next step


def migrate_historical_data_to_daily_prices(
    sqlserver_conn: pyodbc.Connection,
    duckdb_conn: duckdb.DuckDBPyConnection,
    ticker_mapping: pd.DataFrame
) -> None:
    """Migrate historical_data from SQL Server to daily_prices table in DuckDB.

    Maps:
    - ticker -> security_id (via ticker_mapping)
    - date -> date
    - open, high, low, close -> open, high, low, close, adj_close (close = adj_close)
    - volume -> volume
    """
    logger.info("Migrating historical_data to daily_prices table...")

    # First, check if daily_prices already has data
    existing_count = duckdb_conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    if existing_count > 0:
        logger.warning(f"Daily_prices table already has {existing_count} records. Clearing...")
        duckdb_conn.execute("DELETE FROM daily_prices")

    # Get total count
    total_count = pd.read_sql("SELECT COUNT(*) as cnt FROM dbo.historical_data", sqlserver_conn).iloc[0]['cnt']
    logger.info(f"Total historical records to migrate: {total_count:,}")

    # Process in batches
    offset = 0
    total_inserted = 0

    while offset < total_count:
        logger.info(f"Processing batch at offset {offset:,} ({offset/total_count*100:.1f}%)")

        query = f"""
            SELECT
                ticker,
                [date],
                [open],
                [high],
                [low],
                [close],
                volume
            FROM dbo.historical_data
            ORDER BY id
            OFFSET {offset} ROWS
            FETCH NEXT {BATCH_SIZE} ROWS ONLY
        """

        df = pd.read_sql(query, sqlserver_conn)

        if df.empty:
            break

        # Join with ticker_mapping to get security_id
        df = df.merge(ticker_mapping, on='ticker', how='left')

        # Drop rows where security_id is null (ticker not found in mapping)
        before_drop = len(df)
        df = df.dropna(subset=['security_id'])
        after_drop = len(df)
        if before_drop != after_drop:
            logger.warning(f"Dropped {before_drop - after_drop} rows with unknown tickers")

        # Convert security_id to integer
        df['security_id'] = df['security_id'].astype(int)

        # Convert date to date type
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Add adj_close (same as close for now)
        df['adj_close'] = df['close']

        # Select and reorder columns
        df = df[[
            'security_id', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'
        ]]

        # Insert into DuckDB
        duckdb_conn.execute("INSERT INTO daily_prices SELECT * FROM df")
        total_inserted += len(df)

        offset += BATCH_SIZE
        logger.info(f"Inserted {total_inserted:,} records so far...")

    logger.info(f"Completed migration of {total_inserted:,} daily price records")


def migrate_financial_data_to_fundamentals(
    sqlserver_conn: pyodbc.Connection,
    duckdb_conn: duckdb.DuckDBPyConnection,
    ticker_mapping: pd.DataFrame
) -> None:
    """Migrate financial_data from SQL Server to fundamentals table in DuckDB.

    The financial_data table in SQL Server is wide-format with many metric columns.
    The fundamentals table in DuckDB is long-format with (security_id, report_date, metric_name, metric_value).

    We'll unpivot/melt the financial data to fit the long format.
    """
    logger.info("Migrating financial_data to fundamentals table...")

    # First, check if fundamentals already has data
    existing_count = duckdb_conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
    if existing_count > 0:
        logger.warning(f"Fundamentals table already has {existing_count} records. Clearing...")
        duckdb_conn.execute("DELETE FROM fundamentals")

    # Fetch all financial data from SQL Server
    query = """
        SELECT
            ticker,
            asof_date,
            market_cap, pe_ratio, pb_ratio, ev_ebitda, roe, roa,
            profit_margin, operating_margin, current_ratio, debt_to_equity,
            revenue_growth, eps_growth, earnings_growth, fifty_two_week_change,
            price_change_6m, price_change_3m, rsi_score, volatility, beta,
            avg_volume, dollar_volume, fcf_yield
        FROM dbo.financial_data
    """

    df = pd.read_sql(query, sqlserver_conn)
    logger.info(f"Fetched {len(df)} financial records from SQL Server")

    # Join with ticker_mapping to get security_id
    df = df.merge(ticker_mapping, on='ticker', how='left')
    df = df.dropna(subset=['security_id'])
    df['security_id'] = df['security_id'].astype(int)

    # Convert asof_date to date
    df['asof_date'] = pd.to_datetime(df['asof_date']).dt.date

    # Melt the dataframe from wide to long format
    id_vars = ['security_id', 'asof_date']
    value_vars = [
        'market_cap', 'pe_ratio', 'pb_ratio', 'ev_ebitda', 'roe', 'roa',
        'profit_margin', 'operating_margin', 'current_ratio', 'debt_to_equity',
        'revenue_growth', 'eps_growth', 'earnings_growth', 'fifty_two_week_change',
        'price_change_6m', 'price_change_3m', 'rsi_score', 'volatility', 'beta',
        'avg_volume', 'dollar_volume', 'fcf_yield'
    ]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='metric_name',
        value_name='metric_value'
    )

    # Drop rows with null metric values
    df_long = df_long.dropna(subset=['metric_value'])

    # Rename asof_date to report_date
    df_long = df_long.rename(columns={'asof_date': 'report_date'})

    # Reorder columns
    df_long = df_long[['security_id', 'report_date', 'metric_name', 'metric_value']]

    logger.info(f"Unpivoted to {len(df_long)} fundamental metric records")

    # Insert into DuckDB in batches
    batch_size = 50000
    total_inserted = 0

    for i in range(0, len(df_long), batch_size):
        batch = df_long.iloc[i:i+batch_size]
        duckdb_conn.execute("INSERT INTO fundamentals SELECT * FROM batch")
        total_inserted += len(batch)
        logger.info(f"Inserted {total_inserted:,} / {len(df_long):,} fundamental records...")

    logger.info(f"Completed migration of {total_inserted:,} fundamental records")


def main() -> None:
    """Run the migration from SQL Server to DuckDB."""
    logger.info("Starting migration from SQL Server to DuckDB...")

    # Connect to SQL Server
    logger.info("Connecting to SQL Server...")
    sqlserver_conn = pyodbc.connect(SQLSERVER_CONN_STR)

    # Connect to DuckDB
    logger.info(f"Connecting to DuckDB at {DUCKDB_PATH}...")
    duckdb_conn = duckdb.connect(DUCKDB_PATH)

    try:
        # Step 1: Migrate tickers to securities
        ticker_mapping = migrate_tickers_to_securities(sqlserver_conn, duckdb_conn)

        # Step 2: Migrate historical_data to daily_prices
        migrate_historical_data_to_daily_prices(sqlserver_conn, duckdb_conn, ticker_mapping)

        # Step 3: Migrate financial_data to fundamentals
        migrate_financial_data_to_fundamentals(sqlserver_conn, duckdb_conn, ticker_mapping)

        logger.info("Migration completed successfully!")

        # Print summary statistics
        securities_count = duckdb_conn.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        daily_prices_count = duckdb_conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        fundamentals_count = duckdb_conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]

        logger.info(f"\nMigration Summary:")
        logger.info(f"  Securities: {securities_count:,}")
        logger.info(f"  Daily Prices: {daily_prices_count:,}")
        logger.info(f"  Fundamentals: {fundamentals_count:,}")

    finally:
        sqlserver_conn.close()
        duckdb_conn.close()


if __name__ == "__main__":
    main()
