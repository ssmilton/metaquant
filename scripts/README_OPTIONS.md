# YFinance Options Data Fetcher

## Overview

`fetch_yfinance_options.py` is a tool for fetching historical options data from Yahoo Finance and populating it into DuckDB or MS SQL databases using the MetaQuant schema.

## Features

- Fetch options chains for multiple underlying tickers
- Filter by expiration date or date ranges
- Support for calls, puts, or both
- Automatic creation of option securities in the database
- Idempotent operation (can be run multiple times safely)
- Batch processing with logging

## Database Schema

The script populates two main tables:

### `securities` Table
Stores option contract metadata:
- `security_id`: Unique identifier
- `ticker`: Option contract symbol (e.g., AAPL240119C00180000)
- `is_option`: TRUE for options
- `underlying_id`: Links to the underlying equity security_id
- `asset_type`: "option"

### `options_quotes` Table
Stores options pricing and Greeks data:
- `option_id`: References securities.security_id
- `date`: Quote date
- `bid`, `ask`, `last`: Price data
- `iv`: Implied volatility
- `delta`, `gamma`, `theta`, `vega`: Greeks (when available)
- `open_interest`: Number of open contracts
- `underlying_price`: Price of underlying at quote time

## Installation

Ensure the required dependencies are installed:

```bash
pip install yfinance duckdb pandas
```

Or install the project dependencies:

```bash
pip install -e .
```

## Usage

### Basic Usage

Fetch options for a single ticker:

```bash
python scripts/fetch_yfinance_options.py AAPL
```

### Multiple Tickers

```bash
python scripts/fetch_yfinance_options.py AAPL MSFT TSLA
```

### Using a Tickers File

```bash
python scripts/fetch_yfinance_options.py --tickers-file tickers.txt
```

Where `tickers.txt` contains one ticker per line:
```
AAPL
MSFT
TSLA
SPY
```

### Filter by Expiration Date

Fetch options for a specific expiration:

```bash
python scripts/fetch_yfinance_options.py AAPL --expiration-date 2024-03-15
```

### Filter by Days to Expiry

Fetch only near-term options (0-30 days to expiry):

```bash
python scripts/fetch_yfinance_options.py AAPL --days-to-expiry-min 0 --days-to-expiry-max 30
```

Fetch only longer-term options (60+ days):

```bash
python scripts/fetch_yfinance_options.py AAPL --days-to-expiry-min 60
```

### Option Type Selection

Fetch only calls:

```bash
python scripts/fetch_yfinance_options.py AAPL --option-type calls
```

Fetch only puts:

```bash
python scripts/fetch_yfinance_options.py AAPL --option-type puts
```

Fetch both (default):

```bash
python scripts/fetch_yfinance_options.py AAPL --option-type both
```

### Custom Database Path

```bash
python scripts/fetch_yfinance_options.py AAPL --db /path/to/custom.duckdb
```

## Complete Example

Fetch all calls and puts for SPY with 7-60 days to expiry:

```bash
python scripts/fetch_yfinance_options.py SPY \
  --option-type both \
  --days-to-expiry-min 7 \
  --days-to-expiry-max 60 \
  --db data/metaquant.duckdb
```

## Option Symbol Format

The script parses yfinance option symbols in the format:

```
TICKER + YYMMDD + C/P + 8-digit strike
```

Example: `AAPL240119C00180000`
- Ticker: AAPL
- Expiration: 2024-01-19
- Type: Call (C)
- Strike: $180.00

## Data Updates

The script uses an upsert strategy:
- Existing option quotes for the same date are deleted before insertion
- New data is inserted
- This allows you to re-run the script to update data

## Querying the Data

### Get all options for a ticker

```sql
SELECT
    s.ticker as option_symbol,
    s.underlying_id,
    oq.date,
    oq.bid,
    oq.ask,
    oq.iv,
    oq.open_interest
FROM securities s
JOIN options_quotes oq ON s.security_id = oq.option_id
WHERE s.is_option = TRUE
  AND s.underlying_id = (SELECT security_id FROM securities WHERE ticker = 'AAPL')
ORDER BY oq.date DESC;
```

### Get implied volatility surface

```sql
SELECT
    s.ticker,
    oq.date,
    oq.iv,
    oq.delta,
    oq.underlying_price
FROM securities s
JOIN options_quotes oq ON s.security_id = oq.option_id
WHERE s.is_option = TRUE
  AND s.underlying_id = (SELECT security_id FROM securities WHERE ticker = 'SPY')
  AND oq.date >= '2024-01-01'
ORDER BY oq.date, s.ticker;
```

## Limitations

1. **Greeks**: Yahoo Finance does not consistently provide Greeks (delta, gamma, theta, vega) in their API. These fields may be NULL.

2. **Historical Data**: The script fetches current market data. For true historical backtesting, you would need to run this script daily to build a historical database.

3. **Rate Limiting**: Yahoo Finance may rate-limit requests. For large ticker lists, consider adding delays or running in batches.

4. **Data Quality**: Yahoo Finance data is provided as-is and may have gaps or inaccuracies.

## Integration with MS SQL

While this script is designed for DuckDB, the schema is compatible with MS SQL Server. To use with MS SQL:

1. Create equivalent tables in MS SQL using the schema from `init_duckdb_schema.py`
2. Modify the database connection in the script to use `pyodbc` instead of `duckdb`
3. Adjust the upsert logic for MS SQL syntax (use `MERGE` statements or explicit DELETE/INSERT)

Example MS SQL schema:

```sql
CREATE TABLE securities (
    security_id INT PRIMARY KEY,
    ticker VARCHAR(100),
    exchange VARCHAR(50),
    asset_type VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    is_option BIT,
    underlying_id INT,
    currency VARCHAR(10),
    active BIT
);

CREATE TABLE options_quotes (
    option_id INT,
    date DATE,
    bid FLOAT,
    ask FLOAT,
    last FLOAT,
    iv FLOAT,
    delta FLOAT,
    gamma FLOAT,
    theta FLOAT,
    vega FLOAT,
    open_interest FLOAT,
    underlying_price FLOAT,
    PRIMARY KEY (option_id, date)
);
```

## Troubleshooting

### "No options available"

Some tickers may not have listed options. Verify the ticker has options on Yahoo Finance website.

### "Invalid option symbol format"

The option symbol parser expects standard yfinance format. If Yahoo Finance changes their format, the `parse_option_symbol()` function may need updating.

### Memory Issues

For tickers with many expirations (like SPY), consider:
- Using `--days-to-expiry-max` to limit the date range
- Processing one expiration at a time with `--expiration-date`
- Reducing the ticker list size

## Future Enhancements

Potential improvements for this tool:

1. **Greek Calculation**: Calculate Greeks using Black-Scholes if not provided by yfinance
2. **Historical Snapshots**: Add support for scheduling regular snapshots to build historical database
3. **Parallel Processing**: Add multiprocessing for faster fetching of multiple tickers
4. **Data Validation**: Add checks for suspicious data (bid > ask, negative IV, etc.)
5. **MS SQL Support**: Native MS SQL adapter with proper connection pooling
6. **Delta Hedging**: Calculate delta-hedged positions for underlying securities

## Related Scripts

- `fetch_yfinance_equities.py`: Fetch equity price and fundamental data
- `init_duckdb_schema.py`: Initialize the database schema
- `migrate_sqlserver_to_duckdb.py`: Migrate data between databases

## Support

For issues or questions, refer to the main MetaQuant documentation or check the yfinance documentation at: https://github.com/ranaroussi/yfinance
