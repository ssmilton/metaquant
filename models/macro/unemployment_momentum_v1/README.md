# Unemployment Momentum V1

This macro model follows the civilian unemployment rate (FRED `UNRATE`) with a smoothed RSI. The latest readings from DuckDB are converted into directional signals so downstream components can treat the unemployment trend like a tradable factor.

## Data dependency
1. Ensure the shared DuckDB schema exists (`python scripts/init_duckdb_schema.py`).
2. Pull the latest UNRATE series into the database:
   ```bash
   python scripts/fetch_fred_unemployment.py --db-path data/metaquant.duckdb --start-date 1990-01-01
   ```

## Parameters
- `db_path` (default `data/metaquant.duckdb`): DuckDB file containing the `fred_series` table.
- `series_id` (default `UNRATE`): FRED series to analyze.
- `smoothing_window` (default `6`): EMA span applied to the unemployment rate before computing RSI.
- `rsi_period` (default `14`): RSI lookback on the smoothed series.
- `upper_threshold` / `lower_threshold` (defaults `55` / `45`): RSI cutoffs to emit long (rising unemployment) or short (falling unemployment) signals.

## Signal semantics
Signals use `security_id=0` with metadata carrying the unemployment rate, smoothed value, and RSI reading for each date. Only dates where RSI breaches the configured thresholds are emitted.
