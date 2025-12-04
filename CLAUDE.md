# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetaQuant is a modular, language-agnostic quantitative research and stock screening framework. It centralizes market data in DuckDB, executes external models via a JSON process contract, and provides backtesting and evaluation capabilities. Models can be written in any language and are executed either on the host or in Docker containers.

## Development Setup

Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Common Commands

### Initialize Database Schema
```bash
quantfw init-db --config config/base.yaml
```

Creates the DuckDB schema at the path specified in `config/base.yaml` (default: `data/metaquant.duckdb`).

### Ingest Market Data
```bash
# Fetch equity prices from Yahoo Finance
python scripts/fetch_yfinance_equities.py AAPL MSFT GOOGL --start-date 2020-01-01 --end-date 2023-12-31

# Fetch historical options data
python scripts/fetch_yfinance_options.py AAPL --start-date 2020-01-01 --end-date 2023-12-31

# Or use a tickers file
python scripts/fetch_yfinance_equities.py --tickers-file tickers.txt
```

### List Available Models
```bash
quantfw list-models
```

Discovers all `model.json` manifests under the `models/` directory.

### Run a Backtest
```bash
# Basic backtest
quantfw backtest \
  --model-id sector_momentum_v1 \
  --security-ids 1,2,3 \
  --start-date 2020-01-01 \
  --end-date 2020-12-31 \
  --params '{"lookback_days":30}'

# Using a parameters file (recommended for Windows/PowerShell)
quantfw backtest \
  --model-id sector_momentum_v1 \
  --security-ids 1,2,3 \
  --start-date 2020-01-01 \
  --end-date 2020-12-31 \
  --params-file params.json

# Specify execution node
quantfw backtest --model-id my_model --node local-linux-dev ...
quantfw backtest --model-id my_model --node-tags dev,gpu ...
```

### Run Tests
```bash
uv run pytest
```

## Architecture

### DuckDB-First Storage

MetaQuant uses a read/write separation pattern to avoid DuckDB concurrency issues (DuckDB allows only one writer at a time):

**Common read-only database** (`data/metaquant.duckdb`):
- All models read market data from this shared database
- Contains: `securities`, `daily_prices`, `options_quotes`, `fundamentals`, `market_features`
- ETL helpers normalize data from SQL Server, Oracle, Postgres, CSV/Parquet, or Yahoo Finance into this schema
- Models open this database in read-only mode

**Model-specific write databases**:
- Each model run writes results to its own local DuckDB file (e.g., `data/<model_id>_<run_id>.duckdb`)
- Contains: `model_runs`, `model_signals`, `trades`, `model_metrics`
- Models write their results, status, and checkpoints to these isolated databases
- Eliminates concurrency contention since each process owns its output database

This design allows parallel model execution without database locking conflicts.

### Language-Agnostic Model Contract

Models are external executables that communicate via JSON over stdin/stdout. Each model has a `model.json` manifest declaring:
- `model_id`, `version`, `description`, `author`
- `runtime` (`host` or `docker`)
- `platform` (`linux/amd64`, `windows/amd64`, or `any`)
- For `host` runtime: `entrypoint` (command) and optional `dependencies` (e.g., `requirements_file`)
- For `docker` runtime: `docker_image` and `docker_entrypoint`

**Input payload structure:**
```json
{
  "mode": "backtest",
  "model_id": "sector_momentum_v1",
  "run_id": "<uuid>",
  "universe": [{"security_id": 101}],
  "data": {"prices": [{"security_id": 101, "dates": [...], "close": [...]}]},
  "parameters": {"lookback_days": 30},
  "time_range": {"start": "2020-01-01", "end": "2020-12-31"}
}
```

**Output payload structure:**
```json
{
  "model_id": "sector_momentum_v1",
  "run_id": "<uuid>",
  "signals": [
    {
      "timestamp": "2020-01-02",
      "security_id": 101,
      "signal_type": "long",
      "strength": 1.0,
      "confidence": 0.8,
      "meta": {}
    }
  ]
}
```

See `core/models/base_contract.py` for Pydantic schemas.

### Runtime and Execution Nodes

Models declare a `runtime` and `platform`. Execution nodes (defined in `config/base.yaml`) declare their capabilities:
```yaml
nodes:
  - name: "local-linux-dev"
    platform: "linux/amd64"
    runtimes: ["host", "docker"]
    tags: ["dev", "default"]
```

The backtest engine (`core/backtest/engine.py`) selects a compatible node automatically or you can specify `--node` or `--node-tags`.

**Host runtime models:**
- Each model gets an isolated virtual environment in `.envs/<model_id>-<version>/`
- The runner (`core/models/runner.py:HostModelRunner`) uses `uv` to create the venv and install dependencies from `requirements_file`
- Set `UV_VENV_CLEAR=1` environment variable to force venv recreation (useful for OneDrive/locked file issues on Windows)
- The `entrypoint` command is executed in the model directory with the venv's Python

**Docker runtime models:**
- Executed in containers using the specified `docker_image` and `docker_entrypoint`
- Docker config in `config/base.yaml` controls memory, CPU limits, mounts, and timeout
- Platform must match the node's platform (cannot be `any`)

### Model Discovery and Registry

`core/registry/discovery.py:ModelRegistry` scans `models/` for all `model.json` files. Models are organized as:
```
models/
  <author>/
    <model_name>/
      model.json
      run_model.py (or any entrypoint)
      requirements.txt (optional)
```

The manifest schema is validated by `core/registry/schema.py:ModelManifest`.

### Backtest Engine

`core/backtest/engine.py:BacktestEngine` orchestrates a backtest:
1. Fetches price data from the common read-only DuckDB (`core/data_access/duckdb_adapter.py`)
2. Builds the JSON input payload
3. Selects a compatible execution node
4. Uses `ModelRunnerFactory` to create the appropriate runner (host or docker)
5. Executes the model and validates output
6. Simulates trades using `core/backtest/portfolio.py:Portfolio`
7. Computes basic metrics (total return, volatility, Sharpe, max drawdown)
8. Persists results to a model-specific DuckDB file (not the common database)

The portfolio uses a naive policy: buy 1 unit on `long` signal, sell 1 unit on `short` signal. Transaction costs and slippage are configurable.

**Important**: Results (`model_runs`, `model_signals`, `trades`, `model_metrics`) are written to isolated per-run databases to avoid write contention on the shared market data database.

### Configuration

`core/config.py` loads YAML configuration files. Base config is in `config/base.yaml`:
- `storage.duckdb_path`: DuckDB file location
- `backtest_defaults`: initial capital, transaction costs, slippage
- `nodes`: execution node definitions
- `docker`: Docker runtime configuration
- `env_root`: virtual environment directory for host models (default: `.envs`)

Experiment configs (e.g., `experiments/daily_equities_example.yaml`) define universes, date ranges, models to run, and evaluation metrics.

### CLI

`core/cli.py` uses Typer to expose commands:
- `quantfw list-models`: Discover models
- `quantfw init-db`: Create DuckDB schema
- `quantfw backtest`: Run a backtest with specified parameters

When using `--params` on Windows PowerShell, wrap JSON in single quotes or use `--params-file` to avoid escaping issues.

## Adding a New Model

1. Create directory: `models/<author>/<model_name>/`
2. Add `model.json` with metadata (see `models/alice/sector_momentum_v1/model.json` as example)
3. Implement executable (Python script, Go binary, etc.) that:
   - Reads JSON from stdin
   - Writes JSON to stdout
   - Matches the input/output contract
   - Reads market data from the common `metaquant.duckdb` (read-only)
   - Writes results/checkpoints to its own local DuckDB file (never to the common database)
4. For Python host models, add `requirements.txt` if needed
5. Model is auto-discovered; no core code changes required

## Important Notes

### Windows-Specific

- Use `uv venv` and `uv pip install` for virtual environments (handles OneDrive/locked files better than pip)
- In PowerShell, escape JSON parameters carefully or use `--params-file`
- Virtual environments are created in `.envs/` (set in `config/base.yaml`)
- Set `UV_VENV_CLEAR=1` to force venv recreation if you encounter locked file errors

### Data Ingestion

Before running backtests, ensure data is populated in DuckDB:
- Run `quantfw init-db` to create the schema
- Use ingestion scripts in `scripts/` to populate data from Yahoo Finance, CSV, or other sources
- Verify data exists for your security IDs and date ranges before backtesting

### Model Execution Flow

1. `BacktestEngine.run()` is called with model manifest and parameters
2. Engine reads market data from common `metaquant.duckdb` (read-only)
3. Node selection logic in `core/runtime/nodes.py:select_node()` picks a compatible node
4. `ModelRunnerFactory` creates runner based on manifest's `runtime`
5. For host models: venv is created/verified, dependencies installed, entrypoint executed
6. For docker models: container is launched with input on stdin
7. Model output is validated against `ModelOutput` schema
8. Signals are converted to trades, portfolio is simulated, metrics computed
9. Results persisted to model-specific DuckDB file (`data/<model_id>_<run_id>.duckdb`)

This flow ensures models can run in parallel without database locking conflicts.

### Testing

Tests are in `tests/`. Run with `uv run pytest`. Current test files:
- `test_registry.py`: Model discovery and manifest validation
- `test_runtimes.py`: Host and Docker runner execution
