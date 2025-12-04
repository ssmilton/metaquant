"""CLI entrypoint for MetaQuant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from core.backtest.engine import BacktestEngine
from core.config import load_app_config
from core.data_access.duckdb_adapter import DuckDBAdapter
from core.registry.discovery import ModelRegistry
from core.evaluation import reports

app = typer.Typer(help="MetaQuant CLI")


@app.command()
def list_models(models_path: str = "models") -> None:
    registry = ModelRegistry(models_path)
    for manifest in registry.list_manifests():
        print(f"[bold]{manifest.model_id}[/bold] v{manifest.version} - {manifest.description}")


@app.command()
def init_db(config_path: str = "config/base.yaml") -> None:
    from scripts.init_duckdb_schema import create_schema

    cfg = load_app_config(config_path)
    create_schema(cfg.storage.duckdb_path)
    print(f"Created schema at {cfg.storage.duckdb_path}")


@app.command()
def backtest(
    model_id: str = typer.Option(..., help="Model identifier"),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    security_ids: Optional[str] = typer.Option(None, help="Comma separated security ids"),
    tickers: Optional[str] = typer.Option(None, help="Comma separated ticker symbols (e.g., AAPL,MSFT)"),
    tickers_file: Optional[Path] = typer.Option(None, help="Path to file with ticker symbols (one per line)"),
    params: str = typer.Option("{}", help="JSON string of model parameters"),
    params_file: Optional[Path] = typer.Option(None, help="Path to JSON file with model parameters"),
    lookback_days: int = typer.Option(365, help="Days of historical data before start_date for model lookback calculations (default: 365)"),
    node: Optional[str] = typer.Option(
        None, help="Explicit execution node name (must exist in config nodes)"
    ),
    node_tags: Optional[str] = typer.Option(
        None, help="Comma separated preferred node tags"
    ),
    config_path: str = "config/base.yaml",
) -> None:
    cfg = load_app_config(config_path)
    registry = ModelRegistry()
    manifest = registry.get_model(model_id)

    # Validate mutual exclusivity of security specification methods
    specified_methods = sum([
        security_ids is not None,
        tickers is not None,
        tickers_file is not None
    ])

    if specified_methods == 0:
        print("[red]Error:[/red] Must specify one of: --security-ids, --tickers, or --tickers-file")
        raise typer.Exit(1)

    if specified_methods > 1:
        print("[red]Error:[/red] Cannot specify more than one of: --security-ids, --tickers, --tickers-file")
        raise typer.Exit(1)

    # Open market data database in read-only mode to avoid write contention
    adapter = DuckDBAdapter(cfg.storage.duckdb_path, read_only=True)
    engine = BacktestEngine(
        adapter, nodes=cfg.nodes, docker_config=cfg.docker, env_root=cfg.env_root
    )

    # Resolve security IDs from tickers if needed
    if security_ids:
        ids = [int(x) for x in security_ids.split(",")]
    elif tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        try:
            ids = adapter.lookup_security_ids(ticker_list)
            print(f"[cyan]Resolved tickers {ticker_list} to security IDs {ids}[/cyan]")
        except ValueError as e:
            print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    else:  # tickers_file
        resolved_path = tickers_file.resolve()
        if not resolved_path.exists():
            print(f"[red]Error:[/red] Tickers file not found: {tickers_file}")
            print(f"[yellow]Resolved to:[/yellow] {resolved_path}")
            raise typer.Exit(1)
        ticker_list = [
            line.strip().upper()
            for line in resolved_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not ticker_list:
            print(f"[red]Error:[/red] No tickers found in file: {tickers_file}")
            raise typer.Exit(1)
        try:
            ids = adapter.lookup_security_ids(ticker_list)
            print(f"[cyan]Loaded {len(ticker_list)} tickers from {tickers_file}[/cyan]")
            print(f"[cyan]Resolved to security IDs {ids}[/cyan]")
        except ValueError as e:
            print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Load parameters from file if provided, otherwise parse JSON string
    if params_file:
        # Resolve the path to handle Windows .\file syntax correctly
        resolved_path = params_file.resolve()
        if not resolved_path.exists():
            print(f"[red]Error:[/red] Parameter file not found: {params_file}")
            print(f"[yellow]Resolved to:[/yellow] {resolved_path}")
            raise typer.Exit(1)
        param_dict = json.loads(resolved_path.read_text())
    else:
        try:
            param_dict = json.loads(params)
        except json.JSONDecodeError as e:
            print(f"[red]Error parsing --params JSON:[/red] {e}")
            print(f"[yellow]Tip:[/yellow] In PowerShell, use: --params '{{\"key\":\"value\"}}' or --params-file path/to/params.json")
            raise typer.Exit(1)

    tags = [tag.strip() for tag in node_tags.split(",") if tag.strip()] if node_tags else None
    result = engine.run(
        manifest,
        manifest.path.parent,
        ids,
        start_date,
        end_date,
        param_dict,
        initial_capital=cfg.backtest_defaults.initial_capital,
        transaction_cost_bps=cfg.backtest_defaults.transaction_cost_bps,
        slippage_bps=cfg.backtest_defaults.slippage_bps,
        lookback_days=lookback_days,
        node_name=node,
        node_tags=tags,
    )
    summary = reports.summarize_run(result.run_id, result.equity_curve)
    print("Run ID:", result.run_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
