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
    security_ids: str = typer.Option(..., help="Comma separated security ids"),
    params: str = typer.Option("{}", help="JSON string of model parameters"),
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
    adapter = DuckDBAdapter(cfg.storage.duckdb_path)
    engine = BacktestEngine(
        adapter, nodes=cfg.nodes, docker_config=cfg.docker, env_root=cfg.env_root
    )
    param_dict = json.loads(params)
    ids = [int(x) for x in security_ids.split(",")]
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
        node_name=node,
        node_tags=tags,
    )
    summary = reports.summarize_run(result.run_id, result.equity_curve)
    print("Run ID:", result.run_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
