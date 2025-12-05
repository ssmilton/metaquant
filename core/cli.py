"""CLI entrypoint for MetaQuant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.table import Table
from rich.console import Console

from core.backtest.engine import BacktestEngine
from core.config import load_app_config
from core.data_access.duckdb_adapter import DuckDBAdapter
from core.registry.discovery import ModelRegistry
from core.evaluation import reports
from core.experiments import ExperimentConfig, ExperimentEngine, ExperimentComparison

app = typer.Typer(help="MetaQuant CLI")
console = Console()


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

    # Check if model requires securities based on input_types
    # Models with non-security input types (e.g., fred_series) are macro models
    requires_securities = True
    if hasattr(manifest, 'input_types') and manifest.input_types:
        # If input_types doesn't include standard security data, it's a macro model
        security_input_types = {'daily_prices', 'options_quotes', 'fundamentals'}
        if not any(it in security_input_types for it in manifest.input_types):
            requires_securities = False
            print(f"[cyan]Model {model_id} is a macro model (input_types: {manifest.input_types})[/cyan]")

    # Validate mutual exclusivity of security specification methods
    specified_methods = sum([
        security_ids is not None,
        tickers is not None,
        tickers_file is not None
    ])

    if specified_methods == 0 and requires_securities:
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
    # For macro models, use empty list if no securities specified
    if not requires_securities and specified_methods == 0:
        ids = []
        print("[cyan]Macro model: running without securities[/cyan]")
    elif security_ids:
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

    print("Run ID:", result.run_id)

    # For macro models without portfolio simulation, show signal count instead of metrics
    if result.equity_curve.empty:
        summary = {
            "run_id": result.run_id,
            "signal_count": len(result.signals),
            "note": "Macro model - portfolio simulation not applicable"
        }
    else:
        summary = reports.summarize_run(result.run_id, result.equity_curve)

    print(json.dumps(summary, indent=2))


@app.command()
def run_experiment(
    config_path: Path = typer.Argument(..., help="Path to experiment YAML config"),
    app_config_path: str = typer.Option("config/base.yaml", help="Path to application config"),
) -> None:
    """Run a complete experiment from a YAML configuration file.

    Example:
        quantfw run-experiment experiments/momentum_sweep.yaml
    """
    if not config_path.exists():
        print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    # Load configurations
    app_config = load_app_config(app_config_path)
    experiment_config = ExperimentConfig.from_yaml(config_path)

    print(f"[bold]Experiment:[/bold] {experiment_config.experiment_id}")
    print(f"[cyan]{experiment_config.description}[/cyan]")
    print()

    # Create engine and run experiment
    adapter = DuckDBAdapter(app_config.storage.duckdb_path, read_only=True)
    engine = ExperimentEngine(app_config, adapter)

    try:
        result = engine.run_experiment(experiment_config)

        print()
        print(f"[green]✓[/green] Experiment complete!")
        print(f"[cyan]Results saved to:[/cyan] {result.output_dir}")
        print()

        # Display comparison table
        print("[bold]Performance Comparison:[/bold]")
        _display_comparison_table(result.comparison_df)

    except Exception as e:
        print(f"[red]Error running experiment:[/red] {e}")
        raise typer.Exit(1)
    finally:
        adapter.close()


@app.command()
def show_experiment(
    experiment_dir: Path = typer.Argument(..., help="Path to experiment output directory"),
    metric: str = typer.Option("sharpe", help="Metric to sort by"),
    top_n: Optional[int] = typer.Option(None, help="Show only top N models"),
) -> None:
    """Display results from a completed experiment.

    Example:
        quantfw show-experiment experiments/momentum_sweep
        quantfw show-experiment experiments/momentum_sweep --metric cagr --top-n 5
    """
    if not experiment_dir.exists():
        print(f"[red]Error:[/red] Experiment directory not found: {experiment_dir}")
        raise typer.Exit(1)

    try:
        comparison_df = ExperimentComparison.load_comparison(experiment_dir)

        print(f"[bold]Experiment Results:[/bold] {experiment_dir.name}")
        print()

        # Optionally filter to top N
        if top_n:
            comparison_df = ExperimentComparison.get_top_n(comparison_df, metric, n=top_n)
            print(f"[cyan]Showing top {top_n} by {metric}[/cyan]")
        else:
            comparison_df = ExperimentComparison.rank_by_metric(comparison_df, metric)

        _display_comparison_table(comparison_df)

    except Exception as e:
        print(f"[red]Error loading experiment:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def compare_experiments(
    experiment_dirs: list[Path] = typer.Argument(..., help="Paths to experiment output directories"),
    metric: str = typer.Option("sharpe", help="Metric to compare"),
) -> None:
    """Compare results across multiple experiments.

    Example:
        quantfw compare-experiments experiments/exp1 experiments/exp2 experiments/exp3
    """
    all_results = []

    for exp_dir in experiment_dirs:
        if not exp_dir.exists():
            print(f"[yellow]Warning:[/yellow] Experiment directory not found: {exp_dir}")
            continue

        try:
            comparison_df = ExperimentComparison.load_comparison(exp_dir)
            comparison_df["experiment"] = exp_dir.name
            all_results.append(comparison_df)
        except Exception as e:
            print(f"[yellow]Warning:[/yellow] Could not load {exp_dir}: {e}")

    if not all_results:
        print("[red]Error:[/red] No valid experiments found")
        raise typer.Exit(1)

    # Combine all results
    import pandas as pd
    combined = pd.concat(all_results, ignore_index=True)

    print("[bold]Cross-Experiment Comparison[/bold]")
    print(f"[cyan]Sorted by {metric}[/cyan]")
    print()

    # Sort by metric
    combined = combined.sort_values(metric, ascending=False, na_position="last")

    _display_comparison_table(combined)


@app.command()
def experiment_report(
    experiment_dir: Path = typer.Argument(..., help="Path to experiment output directory"),
    output_file: Optional[Path] = typer.Option(None, help="Save report to file"),
) -> None:
    """Generate a detailed performance report for an experiment.

    Example:
        quantfw experiment-report experiments/momentum_sweep
        quantfw experiment-report experiments/momentum_sweep --output-file report.txt
    """
    if not experiment_dir.exists():
        print(f"[red]Error:[/red] Experiment directory not found: {experiment_dir}")
        raise typer.Exit(1)

    try:
        comparison_df = ExperimentComparison.load_comparison(experiment_dir)

        # Generate report
        output_path = output_file or experiment_dir / "performance_report.txt"
        report = ExperimentComparison.create_performance_report(comparison_df, output_path)

        print(report)

        if output_file:
            print()
            print(f"[green]✓[/green] Report saved to: {output_path}")

    except Exception as e:
        print(f"[red]Error generating report:[/red] {e}")
        raise typer.Exit(1)


def _display_comparison_table(df):
    """Helper to display comparison DataFrame as a rich table."""
    if df.empty:
        print("[yellow]No results to display[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")

    # Add columns
    for col in df.columns:
        table.add_column(col)

    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)


if __name__ == "__main__":
    app()
