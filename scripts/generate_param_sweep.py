"""Generate experiment configurations for parameter sweeps.

This script helps automate the creation of experiment YAML files for grid search
and parameter optimization. It generates all combinations of specified parameters
and creates a ready-to-run experiment configuration.

Usage:
    # Basic grid search
    python scripts/generate_param_sweep.py \\
        --model-id sector_momentum_v1 \\
        --params lookback_days=10,20,30 top_n_sectors=2,3,5 \\
        --tickers AAPL,MSFT,GOOGL \\
        --start-date 2020-01-01 \\
        --end-date 2023-12-31 \\
        --output experiments/momentum_sweep.yaml

    # Using tickers file
    python scripts/generate_param_sweep.py \\
        --model-id sector_momentum_v1 \\
        --params lookback_days=10,20,30 \\
        --tickers-file sp500.txt \\
        --start-date 2020-01-01 \\
        --end-date 2023-12-31 \\
        --output experiments/sweep.yaml
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def parse_param_grid(param_strings: list[str]) -> dict[str, list[Any]]:
    """Parse parameter specifications into a grid dictionary.

    Input format: ["param1=val1,val2,val3", "param2=0.1,0.2"]
    Output: {"param1": [val1, val2, val3], "param2": [0.1, 0.2]}
    """
    param_grid = {}

    for param_spec in param_strings:
        if '=' not in param_spec:
            raise ValueError(f"Invalid parameter spec: {param_spec}. Expected format: param=val1,val2,val3")

        param_name, values_str = param_spec.split('=', 1)
        param_name = param_name.strip()

        # Parse values (attempt to convert to numbers where possible)
        values = []
        for val_str in values_str.split(','):
            val_str = val_str.strip()

            # Try to convert to number
            try:
                # Try int first
                if '.' not in val_str:
                    values.append(int(val_str))
                else:
                    values.append(float(val_str))
            except ValueError:
                # Keep as string (handles booleans, strings, etc.)
                if val_str.lower() == 'true':
                    values.append(True)
                elif val_str.lower() == 'false':
                    values.append(False)
                else:
                    values.append(val_str)

        param_grid[param_name] = values

    return param_grid


def generate_experiment_config(
    model_id: str,
    param_grid: dict[str, list[Any]],
    tickers: list[str] | None = None,
    security_ids: list[int] | None = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    experiment_id: str | None = None,
    description: str | None = None,
    initial_capital: int = 100000,
    transaction_cost_bps: int = 5,
    slippage_bps: int = 1,
    benchmark: str = "SPY",
    metrics: list[str] | None = None,
) -> dict:
    """Generate experiment configuration with all parameter combinations."""

    if not experiment_id:
        experiment_id = f"{model_id}_param_sweep"

    if not description:
        param_summary = ", ".join(f"{k}={len(v)} values" for k, v in param_grid.items())
        description = f"Parameter sweep for {model_id}: {param_summary}"

    if not metrics:
        metrics = ["sharpe", "sortino", "max_drawdown", "cagr", "calmar", "win_rate"]

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"Generating {len(combinations)} parameter combinations:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print()

    # Build model entries
    models = []
    for combo in combinations:
        params = dict(zip(param_names, combo))
        models.append({
            "model_id": model_id,
            "params": params
        })

    # Build universe specification
    universe = {"type": "equity"}
    if tickers:
        universe["tickers"] = tickers
    elif security_ids:
        universe["security_ids"] = security_ids
    else:
        raise ValueError("Must specify either tickers or security_ids")

    # Build experiment config
    config = {
        "experiment_id": experiment_id,
        "description": description,
        "universe": universe,
        "data": {
            "source": "duckdb",
            "start_date": start_date,
            "end_date": end_date
        },
        "backtest": {
            "frequency": "1D",
            "initial_capital": initial_capital,
            "transaction_cost_bps": transaction_cost_bps,
            "slippage_bps": slippage_bps
        },
        "models": models,
        "evaluation": {
            "metrics": metrics,
            "compare_vs_benchmark": benchmark
        }
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sweep experiment configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parameter sweep
  python scripts/generate_param_sweep.py \\
      --model-id sector_momentum_v1 \\
      --params lookback_days=10,20,30 top_n_sectors=2,3,5 \\
      --tickers AAPL,MSFT,GOOGL \\
      --start-date 2020-01-01 \\
      --end-date 2023-12-31 \\
      --output experiments/momentum_sweep.yaml

  # Float parameters and booleans
  python scripts/generate_param_sweep.py \\
      --model-id my_model \\
      --params threshold=0.01,0.05,0.1 use_filter=true,false \\
      --tickers-file sp500.txt \\
      --start-date 2020-01-01 \\
      --end-date 2023-12-31 \\
      --output experiments/my_sweep.yaml

  # Custom experiment ID and benchmark
  python scripts/generate_param_sweep.py \\
      --model-id sector_momentum_v1 \\
      --params lookback_days=20,30,60 \\
      --tickers AAPL,MSFT \\
      --start-date 2020-01-01 \\
      --end-date 2023-12-31 \\
      --experiment-id momentum_optimization_q1_2024 \\
      --benchmark QQQ \\
      --output experiments/momentum_opt.yaml
        """
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier to run parameter sweep on"
    )

    parser.add_argument(
        "--params",
        nargs="+",
        required=True,
        help='Parameter specifications in format: param1=val1,val2,val3 param2=0.1,0.2'
    )

    # Universe specification (mutually exclusive)
    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument(
        "--tickers",
        help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    universe_group.add_argument(
        "--tickers-file",
        type=Path,
        help="Path to file with ticker symbols (one per line)"
    )
    universe_group.add_argument(
        "--security-ids",
        help="Comma-separated list of security IDs (e.g., 1,2,3)"
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Backtest start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="Backtest end date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for experiment YAML file"
    )

    parser.add_argument(
        "--experiment-id",
        help="Custom experiment identifier (default: <model_id>_param_sweep)"
    )

    parser.add_argument(
        "--description",
        help="Custom experiment description"
    )

    parser.add_argument(
        "--initial-capital",
        type=int,
        default=100000,
        help="Initial capital for backtests (default: 100000)"
    )

    parser.add_argument(
        "--transaction-cost-bps",
        type=int,
        default=5,
        help="Transaction cost in basis points (default: 5)"
    )

    parser.add_argument(
        "--slippage-bps",
        type=int,
        default=1,
        help="Slippage in basis points (default: 1)"
    )

    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker for comparison (default: SPY)"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Evaluation metrics (default: sharpe, sortino, max_drawdown, cagr, calmar, win_rate)"
    )

    args = parser.parse_args()

    # Parse parameter grid
    try:
        param_grid = parse_param_grid(args.params)
    except ValueError as e:
        parser.error(str(e))

    # Parse universe
    tickers = None
    security_ids = None

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        if not args.tickers_file.exists():
            parser.error(f"Tickers file not found: {args.tickers_file}")
        tickers = [
            line.strip().upper()
            for line in args.tickers_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not tickers:
            parser.error(f"No tickers found in file: {args.tickers_file}")
    elif args.security_ids:
        security_ids = [int(x.strip()) for x in args.security_ids.split(",")]

    # Generate experiment config
    config = generate_experiment_config(
        model_id=args.model_id,
        param_grid=param_grid,
        tickers=tickers,
        security_ids=security_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        experiment_id=args.experiment_id,
        description=args.description,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        benchmark=args.benchmark,
        metrics=args.metrics,
    )

    # Save to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"✓ Generated experiment configuration:")
    print(f"  Experiment ID: {config['experiment_id']}")
    print(f"  Model: {args.model_id}")
    print(f"  Combinations: {len(config['models'])}")
    print(f"  Universe size: {len(tickers or security_ids or [])}")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Output: {args.output}")
    print()
    print(f"Run with: quantfw run-experiment {args.output}")


if __name__ == "__main__":
    main()
