"""Generate walk-forward optimization experiment configurations.

Walk-forward optimization is a more robust approach to parameter optimization that
helps avoid overfitting. It works by:

1. Dividing the time series into multiple train/test windows
2. Optimizing parameters on the train window
3. Testing on the out-of-sample test window
4. Rolling the window forward and repeating

This script generates separate experiment configs for each train window, allowing
you to identify which parameters are consistently optimal across different periods.

Usage:
    python scripts/generate_walk_forward.py \\
        --model-id sector_momentum_v1 \\
        --params lookback_days=10,20,30 top_n_sectors=2,3,5 \\
        --tickers AAPL,MSFT,GOOGL \\
        --start-date 2020-01-01 \\
        --end-date 2023-12-31 \\
        --train-months 12 \\
        --test-months 3 \\
        --output-dir experiments/walk_forward
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def parse_param_grid(param_strings: list[str]) -> dict[str, list[Any]]:
    """Parse parameter specifications into a grid dictionary."""
    param_grid = {}

    for param_spec in param_strings:
        if '=' not in param_spec:
            raise ValueError(f"Invalid parameter spec: {param_spec}. Expected format: param=val1,val2,val3")

        param_name, values_str = param_spec.split('=', 1)
        param_name = param_name.strip()

        values = []
        for val_str in values_str.split(','):
            val_str = val_str.strip()

            try:
                if '.' not in val_str:
                    values.append(int(val_str))
                else:
                    values.append(float(val_str))
            except ValueError:
                if val_str.lower() == 'true':
                    values.append(True)
                elif val_str.lower() == 'false':
                    values.append(False)
                else:
                    values.append(val_str)

        param_grid[param_name] = values

    return param_grid


def generate_walk_forward_windows(
    start_date: str,
    end_date: str,
    train_months: int,
    test_months: int,
    step_months: int | None = None
) -> list[tuple[str, str, str, str]]:
    """Generate train/test date ranges for walk-forward optimization.

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples
    """
    if step_months is None:
        step_months = test_months  # Non-overlapping by default

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    windows = []
    current_start = start

    while True:
        # Calculate train window
        train_start = current_start
        train_end = train_start + timedelta(days=train_months * 30)  # Approximate months

        # Calculate test window
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_months * 30)

        # Stop if test window exceeds end date
        if test_end > end:
            break

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d")
        ))

        # Move window forward
        current_start = current_start + timedelta(days=step_months * 30)

    return windows


def generate_walk_forward_configs(
    model_id: str,
    param_grid: dict[str, list[Any]],
    tickers: list[str] | None = None,
    security_ids: list[int] | None = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    train_months: int = 12,
    test_months: int = 3,
    step_months: int | None = None,
    experiment_id_prefix: str | None = None,
    initial_capital: int = 100000,
    transaction_cost_bps: int = 5,
    slippage_bps: int = 1,
    benchmark: str = "SPY",
    metrics: list[str] | None = None,
) -> list[dict]:
    """Generate multiple experiment configs for walk-forward optimization."""

    if not experiment_id_prefix:
        experiment_id_prefix = f"{model_id}_wf"

    if not metrics:
        metrics = ["sharpe", "sortino", "max_drawdown", "cagr", "calmar", "win_rate"]

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    # Generate walk-forward windows
    windows = generate_walk_forward_windows(
        start_date, end_date, train_months, test_months, step_months
    )

    print(f"Walk-Forward Configuration:")
    print(f"  Parameter combinations: {len(combinations)}")
    print(f"  Time windows: {len(windows)}")
    print(f"  Total experiments: {len(windows)} (train) + {len(windows)} (test)")
    print()

    configs = []

    # Generate experiment for each window
    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        # Build universe specification
        universe = {"type": "equity"}
        if tickers:
            universe["tickers"] = tickers
        elif security_ids:
            universe["security_ids"] = security_ids

        # Build model entries
        models = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            models.append({
                "model_id": model_id,
                "params": params
            })

        # Train experiment
        train_config = {
            "experiment_id": f"{experiment_id_prefix}_train_{idx:02d}",
            "description": f"Walk-forward train window {idx}: {train_start} to {train_end}",
            "universe": universe,
            "data": {
                "source": "duckdb",
                "start_date": train_start,
                "end_date": train_end
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

        # Test experiment (same parameters, different date range)
        test_config = {
            "experiment_id": f"{experiment_id_prefix}_test_{idx:02d}",
            "description": f"Walk-forward test window {idx}: {test_start} to {test_end}",
            "universe": universe,
            "data": {
                "source": "duckdb",
                "start_date": test_start,
                "end_date": test_end
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

        configs.append({
            "window": idx,
            "train_config": train_config,
            "test_config": test_config,
            "train_period": f"{train_start} to {train_end}",
            "test_period": f"{test_start} to {test_end}"
        })

        print(f"Window {idx}:")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end}")

    print()
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate walk-forward optimization experiment configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Walk-forward optimization workflow:
  1. This script generates train/test experiment pairs
  2. Run all train experiments and identify best parameters in each window
  3. Run corresponding test experiments with those parameters
  4. Analyze stability: which parameters perform well consistently?

Example:
  python scripts/generate_walk_forward.py \\
      --model-id sector_momentum_v1 \\
      --params lookback_days=10,20,30 top_n_sectors=2,3,5 \\
      --tickers AAPL,MSFT,GOOGL \\
      --start-date 2020-01-01 \\
      --end-date 2023-12-31 \\
      --train-months 12 \\
      --test-months 3 \\
      --output-dir experiments/walk_forward
        """
    )

    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--params", nargs="+", required=True, help="Parameter grid specs")

    universe_group = parser.add_mutually_exclusive_group(required=True)
    universe_group.add_argument("--tickers", help="Comma-separated ticker symbols")
    universe_group.add_argument("--tickers-file", type=Path, help="File with ticker symbols")
    universe_group.add_argument("--security-ids", help="Comma-separated security IDs")

    parser.add_argument("--start-date", required=True, help="Overall start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Overall end date (YYYY-MM-DD)")
    parser.add_argument("--train-months", type=int, required=True, help="Training window size in months")
    parser.add_argument("--test-months", type=int, required=True, help="Test window size in months")
    parser.add_argument("--step-months", type=int, help="Step size in months (default: same as test-months)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for experiment YAMLs")

    parser.add_argument("--experiment-id-prefix", help="Prefix for experiment IDs (default: <model_id>_wf)")
    parser.add_argument("--initial-capital", type=int, default=100000)
    parser.add_argument("--transaction-cost-bps", type=int, default=5)
    parser.add_argument("--slippage-bps", type=int, default=1)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--metrics", nargs="+")

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
    elif args.security_ids:
        security_ids = [int(x.strip()) for x in args.security_ids.split(",")]

    # Generate walk-forward configs
    configs = generate_walk_forward_configs(
        model_id=args.model_id,
        param_grid=param_grid,
        tickers=tickers,
        security_ids=security_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        experiment_id_prefix=args.experiment_id_prefix,
        initial_capital=args.initial_capital,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        benchmark=args.benchmark,
        metrics=args.metrics,
    )

    # Save configs to files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create a master manifest
    manifest = {
        "walk_forward_optimization": {
            "model_id": args.model_id,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "step_months": args.step_months or args.test_months,
            "total_windows": len(configs),
            "parameter_grid": param_grid,
        },
        "windows": []
    }

    for config_set in configs:
        window_num = config_set["window"]

        # Save train config
        train_file = args.output_dir / f"train_{window_num:02d}.yaml"
        with open(train_file, 'w') as f:
            yaml.dump(config_set["train_config"], f, default_flow_style=False, sort_keys=False, indent=2)

        # Save test config
        test_file = args.output_dir / f"test_{window_num:02d}.yaml"
        with open(test_file, 'w') as f:
            yaml.dump(config_set["test_config"], f, default_flow_style=False, sort_keys=False, indent=2)

        manifest["windows"].append({
            "window": window_num,
            "train_file": str(train_file.name),
            "test_file": str(test_file.name),
            "train_period": config_set["train_period"],
            "test_period": config_set["test_period"]
        })

    # Save manifest
    manifest_file = args.output_dir / "manifest.yaml"
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"✓ Generated walk-forward optimization configs:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Windows: {len(configs)}")
    print(f"  Total experiments: {len(configs) * 2} ({len(configs)} train + {len(configs)} test)")
    print(f"  Manifest: {manifest_file}")
    print()
    print(f"Next steps:")
    print(f"  1. Run all train experiments:")
    print(f"     for f in {args.output_dir}/train_*.yaml; do quantfw run-experiment \"$f\"; done")
    print(f"  2. Identify best parameters from each train window")
    print(f"  3. Run corresponding test experiments with those parameters")
    print(f"  4. Analyze parameter stability across windows")


if __name__ == "__main__":
    main()
