"""Experiment execution engine for batch model evaluation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from core.backtest.engine import BacktestEngine, BacktestResult
from core.data_access.duckdb_adapter import DuckDBAdapter
from core.registry.discovery import ModelRegistry
from core.config import AppConfig
from .schema import ExperimentConfig, ModelRunSpec

logger = logging.getLogger(__name__)


@dataclass
class ModelRunResult:
    """Results from a single model run within an experiment."""

    run_spec: ModelRunSpec
    backtest_result: BacktestResult
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark calculation."""

    benchmark_name: str
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]


@dataclass
class ExperimentResult:
    """Complete results from an experiment."""

    experiment_id: str
    config: ExperimentConfig
    model_results: List[ModelRunResult]
    benchmark_results: List[BenchmarkResult]
    comparison_df: pd.DataFrame
    output_dir: Path


class ExperimentEngine:
    """Engine for orchestrating experiment execution."""

    def __init__(
        self,
        config: AppConfig,
        market_data_adapter: DuckDBAdapter,
        registry: Optional[ModelRegistry] = None
    ) -> None:
        self.config = config
        self.market_data_adapter = market_data_adapter
        self.registry = registry or ModelRegistry()
        self.backtest_engine = BacktestEngine(
            market_data_adapter,
            nodes=config.nodes,
            docker_config=config.docker,
            env_root=config.env_root
        )

    def run_experiment(self, experiment_config: ExperimentConfig) -> ExperimentResult:
        """Execute a complete experiment.

        Args:
            experiment_config: Experiment configuration

        Returns:
            ExperimentResult containing all run results and comparisons
        """
        logger.info("Starting experiment: %s", experiment_config.experiment_id)
        experiment_config.validate()

        # Resolve output directory
        output_dir = Path(experiment_config.output_dir) if experiment_config.output_dir else \
            Path("experiments") / experiment_config.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve universe to security IDs
        security_ids = self._resolve_universe(experiment_config)
        logger.info("Universe resolved to %d securities", len(security_ids))

        # Run all models
        model_results = []
        all_runs = experiment_config.get_all_runs()
        total_runs = len(all_runs)

        for idx, run_spec in enumerate(all_runs, 1):
            logger.info("Running model %d/%d: %s", idx, total_runs, run_spec.display_name)
            result = self._run_single_model(experiment_config, run_spec, security_ids)
            model_results.append(result)

            if not result.success:
                logger.error("Model run failed: %s - %s", run_spec.display_name, result.error)

        # Calculate benchmarks
        benchmark_results = []
        for benchmark_spec in experiment_config.benchmarks:
            logger.info("Calculating benchmark: %s", benchmark_spec.display_name)
            benchmark_result = self._calculate_benchmark(
                experiment_config, benchmark_spec, security_ids
            )
            if benchmark_result:
                benchmark_results.append(benchmark_result)

        # Create comparison DataFrame
        comparison_df = self._create_comparison(
            experiment_config, model_results, benchmark_results
        )

        # Save results
        self._save_results(output_dir, experiment_config, model_results, benchmark_results, comparison_df)

        logger.info("Experiment complete: %s", experiment_config.experiment_id)

        return ExperimentResult(
            experiment_id=experiment_config.experiment_id,
            config=experiment_config,
            model_results=model_results,
            benchmark_results=benchmark_results,
            comparison_df=comparison_df,
            output_dir=output_dir
        )

    def _resolve_universe(self, experiment_config: ExperimentConfig) -> List[int]:
        """Resolve universe specification to security IDs."""
        universe = experiment_config.universe

        if universe.security_ids is not None:
            return universe.security_ids

        if universe.tickers is not None:
            return self.market_data_adapter.lookup_security_ids(universe.tickers)

        if universe.tickers_file is not None:
            tickers_path = Path(universe.tickers_file)
            if not tickers_path.exists():
                raise FileNotFoundError(f"Tickers file not found: {universe.tickers_file}")

            ticker_list = [
                line.strip().upper()
                for line in tickers_path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if not ticker_list:
                raise ValueError(f"No tickers found in file: {universe.tickers_file}")

            return self.market_data_adapter.lookup_security_ids(ticker_list)

        # For macro models, allow empty universe
        return []

    def _run_single_model(
        self,
        experiment_config: ExperimentConfig,
        run_spec: ModelRunSpec,
        security_ids: List[int]
    ) -> ModelRunResult:
        """Run a single model backtest."""
        try:
            # Get model manifest
            manifest = self.registry.get_model(run_spec.model_id)
            model_dir = manifest.path.parent

            # Use experiment-level settings or config defaults
            initial_capital = experiment_config.initial_capital or self.config.backtest_defaults.initial_capital
            transaction_cost_bps = experiment_config.transaction_cost_bps or self.config.backtest_defaults.transaction_cost_bps
            slippage_bps = experiment_config.slippage_bps or self.config.backtest_defaults.slippage_bps

            # Run backtest
            backtest_result = self.backtest_engine.run(
                manifest=manifest,
                model_dir=str(model_dir),
                security_ids=security_ids,
                start_date=experiment_config.start_date,
                end_date=experiment_config.end_date,
                params=run_spec.parameters,
                initial_capital=initial_capital,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                lookback_days=experiment_config.lookback_days,
                node_name=run_spec.node,
                node_tags=run_spec.node_tags
            )

            return ModelRunResult(
                run_spec=run_spec,
                backtest_result=backtest_result,
                success=True
            )

        except Exception as e:
            logger.exception("Error running model %s", run_spec.display_name)
            return ModelRunResult(
                run_spec=run_spec,
                backtest_result=None,
                success=False,
                error=str(e)
            )

    def _calculate_benchmark(
        self,
        experiment_config: ExperimentConfig,
        benchmark_spec,
        security_ids: List[int]
    ) -> Optional[BenchmarkResult]:
        """Calculate benchmark performance."""
        try:
            if benchmark_spec.type == "buy_and_hold":
                return self._calculate_buy_and_hold_benchmark(
                    experiment_config, benchmark_spec, security_ids
                )
            elif benchmark_spec.type == "equal_weight":
                return self._calculate_equal_weight_benchmark(
                    experiment_config, security_ids
                )
            else:
                logger.warning("Unsupported benchmark type: %s", benchmark_spec.type)
                return None

        except Exception as e:
            logger.exception("Error calculating benchmark %s", benchmark_spec.display_name)
            return None

    def _calculate_buy_and_hold_benchmark(
        self,
        experiment_config: ExperimentConfig,
        benchmark_spec,
        security_ids: List[int]
    ) -> BenchmarkResult:
        """Calculate buy-and-hold benchmark for a single security."""
        # If ticker specified, use it; otherwise use first security in universe
        if benchmark_spec.ticker:
            ticker_ids = self.market_data_adapter.lookup_security_ids([benchmark_spec.ticker])
            security_id = ticker_ids[0]
        elif security_ids:
            security_id = security_ids[0]
        else:
            raise ValueError("No security available for buy-and-hold benchmark")

        # Fetch prices
        prices = self.market_data_adapter.fetch_daily_prices(
            [security_id],
            experiment_config.start_date,
            experiment_config.end_date
        )

        if prices.empty:
            raise ValueError(f"No price data for security {security_id}")

        # Calculate equity curve (normalized to initial capital)
        initial_capital = experiment_config.initial_capital or self.config.backtest_defaults.initial_capital
        prices = prices.sort_values("date")
        first_price = prices["close"].iloc[0]
        prices["equity"] = (prices["close"] / first_price) * initial_capital

        equity_curve = prices[["date", "equity"]].copy()

        # Calculate metrics
        from core.backtest.engine import BacktestEngine
        metrics = BacktestEngine._compute_basic_metrics(equity_curve)

        return BenchmarkResult(
            benchmark_name=benchmark_spec.display_name,
            equity_curve=equity_curve,
            metrics=metrics
        )

    def _calculate_equal_weight_benchmark(
        self,
        experiment_config: ExperimentConfig,
        security_ids: List[int]
    ) -> BenchmarkResult:
        """Calculate equal-weight portfolio benchmark."""
        if not security_ids:
            raise ValueError("No securities available for equal-weight benchmark")

        # Fetch prices for all securities
        prices = self.market_data_adapter.fetch_daily_prices(
            security_ids,
            experiment_config.start_date,
            experiment_config.end_date
        )

        if prices.empty:
            raise ValueError("No price data for equal-weight benchmark")

        # Calculate daily returns for each security
        prices = prices.sort_values(["security_id", "date"])
        prices["return"] = prices.groupby("security_id")["close"].pct_change()

        # Calculate equal-weight portfolio returns
        daily_returns = prices.groupby("date")["return"].mean().reset_index()
        daily_returns = daily_returns.sort_values("date")

        # Calculate equity curve
        initial_capital = experiment_config.initial_capital or self.config.backtest_defaults.initial_capital
        daily_returns["equity"] = initial_capital * (1 + daily_returns["return"]).cumprod()
        daily_returns.loc[daily_returns.index[0], "equity"] = initial_capital

        equity_curve = daily_returns[["date", "equity"]].copy()

        # Calculate metrics
        from core.backtest.engine import BacktestEngine
        metrics = BacktestEngine._compute_basic_metrics(equity_curve)

        return BenchmarkResult(
            benchmark_name="Equal Weight Portfolio",
            equity_curve=equity_curve,
            metrics=metrics
        )

    def _create_comparison(
        self,
        experiment_config: ExperimentConfig,
        model_results: List[ModelRunResult],
        benchmark_results: List[BenchmarkResult]
    ) -> pd.DataFrame:
        """Create comparison DataFrame with all metrics."""
        rows = []

        # Add model results
        for result in model_results:
            if result.success and result.backtest_result:
                row = {
                    "name": result.run_spec.display_name,
                    "type": "model",
                    "model_id": result.run_spec.model_id,
                    "run_id": result.backtest_result.run_id,
                    **result.backtest_result.metrics
                }
                rows.append(row)
            else:
                # Include failed runs with null metrics
                row = {
                    "name": result.run_spec.display_name,
                    "type": "model",
                    "model_id": result.run_spec.model_id,
                    "run_id": None,
                    "error": result.error
                }
                # Add null values for all requested metrics
                for metric in experiment_config.metrics:
                    row[metric] = None
                rows.append(row)

        # Add benchmark results
        for result in benchmark_results:
            row = {
                "name": result.benchmark_name,
                "type": "benchmark",
                "model_id": None,
                "run_id": None,
                **result.metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _save_results(
        self,
        output_dir: Path,
        experiment_config: ExperimentConfig,
        model_results: List[ModelRunResult],
        benchmark_results: List[BenchmarkResult],
        comparison_df: pd.DataFrame
    ) -> None:
        """Save experiment results to disk."""
        # Save config
        config_path = output_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(experiment_config.model_dump_json(indent=2))

        # Save comparison table
        comparison_path = output_dir / "comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("Saved comparison to %s", comparison_path)

        # Save detailed JSON results
        results_data = {
            "experiment_id": experiment_config.experiment_id,
            "description": experiment_config.description,
            "models": [
                {
                    "name": r.run_spec.display_name,
                    "model_id": r.run_spec.model_id,
                    "parameters": r.run_spec.parameters,
                    "run_id": r.backtest_result.run_id if r.success else None,
                    "success": r.success,
                    "error": r.error,
                    "metrics": r.backtest_result.metrics if r.success else None
                }
                for r in model_results
            ],
            "benchmarks": [
                {
                    "name": b.benchmark_name,
                    "metrics": b.metrics
                }
                for b in benchmark_results
            ]
        }

        results_path = output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
        logger.info("Saved detailed results to %s", results_path)

        # Save equity curves
        equity_dir = output_dir / "equity_curves"
        equity_dir.mkdir(exist_ok=True)

        for result in model_results:
            if result.success and not result.backtest_result.equity_curve.empty:
                curve_path = equity_dir / f"{result.run_spec.display_name}.csv"
                result.backtest_result.equity_curve.to_csv(curve_path, index=False)

        for result in benchmark_results:
            curve_path = equity_dir / f"{result.benchmark_name}.csv"
            result.equity_curve.to_csv(curve_path, index=False)

        logger.info("Saved equity curves to %s", equity_dir)
