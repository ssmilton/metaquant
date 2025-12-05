"""Experiment configuration schemas."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class UniverseSpec(BaseModel):
    """Specification for the universe of securities to backtest."""

    security_ids: Optional[List[int]] = Field(
        default=None, description="Explicit list of security IDs"
    )
    tickers: Optional[List[str]] = Field(
        default=None, description="List of ticker symbols"
    )
    tickers_file: Optional[str] = Field(
        default=None, description="Path to file with ticker symbols (one per line)"
    )

    def validate_exclusive(self) -> None:
        """Ensure only one universe specification method is provided."""
        specified = sum([
            self.security_ids is not None,
            self.tickers is not None,
            self.tickers_file is not None
        ])
        if specified == 0:
            raise ValueError("Must specify one of: security_ids, tickers, or tickers_file")
        if specified > 1:
            raise ValueError("Cannot specify more than one of: security_ids, tickers, or tickers_file")


class ModelRunSpec(BaseModel):
    """Specification for a single model run within an experiment."""

    model_id: str = Field(..., description="Model identifier")
    name: Optional[str] = Field(
        default=None, description="Optional friendly name for this run (defaults to model_id)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model parameters"
    )
    node: Optional[str] = Field(
        default=None, description="Execution node name"
    )
    node_tags: Optional[List[str]] = Field(
        default=None, description="Preferred node tags"
    )

    @property
    def display_name(self) -> str:
        """Get the display name for this run."""
        return self.name or self.model_id


class ParameterSweep(BaseModel):
    """Specification for parameter sweep over a single model."""

    model_id: str = Field(..., description="Model identifier")
    base_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Base parameters shared across all sweep runs"
    )
    sweep_parameters: Dict[str, List[Any]] = Field(
        ..., description="Parameters to sweep over (key -> list of values)"
    )
    node: Optional[str] = Field(
        default=None, description="Execution node name"
    )
    node_tags: Optional[List[str]] = Field(
        default=None, description="Preferred node tags"
    )

    def generate_runs(self) -> List[ModelRunSpec]:
        """Generate individual model runs from the parameter sweep."""
        import itertools

        if not self.sweep_parameters:
            return [ModelRunSpec(
                model_id=self.model_id,
                parameters=self.base_parameters,
                node=self.node,
                node_tags=self.node_tags
            )]

        # Generate all combinations of sweep parameters
        param_names = list(self.sweep_parameters.keys())
        param_values = [self.sweep_parameters[k] for k in param_names]

        runs = []
        for combo in itertools.product(*param_values):
            params = self.base_parameters.copy()
            params.update(dict(zip(param_names, combo)))

            # Create a name that includes the sweep parameter values
            sweep_parts = [f"{k}={v}" for k, v in zip(param_names, combo)]
            name = f"{self.model_id}[{','.join(sweep_parts)}]"

            runs.append(ModelRunSpec(
                model_id=self.model_id,
                name=name,
                parameters=params,
                node=self.node,
                node_tags=self.node_tags
            ))

        return runs


class BenchmarkSpec(BaseModel):
    """Specification for a benchmark to compare against."""

    type: str = Field(..., description="Benchmark type: 'buy_and_hold', 'equal_weight', 'market_index'")
    ticker: Optional[str] = Field(
        default=None, description="Ticker symbol for buy_and_hold or market_index benchmarks"
    )
    name: Optional[str] = Field(
        default=None, description="Friendly name for the benchmark"
    )

    @property
    def display_name(self) -> str:
        """Get the display name for this benchmark."""
        if self.name:
            return self.name
        if self.type == "buy_and_hold":
            return f"Buy & Hold ({self.ticker})" if self.ticker else "Buy & Hold"
        elif self.type == "equal_weight":
            return "Equal Weight Portfolio"
        elif self.type == "market_index":
            return f"Market Index ({self.ticker})" if self.ticker else "Market Index"
        return self.type


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    description: str = Field(..., description="Human-readable description of the experiment")
    universe: UniverseSpec = Field(..., description="Universe of securities to test")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    lookback_days: int = Field(
        default=365, description="Days of historical data before start_date for lookback"
    )

    # Model execution specifications
    models: List[ModelRunSpec] = Field(
        default_factory=list, description="Individual model runs"
    )
    parameter_sweeps: List[ParameterSweep] = Field(
        default_factory=list, description="Parameter sweep specifications"
    )

    # Benchmarks for comparison
    benchmarks: List[BenchmarkSpec] = Field(
        default_factory=list, description="Benchmarks to compare against"
    )

    # Execution settings
    initial_capital: Optional[float] = Field(
        default=None, description="Initial capital (uses config default if not specified)"
    )
    transaction_cost_bps: Optional[float] = Field(
        default=None, description="Transaction costs in bps"
    )
    slippage_bps: Optional[float] = Field(
        default=None, description="Slippage in bps"
    )

    # Evaluation settings
    metrics: List[str] = Field(
        default_factory=lambda: ["total_return", "cagr", "sharpe", "volatility", "max_drawdown", "hit_rate"],
        description="Metrics to compute and compare"
    )

    # Output settings
    output_dir: Optional[str] = Field(
        default=None, description="Directory for experiment results (defaults to experiments/<experiment_id>)"
    )

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        """Load experiment config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ExperimentConfig(**data)

    def get_all_runs(self) -> List[ModelRunSpec]:
        """Get all model runs including those from parameter sweeps."""
        runs = self.models.copy()
        for sweep in self.parameter_sweeps:
            runs.extend(sweep.generate_runs())
        return runs

    def validate(self) -> None:
        """Validate the experiment configuration."""
        self.universe.validate_exclusive()

        all_runs = self.get_all_runs()
        if not all_runs and not self.benchmarks:
            raise ValueError("Experiment must have at least one model run or benchmark")
