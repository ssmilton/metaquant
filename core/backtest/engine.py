"""Backtesting engine for MetaQuant."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from core.data_access.duckdb_adapter import DuckDBAdapter
from core.models.base_contract import ModelInput, ModelOutput, SecurityDefinition, TimeRange
from core.models.runner import ModelRunnerFactory
from core.registry.schema import ExecutionNode, ModelManifest
from core.runtime.nodes import select_node
from core.config import DockerRuntimeConfig
from scripts.init_model_results_schema import create_model_results_schema
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    run_id: str
    signals: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]


class BacktestEngine:
    def __init__(
        self,
        adapter: DuckDBAdapter,
        nodes: Optional[List[ExecutionNode]] = None,
        docker_config: Optional[DockerRuntimeConfig] = None,
        env_root: str = ".envs",
        results_db_dir: str | Path = "data",
    ) -> None:
        self.market_data_adapter = adapter  # Read-only adapter for market data
        self.results_db_dir = Path(results_db_dir)
        self.nodes = nodes or [
            ExecutionNode(
                name="local-linux-dev",
                platform="linux/amd64",
                runtimes=["host", "docker"],
                tags=["dev", "default"],
            )
        ]
        self.runner_factory = ModelRunnerFactory(
            docker_config=docker_config, env_root=env_root
        )

    def build_payload(
        self,
        manifest: ModelManifest,
        run_id: str,
        security_ids: Iterable[int],
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
        params: Dict,
    ) -> Dict:
        universe = [SecurityDefinition(security_id=i) for i in security_ids]
        price_payloads: List[Dict] = []
        for sid, group in prices.groupby("security_id"):
            price_payloads.append(
                {
                    "security_id": int(sid),
                    "dates": group["date"].astype(str).tolist(),
                    "close": group["close"].tolist(),
                }
            )
        model_input = ModelInput(
            mode="backtest",
            model_id=manifest.model_id,
            run_id=run_id,
            universe=universe,
            data={"prices": price_payloads},
            parameters=params,
            time_range=TimeRange(start=start_date, end=end_date),
        )
        return model_input.model_dump()

    def _select_node(
        self,
        manifest: ModelManifest,
        node_name: Optional[str] = None,
        node_tags: Optional[List[str]] = None,
    ) -> ExecutionNode:
        if node_name:
            for node in self.nodes:
                if node.name == node_name:
                    if not node.supports(manifest):
                        raise ValueError(
                            f"Node {node_name} does not support runtime {manifest.runtime} and platform {manifest.platform}"
                        )
                    return node
            raise ValueError(f"Requested node {node_name} not found")

        return select_node(manifest, self.nodes, preferred_tags=node_tags)

    def run(
        self,
        manifest: ModelManifest,
        model_dir: str,
        security_ids: Iterable[int],
        start_date: str,
        end_date: str,
        params: Dict,
        initial_capital: float = 100_000,
        transaction_cost_bps: float = 5,
        slippage_bps: float = 1,
        node_name: Optional[str] = None,
        node_tags: Optional[List[str]] = None,
    ) -> BacktestResult:
        run_id = str(uuid.uuid4())
        prices = self.market_data_adapter.fetch_daily_prices(security_ids, start_date, end_date)

        if prices.empty:
            # Get available date ranges to provide helpful error message
            available_ranges = self.market_data_adapter.connection.execute("""
                SELECT security_id, MIN(date) as min_date, MAX(date) as max_date
                FROM daily_prices
                WHERE security_id IN (SELECT UNNEST(?))
                GROUP BY security_id
            """, [list(security_ids)]).fetchall()

            error_msg = (
                f"No price data found for security_ids={list(security_ids)} "
                f"in date range {start_date} to {end_date}.\n"
            )
            if available_ranges:
                error_msg += "Available data ranges for requested securities:\n"
                for sid, min_date, max_date in available_ranges:
                    error_msg += f"  - security_id={sid}: {min_date} to {max_date}\n"
            else:
                error_msg += "No price data exists for these securities. Run the data ingestion script first."

            raise ValueError(error_msg)

        payload = self.build_payload(manifest, run_id, security_ids, prices, start_date, end_date, params)

        node = self._select_node(manifest, node_name=node_name, node_tags=node_tags)
        runner = self.runner_factory.create_runner(manifest, model_dir, node)
        output = runner.run(payload)
        validated = ModelOutput(**output)
        signals_df = pd.DataFrame([s.model_dump() for s in validated.signals])

        portfolio = Portfolio(initial_capital, transaction_cost_bps, slippage_bps)
        trades: List[Dict] = []
        # naive policy: buy 1 unit on long, sell all on short
        if not signals_df.empty:
            for _, signal in signals_df.sort_values("timestamp").iterrows():
                sid = int(signal["security_id"])
                ts = signal["timestamp"]
                price_row = prices[(prices["security_id"] == sid) & (prices["date"] == ts)]
                if price_row.empty:
                    continue
                px = float(price_row.iloc[0]["close"])
                if signal["signal_type"] == "long":
                    portfolio.buy(ts, sid, px, quantity=1)
                elif signal["signal_type"] == "short":
                    portfolio.sell(ts, sid, px, quantity=1)
        for trade in portfolio.trades:
            trades.append(trade.__dict__)
        trades_df = pd.DataFrame(trades)

        equity_records: List[Dict] = []
        for date in sorted(prices["date"].unique()):
            equity = portfolio.mark_to_market(date, prices)
            equity_records.append({"date": date, "equity": equity})
        equity_df = pd.DataFrame(equity_records)
        metrics = self._compute_basic_metrics(equity_df)

        # Create model-specific results database
        results_db_path = self.results_db_dir / f"{manifest.model_id}_{run_id}.duckdb"
        logger.info("Writing results to %s", results_db_path)
        create_model_results_schema(results_db_path)
        results_adapter = DuckDBAdapter(results_db_path, read_only=False)

        # Persist results to model-specific database
        try:
            results_adapter.insert_model_run(
                {
                    "run_id": run_id,
                    "model_id": manifest.model_id,
                    "model_version": manifest.version,
                    "experiment_id": None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "params_json": params,
                    "created_at": pd.Timestamp.utcnow(),
                    "notes": "",
                }
            )
            results_adapter.insert_signals(signals_df.assign(run_id=run_id))
            if not trades_df.empty:
                results_adapter.insert_trades(trades_df.assign(run_id=run_id))
            metrics_df = pd.DataFrame(
                [
                    {"run_id": run_id, "metric_name": k, "scope": "total", "metric_value": v}
                    for k, v in metrics.items()
                ]
            )
            results_adapter.insert_metrics(metrics_df)
        finally:
            results_adapter.close()

        return BacktestResult(run_id, signals_df, trades_df, equity_df, metrics)

    @staticmethod
    def _compute_basic_metrics(equity_df: pd.DataFrame) -> Dict[str, float]:
        if equity_df.empty:
            return {"total_return": 0.0}
        equity_df = equity_df.sort_values("date").reset_index(drop=True)
        equity_df["return"] = equity_df["equity"].pct_change().fillna(0.0)
        total_return = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1
        volatility = equity_df["return"].std() * (252 ** 0.5)
        sharpe = (equity_df["return"].mean() * 252) / volatility if volatility else 0.0
        drawdown = (equity_df["equity"] / equity_df["equity"].cummax() - 1).min()
        return {
            "total_return": float(total_return),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
        }
