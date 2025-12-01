"""Backtesting engine for MetaQuant."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from core.data_access.duckdb_adapter import DuckDBAdapter
from core.models.base_contract import ModelInput, ModelOutput, SecurityDefinition, TimeRange
from core.models.runner import ModelRunner
from core.registry.schema import ModelManifest
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
    def __init__(self, adapter: DuckDBAdapter) -> None:
        self.adapter = adapter

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
    ) -> BacktestResult:
        run_id = str(uuid.uuid4())
        prices = self.adapter.fetch_daily_prices(security_ids, start_date, end_date)
        payload = self.build_payload(manifest, run_id, security_ids, prices, start_date, end_date, params)

        runner = ModelRunner(manifest, model_dir)
        output = runner.run(payload)
        validated = ModelOutput(**output)
        signals_df = pd.DataFrame([s.model_dump() for s in validated.signals])

        portfolio = Portfolio(initial_capital, transaction_cost_bps, slippage_bps)
        trades: List[Dict] = []
        # naive policy: buy 1 unit on long, sell all on short
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

        # persistence stubs
        self.adapter.insert_model_run(
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
        self.adapter.insert_signals(signals_df.assign(run_id=run_id))
        if not trades_df.empty:
            self.adapter.insert_trades(trades_df.assign(run_id=run_id))
        metrics_df = pd.DataFrame(
            [
                {"run_id": run_id, "metric_name": k, "scope": "total", "metric_value": v}
                for k, v in metrics.items()
            ]
        )
        self.adapter.insert_metrics(metrics_df)

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
