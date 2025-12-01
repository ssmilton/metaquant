"""Simple portfolio accounting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class Trade:
    timestamp: str
    security_id: int
    side: str
    quantity: float
    price: float
    fees: float
    pnl: float


@dataclass
class Portfolio:
    initial_cash: float
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    cash: float = field(init=False)
    positions: Dict[int, float] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def _calc_fee(self, price: float, quantity: float) -> float:
        gross = price * quantity
        fee = gross * (self.transaction_cost_bps + self.slippage_bps) / 10_000
        return fee

    def buy(self, timestamp: str, security_id: int, price: float, quantity: float) -> None:
        fee = self._calc_fee(price, quantity)
        cost = price * quantity + fee
        self.cash -= cost
        self.positions[security_id] = self.positions.get(security_id, 0.0) + quantity
        self.trades.append(Trade(timestamp, security_id, "buy", quantity, price, fee, pnl=0.0))

    def sell(self, timestamp: str, security_id: int, price: float, quantity: float) -> None:
        fee = self._calc_fee(price, quantity)
        proceeds = price * quantity - fee
        self.cash += proceeds
        self.positions[security_id] = self.positions.get(security_id, 0.0) - quantity
        self.trades.append(Trade(timestamp, security_id, "sell", quantity, price, fee, pnl=0.0))

    def mark_to_market(self, date: str, price_data: pd.DataFrame) -> float:
        equity = self.cash
        for security_id, qty in self.positions.items():
            if qty == 0:
                continue
            match = price_data[(price_data["security_id"] == security_id) & (price_data["date"] == date)]
            if match.empty:
                continue
            equity += qty * float(match.iloc[0]["close"])
        return equity
