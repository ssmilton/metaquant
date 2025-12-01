"""Performance metrics utilities."""
from __future__ import annotations

import pandas as pd


def compute_total_return(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    return float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1)


def compute_cagr(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    equity_curve = equity_curve.sort_values("date")
    days = (pd.to_datetime(equity_curve["date"]).iloc[-1] - pd.to_datetime(equity_curve["date"]).iloc[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float((equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0]) ** (1 / years) - 1)


def compute_sharpe(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    returns = equity_curve.sort_values("date")["equity"].pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() * 252) / (returns.std() * (252 ** 0.5)))


def compute_max_drawdown(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    equity_curve = equity_curve.sort_values("date")
    running_max = equity_curve["equity"].cummax()
    return float((equity_curve["equity"] / running_max - 1).min())


def compute_volatility(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    returns = equity_curve.sort_values("date")["equity"].pct_change().dropna()
    return float(returns.std() * (252 ** 0.5))


def compute_hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())
