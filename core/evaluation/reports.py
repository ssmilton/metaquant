"""Reporting utilities for model performance."""
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from . import metrics

logger = logging.getLogger(__name__)


def summarize_run(run_id: str, equity_curve: pd.DataFrame) -> Dict[str, float]:
    returns = equity_curve.sort_values("date")["equity"].pct_change().fillna(0.0)
    summary = {
        "run_id": run_id,
        "total_return": metrics.compute_total_return(equity_curve),
        "cagr": metrics.compute_cagr(equity_curve),
        "sharpe": metrics.compute_sharpe(equity_curve),
        "volatility": metrics.compute_volatility(equity_curve),
        "max_drawdown": metrics.compute_max_drawdown(equity_curve),
        "hit_rate": metrics.compute_hit_rate(returns),
    }
    logger.debug("Run %s summary: %s", run_id, summary)
    return summary
