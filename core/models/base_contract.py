"""JSON contract definitions for model IO."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    start: str
    end: str


class SecurityDefinition(BaseModel):
    security_id: int
    ticker: Optional[str] = None


class PricesPayload(BaseModel):
    security_id: int
    dates: List[str]
    open: Optional[List[float]] = None
    high: Optional[List[float]] = None
    low: Optional[List[float]] = None
    close: List[float]
    volume: Optional[List[float]] = None


class MarketFeaturesPayload(BaseModel):
    date: str
    vix: Optional[float] = None
    spx_return_1d: Optional[float] = None
    realized_vol_30d: Optional[float] = None
    regime_label: Optional[str] = None


class ModelInput(BaseModel):
    mode: str = Field(..., description="backtest or live")
    model_id: str
    run_id: str
    universe: List[SecurityDefinition]
    data: Dict[str, list]
    parameters: Dict[str, object] = Field(default_factory=dict)
    time_range: TimeRange


class Signal(BaseModel):
    timestamp: str
    security_id: int
    signal_type: str
    strength: float
    confidence: float = 1.0
    meta: Dict[str, object] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    model_id: str
    run_id: str
    signals: List[Signal]
