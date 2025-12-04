"""Generate unemployment momentum signals using smoothed RSI."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

# Database path - go up 3 levels from model dir to project root
DB_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'metaquant.duckdb'
@dataclass
class ModelParams:
    db_path: Path = DB_PATH
    series_id: str = "UNRATE"
    smoothing_window: int = 6
    rsi_period: int = 14
    upper_threshold: float = 55.0
    lower_threshold: float = 45.0

    @classmethod
    def from_payload(cls, params: Dict[str, object]) -> "ModelParams":
        return cls(
            db_path=Path(params.get("db_path", cls.db_path)),
            series_id=str(params.get("series_id", cls.series_id)),
            smoothing_window=int(params.get("smoothing_window", cls.smoothing_window)),
            rsi_period=int(params.get("rsi_period", cls.rsi_period)),
            upper_threshold=float(params.get("upper_threshold", cls.upper_threshold)),
            lower_threshold=float(params.get("lower_threshold", cls.lower_threshold)),
        )


def load_fred_series(db_path: Path, series_id: str, end_date: Optional[str]) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    clauses = ["series_id = ?"]
    params: List[object] = [series_id]
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)

    where_clause = " AND ".join(clauses)
    query = f"""
        SELECT date, value, title
        FROM fred_series
        WHERE {where_clause}
        ORDER BY date
    """
    try:
        df = con.execute(query, params).fetch_df()
    except duckdb.Error as exc:  # pragma: no cover - defensive
        con.close()
        raise SystemExit(
            f"Unable to read fred_series from {db_path}: {exc}. Run scripts/init_duckdb_schema.py to create the table."
        ) from exc
    con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def compute_smoothed_rsi(series: pd.Series, smoothing_window: int, rsi_period: int) -> pd.DataFrame:
    smoothed = series.ewm(span=smoothing_window, adjust=False).mean()
    delta = smoothed.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.DataFrame({"smoothed": smoothed, "rsi": rsi})


def build_signals(
    df: pd.DataFrame, params: ModelParams, series_title: str, start_date: Optional[str], end_date: Optional[str]
) -> List[Dict]:
    signals: List[Dict] = []
    indicator = compute_smoothed_rsi(df["value"], params.smoothing_window, params.rsi_period)
    df = df.copy()
    df["smoothed_rate"] = indicator["smoothed"]
    df["rsi"] = indicator["rsi"]

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    for row in df.dropna(subset=["rsi"]).itertuples():
        signal_type: Optional[str]
        strength: float
        if row.rsi >= params.upper_threshold:
            signal_type = "long"
            strength = min(1.0, (row.rsi - 50) / 50)
        elif row.rsi <= params.lower_threshold:
            signal_type = "short"
            strength = min(1.0, (50 - row.rsi) / 50)
        else:
            continue

        confidence = min(1.0, abs(row.rsi - 50) / 50)
        signals.append(
            {
                "timestamp": row.date.strftime("%Y-%m-%d"),
                "security_id": 0,
                "signal_type": signal_type,
                "strength": float(strength),
                "confidence": float(confidence),
                "meta": {
                    "series_id": params.series_id,
                    "series_title": series_title,
                    "unemployment_rate": float(row.value),
                    "smoothed_rate": float(row.smoothed_rate),
                    "rsi": float(row.rsi),
                    "rsi_period": params.rsi_period,
                    "smoothing_window": params.smoothing_window,
                },
            }
        )
    return signals


def main() -> None:
    payload = json.load(sys.stdin)
    params = ModelParams.from_payload(payload.get("parameters", {}))
    time_range = payload.get("time_range", {})
    start_date = time_range.get("start")
    end_date = time_range.get("end")

    df = load_fred_series(params.db_path, params.series_id, end_date)
    if df.empty:
        raise SystemExit(
            f"No data found for series {params.series_id} in {params.db_path}. Run fetch_fred_unemployment.py first."
        )

    signals = build_signals(df, params, df["title"].iloc[0], start_date, end_date)
    output = {"model_id": payload.get("model_id", ""), "run_id": payload.get("run_id", ""), "signals": signals}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
