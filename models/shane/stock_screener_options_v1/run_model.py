from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SecuritySeries:
    security_id: int
    dates: List[str]
    close: List[float]
    volume: Optional[List[float]] = None


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan


def build_candidate(security: SecuritySeries) -> Optional[Dict]:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(security.dates),
            "close": security.close,
            "volume": security.volume if security.volume else None,
        }
    ).dropna(subset=["close"])

    if df.shape[0] < 130:  # need ~6 months of data for metrics
        return None

    df = df.sort_values("date")
    latest = df.iloc[-1]

    try:
        ret_6m = (latest.close / df.iloc[-126].close) - 1
        ret_3m = (latest.close / df.iloc[-63].close) - 1
    except Exception:
        return None

    rsi = compute_rsi(df["close"], period=14)
    vol = df["close"].pct_change().std() * math.sqrt(252)

    avg_vol = df["volume"].mean() if df["volume"].notna().any() else np.nan
    dollar_vol = avg_vol * latest.close if not math.isnan(avg_vol) else np.nan

    return {
        "security_id": security.security_id,
        "timestamp": latest.date.strftime("%Y-%m-%d"),
        "price": latest.close,
        "ret_6m": ret_6m,
        "ret_3m": ret_3m,
        "rsi": rsi,
        "vol": vol,
        "avg_volume": avg_vol,
        "dollar_volume": dollar_vol,
    }


def score_candidates(candidates: List[Dict], max_positions: int) -> List[Dict]:
    df = pd.DataFrame(candidates)
    if df.empty:
        return []

    # Rank-based scores; NaN become lowest rank (0)
    df["momentum_score"] = (
        df["ret_6m"].rank(pct=True, na_option="bottom") * 0.6
        + df["ret_3m"].rank(pct=True, na_option="bottom") * 0.4
    )

    df["rsi_score"] = df["rsi"].apply(
        lambda x: 1 - abs(x - 50) / 50 if pd.notna(x) and 30 <= x <= 70 else 0.2
    )

    df["vol_score"] = (1 / (df["vol"] + 1e-6)).rank(pct=True, na_option="bottom")

    if "avg_volume" in df.columns:
        df["liq_score"] = (
            df["avg_volume"].rank(pct=True, na_option="bottom") * 0.6
            + df["dollar_volume"].rank(pct=True, na_option="bottom") * 0.4
        ).fillna(0)
    else:
        df["liq_score"] = 0

    weights = {
        "momentum_score": 0.25,
        "rsi_score": 0.10,
        "vol_score": 0.25,
        "liq_score": 0.10,
    }

    df["composite"] = 0.0
    for col, w in weights.items():
        df["composite"] += df[col] * w

    top = df.sort_values("composite", ascending=False).head(max_positions)

    max_comp = max(top["composite"].max(), 1e-6)
    top["strength"] = top["composite"] / max_comp
    top["confidence"] = top["vol"].apply(lambda v: max(0.2, min(0.95, 1 / (1 + v))) if pd.notna(v) else 0.5)

    return top.to_dict(orient="records")


def make_signal(row: Dict, model_id: str, run_id: str) -> Dict:
    return {
        "timestamp": row["timestamp"],
        "security_id": int(row["security_id"]),
        "signal_type": "long",
        "strength": float(row["strength"]),
        "confidence": float(row["confidence"]),
        "meta": {
            "composite": float(row["composite"]),
            "ret_6m": float(row["ret_6m"]),
            "ret_3m": float(row["ret_3m"]),
            "vol": float(row["vol"]),
        },
    }


def main() -> None:
    payload = json.load(sys.stdin)
    model_id = payload.get("model_id", "stock_screener_options_v1")
    run_id = payload.get("run_id", "")
    params = payload.get("parameters", {}) or {}
    max_positions = int(params.get("max_positions", 8))

    price_payloads = payload.get("data", {}).get("prices", [])
    securities: List[SecuritySeries] = []
    for entry in price_payloads:
        securities.append(
            SecuritySeries(
                security_id=int(entry["security_id"]),
                dates=entry.get("dates", []),
                close=entry.get("close", []),
                volume=entry.get("volume"),
            )
        )

    candidates: List[Dict] = []
    for sec in securities:
        cand = build_candidate(sec)
        if cand:
            candidates.append(cand)

    scored = score_candidates(candidates, max_positions)

    signals = [make_signal(row, model_id, run_id) for row in scored]

    output = {"model_id": model_id, "run_id": run_id, "signals": signals}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
