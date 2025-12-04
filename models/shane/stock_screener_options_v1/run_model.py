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


@dataclass
class OptionQuote:
    """Single option quote data point."""
    option_id: int
    date: str
    bid: float
    ask: float
    last: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    open_interest: float
    underlying_price: float
    is_call: bool  # derived from option_id or passed separately


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan


def compute_options_metrics(
    security_id: int, options_data: List[Dict], lookback_days: int = 30
) -> Dict[str, float]:
    """Compute options-aware metrics for a security.

    Args:
        security_id: The underlying security ID
        options_data: List of option quotes for this underlying
        lookback_days: Number of recent days to analyze

    Returns:
        Dictionary of options metrics
    """
    if not options_data:
        return {
            "avg_iv": np.nan,
            "iv_rank": np.nan,
            "put_call_oi_ratio": np.nan,
            "put_call_vol_ratio": np.nan,
            "options_activity": np.nan,
        }

    df = pd.DataFrame(options_data)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to recent data
    cutoff_date = df["date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["date"] >= cutoff_date].copy()

    if recent.empty:
        return {
            "avg_iv": np.nan,
            "iv_rank": np.nan,
            "put_call_oi_ratio": np.nan,
            "put_call_vol_ratio": np.nan,
            "options_activity": np.nan,
        }

    # Compute average implied volatility for recent period
    avg_iv = recent["iv"].mean()

    # IV rank: where current IV sits in historical distribution (0-100 percentile)
    all_iv = df["iv"].dropna()
    if len(all_iv) > 1:
        iv_rank = (all_iv <= avg_iv).sum() / len(all_iv) * 100
    else:
        iv_rank = 50.0

    # Separate calls and puts (assume is_call field or delta sign)
    # Use delta as proxy: positive delta = calls, negative delta = puts
    recent["is_call"] = recent["delta"] > 0
    calls = recent[recent["is_call"]]
    puts = recent[~recent["is_call"]]

    # Put/Call open interest ratio
    call_oi = calls["open_interest"].sum()
    put_oi = puts["open_interest"].sum()
    pc_oi_ratio = put_oi / call_oi if call_oi > 0 else np.nan

    # Put/Call volume ratio (using last traded volume as proxy)
    # Note: volume isn't in the schema, so we'll skip this or use open_interest change
    # For now, set as NaN
    pc_vol_ratio = np.nan

    # Options activity: total open interest as liquidity indicator
    total_oi = recent["open_interest"].sum()
    options_activity = total_oi

    return {
        "avg_iv": avg_iv,
        "iv_rank": iv_rank,
        "put_call_oi_ratio": pc_oi_ratio,
        "put_call_vol_ratio": pc_vol_ratio,
        "options_activity": options_activity,
    }


def build_candidate(
    security: SecuritySeries, options_data: Optional[List[Dict]] = None
) -> Optional[Dict]:
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

    # Compute options metrics if options data available
    options_metrics = compute_options_metrics(
        security.security_id, options_data or [], lookback_days=30
    )

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
        # Options metrics
        "avg_iv": options_metrics["avg_iv"],
        "iv_rank": options_metrics["iv_rank"],
        "put_call_oi_ratio": options_metrics["put_call_oi_ratio"],
        "options_activity": options_metrics["options_activity"],
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

    # Options-aware scoring
    # IV rank score: prefer moderate IV (40-60 percentile is neutral, lower/higher gets scored)
    # Lower IV rank can indicate complacency (contrarian bullish), higher can indicate fear
    if "iv_rank" in df.columns:
        df["iv_score"] = df["iv_rank"].apply(
            lambda x: (
                # Low IV rank (< 30): high score (cheap options, potential breakout)
                0.8 + (30 - x) / 30 * 0.2 if pd.notna(x) and x < 30
                # Moderate IV rank (30-70): neutral score
                else 0.6 if pd.notna(x) and 30 <= x <= 70
                # High IV rank (> 70): lower score (expensive options, high uncertainty)
                else 0.4 - (x - 70) / 30 * 0.2 if pd.notna(x) and x > 70
                else 0.5  # Default for NaN
            )
        )
    else:
        df["iv_score"] = 0.5

    # Put/Call ratio score: lower ratio can indicate bullish sentiment
    # Ratio < 0.7 = bullish (high score), 0.7-1.3 = neutral, > 1.3 = bearish (low score)
    if "put_call_oi_ratio" in df.columns:
        df["pc_score"] = df["put_call_oi_ratio"].apply(
            lambda x: (
                0.8 if pd.notna(x) and x < 0.7  # Bullish
                else 0.6 if pd.notna(x) and 0.7 <= x <= 1.3  # Neutral
                else 0.4 if pd.notna(x)  # Bearish
                else 0.5  # Default for NaN
            )
        )
    else:
        df["pc_score"] = 0.5

    # Options activity score: higher is better (more liquid options market)
    if "options_activity" in df.columns:
        df["options_activity_score"] = df["options_activity"].rank(
            pct=True, na_option="bottom"
        )
    else:
        df["options_activity_score"] = 0.5

    # Updated weights to include options factors
    weights = {
        "momentum_score": 0.20,      # Reduced to make room for options
        "rsi_score": 0.08,
        "vol_score": 0.20,
        "liq_score": 0.08,
        "iv_score": 0.15,            # Options: IV rank
        "pc_score": 0.12,            # Options: Put/Call ratio
        "options_activity_score": 0.17,  # Options: Activity/liquidity
    }

    df["composite"] = 0.0
    for col, w in weights.items():
        if col in df.columns:
            df["composite"] += df[col] * w

    top = df.sort_values("composite", ascending=False).head(max_positions)

    max_comp = max(top["composite"].max(), 1e-6)
    top["strength"] = top["composite"] / max_comp
    top["confidence"] = top["vol"].apply(
        lambda v: max(0.2, min(0.95, 1 / (1 + v))) if pd.notna(v) else 0.5
    )

    return top.to_dict(orient="records")


def make_signal(row: Dict, model_id: str, run_id: str) -> Dict:
    meta = {
        "composite": float(row["composite"]),
        "ret_6m": float(row["ret_6m"]),
        "ret_3m": float(row["ret_3m"]),
        "vol": float(row["vol"]),
    }

    # Add options metrics to metadata if available
    if "avg_iv" in row and pd.notna(row["avg_iv"]):
        meta["avg_iv"] = float(row["avg_iv"])
    if "iv_rank" in row and pd.notna(row["iv_rank"]):
        meta["iv_rank"] = float(row["iv_rank"])
    if "put_call_oi_ratio" in row and pd.notna(row["put_call_oi_ratio"]):
        meta["put_call_oi_ratio"] = float(row["put_call_oi_ratio"])
    if "options_activity" in row and pd.notna(row["options_activity"]):
        meta["options_activity"] = float(row["options_activity"])

    return {
        "timestamp": row["timestamp"],
        "security_id": int(row["security_id"]),
        "signal_type": "long",
        "strength": float(row["strength"]),
        "confidence": float(row["confidence"]),
        "meta": meta,
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

    # Extract options data if provided
    # Expected format: { "options": { <security_id>: [ list of option quotes ] } }
    options_by_security = payload.get("data", {}).get("options", {})

    candidates: List[Dict] = []
    for sec in securities:
        # Get options data for this security (if available)
        sec_options = options_by_security.get(str(sec.security_id), [])
        cand = build_candidate(sec, sec_options)
        if cand:
            candidates.append(cand)

    scored = score_candidates(candidates, max_positions)

    signals = [make_signal(row, model_id, run_id) for row in scored]

    output = {"model_id": model_id, "run_id": run_id, "signals": signals}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
