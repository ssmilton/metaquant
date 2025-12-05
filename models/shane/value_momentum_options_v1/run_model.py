"""Value-Momentum-Options Hybrid Model v1 with Sector Rotation.

Aggressive multi-factor model combining:
- Value: P/E, P/B, FCF yield
- Momentum: 3m, 6m, 12m returns
- Quality: ROE, profit margins, ROA, growth, leverage
- Risk: Inverse volatility
- Options: IV rank, options liquidity, put/call ratios
- Sector Rotation: Dynamic sector tilts based on relative strength

Options overlay:
- Covered calls when IV rank > 60
- Protective puts when VIX > 25 or IV rank < 30

Sector rotation:
- Overweight sectors with strong momentum and fundamentals
- Underweight sectors with weak relative performance
- Dynamic sector limits (15-45% based on sector strength)

Target: Sharpe > 0.8, Max DD < 35%, 15-30 positions, 30-day minimum hold.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd


# =====================================================================
# Configuration and Constants
# =====================================================================

# Database path - go up 3 levels from model dir to project root
DB_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'metaquant.duckdb'

DEFAULT_PARAMS = {
    'target_positions': 20,
    'min_market_cap': 500_000_000,        # $500M
    'min_dollar_volume': 5_000_000,       # $5M daily
    'max_volatility': 0.80,               # 80% annualized
    'min_roe': 0.10,                      # 10%
    'max_pe': 50,
    'max_pb': 10,
    'min_profit_margin': 0.05,            # 5%
    'max_debt_to_equity': 2.0,
    'momentum_6m_min': -0.10,             # -10%
    'momentum_3m_min': -0.15,             # -15%
    'sector_max_pct': 0.35,               # 35% base sector concentration
    'position_weight_cap': 0.10,          # 10% max single position
    'stop_loss_pct': -0.20,               # -20% hard stop
    'trailing_stop_trigger': 0.15,        # Activate at +15% gain
    'trailing_stop_pct': 0.10,            # 10% trailing stop
    'covered_call_iv_threshold': 60,      # IV rank
    'protective_put_vix_threshold': 25,   # VIX level
    'protective_put_iv_threshold': 30,    # IV rank
    'min_options_oi': 100,                # Minimum open interest
    # Sector rotation parameters
    'enable_sector_rotation': True,       # Enable sector rotation overlay
    'sector_rotation_strength': 0.30,     # 30% tilt strength (0=none, 1=max)
    'sector_lookback_months': 3,          # Lookback for sector momentum
    'sector_max_tilt': 0.45,              # Max sector allocation for strong sectors
    'sector_min_tilt': 0.15,              # Min sector allocation for weak sectors
}

FACTOR_WEIGHTS = {
    'value_score': 0.25,
    'momentum_score': 0.25,
    'quality_score': 0.20,
    'risk_score': 0.10,
    'options_score': 0.20,
}


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class SecuritySeries:
    """Price series for a single security."""
    security_id: int
    dates: List[str]
    close: List[float]
    volume: Optional[List[float]] = None


# =====================================================================
# DuckDB Data Loading
# =====================================================================

def load_fundamentals(security_ids: List[int]) -> pd.DataFrame:
    """Load and pivot fundamentals from DuckDB.

    Returns DataFrame with security_id as index and fundamental metrics as columns.
    """
    if not security_ids:
        return pd.DataFrame()

    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        placeholders = ','.join(['?'] * len(security_ids))
        query = f"""
            SELECT security_id, report_date, metric_name, metric_value
            FROM fundamentals
            WHERE security_id IN ({placeholders})
            ORDER BY security_id, report_date DESC
        """
        df = con.execute(query, security_ids).fetch_df()
        con.close()

        if df.empty:
            return pd.DataFrame()

        # Pivot to wide format: one row per security with latest metrics
        pivot = df.pivot_table(
            index='security_id',
            columns='metric_name',
            values='metric_value',
            aggfunc='first'  # Take most recent report_date
        )
        return pivot

    except Exception as e:
        print(f"Warning: Failed to load fundamentals: {e}", file=sys.stderr)
        return pd.DataFrame()


def load_securities(security_ids: List[int]) -> pd.DataFrame:
    """Load ticker, sector, industry metadata from DuckDB."""
    if not security_ids:
        return pd.DataFrame()

    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        placeholders = ','.join(['?'] * len(security_ids))
        query = f"""
            SELECT security_id, ticker, sector, industry
            FROM securities
            WHERE security_id IN ({placeholders})
        """
        df = con.execute(query, security_ids).fetch_df()
        con.close()
        return df.set_index('security_id')

    except Exception as e:
        print(f"Warning: Failed to load securities metadata: {e}", file=sys.stderr)
        return pd.DataFrame()


# =====================================================================
# Statistical Utilities
# =====================================================================

def zscore_winsorized(series: pd.Series) -> pd.Series:
    """Compute z-score with winsorization at ±3 standard deviations."""
    median = series.median()
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0, index=series.index)
    z = (series - median) / std
    return z.clip(-3, 3)


def rank_pct(series: pd.Series) -> pd.Series:
    """Compute rank percentile [0, 1]."""
    return series.rank(pct=True, na_option='bottom')


# =====================================================================
# Options Metrics (adapted from stock_screener_options_v1)
# =====================================================================

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
            "options_activity": np.nan,
        }

    try:
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
                "options_activity": np.nan,
            }

        # Compute average implied volatility
        avg_iv = recent["iv"].mean()

        # IV rank: where current IV sits in historical distribution (0-100 percentile)
        all_iv = df["iv"].dropna()
        if len(all_iv) > 1:
            iv_rank = (all_iv <= avg_iv).sum() / len(all_iv) * 100
        else:
            iv_rank = 50.0

        # Separate calls and puts (use delta as proxy: positive = calls, negative = puts)
        recent["is_call"] = recent["delta"] > 0
        calls = recent[recent["is_call"]]
        puts = recent[~recent["is_call"]]

        # Put/Call open interest ratio
        call_oi = calls["open_interest"].sum()
        put_oi = puts["open_interest"].sum()
        pc_oi_ratio = put_oi / call_oi if call_oi > 0 else np.nan

        # Options activity: total open interest as liquidity indicator
        total_oi = recent["open_interest"].sum()

        return {
            "avg_iv": avg_iv,
            "iv_rank": iv_rank,
            "put_call_oi_ratio": pc_oi_ratio,
            "options_activity": total_oi,
        }

    except Exception as e:
        print(f"Warning: Failed to compute options metrics for security {security_id}: {e}", file=sys.stderr)
        return {
            "avg_iv": np.nan,
            "iv_rank": np.nan,
            "put_call_oi_ratio": np.nan,
            "options_activity": np.nan,
        }


# =====================================================================
# Price Series Processing
# =====================================================================

def build_price_dataframe(price_payload: Dict) -> pd.DataFrame:
    """Build DataFrame from price payload."""
    df = pd.DataFrame({
        'date': pd.to_datetime(price_payload.get('dates', [])),
        'close': price_payload.get('close', []),
        'volume': price_payload.get('volume'),
    })
    return df.dropna(subset=['close']).sort_values('date')


def compute_price_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute price-based metrics: returns, volatility, RSI."""
    # Need at least 63 trading days (3 months) for meaningful metrics
    if df.shape[0] < 63:
        return {
            'ret_12m': np.nan,
            'ret_6m': np.nan,
            'ret_3m': np.nan,
            'volatility': np.nan,
            'avg_volume': np.nan,
            'dollar_volume': np.nan,
        }

    latest = df.iloc[-1]

    try:
        ret_12m = (latest.close / df.iloc[-252].close) - 1 if len(df) >= 252 else np.nan
        ret_6m = (latest.close / df.iloc[-126].close) - 1 if len(df) >= 126 else np.nan
        ret_3m = (latest.close / df.iloc[-63].close) - 1 if len(df) >= 63 else np.nan
    except Exception:
        ret_12m = ret_6m = ret_3m = np.nan

    # Annualized volatility (use all available data)
    returns = df['close'].pct_change()
    volatility = returns.std() * math.sqrt(252)

    # Volume metrics
    avg_volume = df['volume'].mean() if 'volume' in df and df['volume'].notna().any() else np.nan
    dollar_volume = avg_volume * latest.close if not pd.isna(avg_volume) else np.nan

    return {
        'ret_12m': ret_12m,
        'ret_6m': ret_6m,
        'ret_3m': ret_3m,
        'volatility': volatility,
        'avg_volume': avg_volume,
        'dollar_volume': dollar_volume,
    }


# =====================================================================
# Candidate Building
# =====================================================================

def build_candidate(
    sec_id: int,
    price_df: pd.DataFrame,
    fundamentals: Optional[pd.Series],
    options_metrics: Dict[str, float],
    sector: Optional[str],
) -> Optional[Dict]:
    """Build candidate dictionary from price, fundamentals, and options data."""

    if price_df.shape[0] < 63:  # Need at least 3 months of data
        return None

    latest = price_df.iloc[-1]
    price_metrics = compute_price_metrics(price_df)

    candidate = {
        'security_id': sec_id,
        'timestamp': latest.date.strftime('%Y-%m-%d'),
        'price': latest.close,
        'sector': sector if sector else 'Unknown',
        **price_metrics,
        **options_metrics,
    }

    # Add fundamentals if available
    if fundamentals is not None and not fundamentals.empty:
        for metric in ['trailing_pe', 'forward_pe', 'price_to_book', 'roe', 'roa',
                       'profit_margins', 'revenue_growth', 'earnings_growth',
                       'free_cash_flow', 'debt_to_equity', 'market_cap']:
            candidate[metric] = fundamentals.get(metric, np.nan)
    else:
        # Fill with NaN if no fundamentals
        for metric in ['trailing_pe', 'forward_pe', 'price_to_book', 'roe', 'roa',
                       'profit_margins', 'revenue_growth', 'earnings_growth',
                       'free_cash_flow', 'debt_to_equity', 'market_cap']:
            candidate[metric] = np.nan

    return candidate


# =====================================================================
# Factor Scoring Functions
# =====================================================================

def compute_value_score(df: pd.DataFrame) -> pd.Series:
    """Compute value score from P/E, P/B, FCF yield.

    Lower valuations = higher scores.
    """
    # Z-score normalization (inverted for valuation multiples)
    pe_score = -zscore_winsorized(df['trailing_pe'])
    pb_score = -zscore_winsorized(df['price_to_book'])

    # FCF yield = FCF / Market Cap (higher is better)
    df['fcf_yield'] = df['free_cash_flow'] / df['market_cap']
    fcf_score = zscore_winsorized(df['fcf_yield'])

    # Composite value score
    value_score = (
        0.40 * pe_score +
        0.30 * pb_score +
        0.30 * fcf_score
    )
    return value_score.fillna(0)


def compute_momentum_score(df: pd.DataFrame) -> pd.Series:
    """Compute momentum score from 3m, 6m, 12m returns.

    Higher returns = higher scores.
    """
    momentum_score = (
        0.25 * rank_pct(df['ret_12m']) +
        0.50 * rank_pct(df['ret_6m']) +
        0.25 * rank_pct(df['ret_3m'])
    )
    return momentum_score.fillna(0)


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """Compute quality score from ROE, margins, ROA, growth, leverage."""

    # Profitability (higher is better)
    roe_score = zscore_winsorized(df['roe'])
    margin_score = zscore_winsorized(df['profit_margins'])
    roa_score = zscore_winsorized(df['roa'])

    # Growth stability: prefer 5-30% revenue growth
    def growth_score_fn(x):
        if pd.isna(x):
            return 0.0
        if 0.05 <= x <= 0.30:
            return 1.0
        elif x > 0:
            return 0.5
        return 0.0

    growth_score = df['revenue_growth'].apply(growth_score_fn)

    # Leverage (lower is better)
    debt_score = -zscore_winsorized(df['debt_to_equity'])

    # Composite quality score
    quality_score = (
        0.35 * roe_score +
        0.25 * margin_score +
        0.15 * roa_score +
        0.15 * growth_score +
        0.10 * debt_score
    )
    return quality_score.fillna(0)


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Compute risk score from volatility (inverse relationship)."""
    return 1 / (1 + df['volatility'])


def compute_options_score(df: pd.DataFrame) -> pd.Series:
    """Compute options score from IV rank, options liquidity, P/C ratio."""

    # IV rank score: prefer moderate IV (40-70 percentile)
    def iv_score_fn(x):
        if pd.isna(x):
            return 0.5
        if 40 <= x <= 70:
            return 0.9  # Optimal: moderate IV
        elif 30 <= x < 40:
            return 0.7  # Low IV
        elif 70 < x <= 85:
            return 0.5  # High IV
        return 0.3  # Very high or very low

    iv_score = df['iv_rank'].apply(iv_score_fn)

    # Options activity score: higher open interest = better
    oi_score = rank_pct(df['options_activity'])

    # Put/Call ratio score: neutral to slightly bullish preferred
    def pc_score_fn(x):
        if pd.isna(x):
            return 0.5
        if 0.8 <= x <= 1.2:
            return 0.8  # Neutral
        elif 0.5 <= x < 0.8:
            return 0.7  # Bullish
        return 0.5  # Bearish

    pc_score = df['put_call_oi_ratio'].apply(pc_score_fn)

    # Composite options score
    options_score = (
        0.50 * iv_score +
        0.35 * oi_score +
        0.15 * pc_score
    )
    return options_score.fillna(0.5)


# =====================================================================
# Sector Rotation
# =====================================================================

def compute_sector_scores(df: pd.DataFrame, params: Dict) -> Dict[str, float]:
    """Compute sector-level scores based on momentum and fundamentals.

    Returns dictionary mapping sector -> score (0-1 range).
    Strong sectors get scores > 0.5, weak sectors get < 0.5.
    """
    if df.empty or 'sector' not in df.columns:
        return {}

    # Group by sector
    sector_groups = df.groupby('sector')

    sector_stats = {}
    for sector, group in sector_groups:
        if len(group) < 2:  # Need at least 2 stocks for meaningful stats
            continue

        # Sector momentum (average returns)
        lookback_months = params.get('sector_lookback_months', 3)
        if lookback_months == 3:
            sector_momentum = group['ret_3m'].mean()
        elif lookback_months == 6:
            sector_momentum = group['ret_6m'].mean()
        else:  # 12 months
            sector_momentum = group['ret_12m'].mean()

        # Sector fundamentals (average quality metrics)
        sector_roe = group['roe'].mean()
        sector_margin = group['profit_margins'].mean()
        sector_pe = group['trailing_pe'].median()  # Use median for P/E (less sensitive to outliers)

        sector_stats[sector] = {
            'momentum': sector_momentum,
            'roe': sector_roe,
            'margin': sector_margin,
            'pe': sector_pe,
            'count': len(group),
        }

    if not sector_stats:
        return {}

    # Normalize momentum and fundamentals across sectors
    sectors = list(sector_stats.keys())
    momentums = [sector_stats[s]['momentum'] for s in sectors]
    roes = [sector_stats[s]['roe'] for s in sectors]
    margins = [sector_stats[s]['margin'] for s in sectors]
    pes = [sector_stats[s]['pe'] for s in sectors]

    # Rank percentile for each metric
    momentum_ranks = pd.Series(momentums, index=sectors).rank(pct=True)
    roe_ranks = pd.Series(roes, index=sectors).rank(pct=True)
    margin_ranks = pd.Series(margins, index=sectors).rank(pct=True)
    pe_ranks = pd.Series(pes, index=sectors).rank(pct=True, ascending=False)  # Lower P/E is better

    # Compute composite sector score (60% momentum, 40% fundamentals)
    sector_scores = {}
    for sector in sectors:
        momentum_score = momentum_ranks[sector]
        fundamental_score = (
            0.40 * roe_ranks[sector] +
            0.30 * margin_ranks[sector] +
            0.30 * pe_ranks[sector]
        )

        # Composite: 60% momentum, 40% fundamentals
        sector_scores[sector] = 0.60 * momentum_score + 0.40 * fundamental_score

    return sector_scores


def apply_sector_rotation(df: pd.DataFrame, sector_scores: Dict[str, float], params: Dict) -> pd.DataFrame:
    """Apply sector rotation multiplier to position weights.

    Strong sectors get higher allocations, weak sectors get lower allocations.
    """
    if not params.get('enable_sector_rotation', True):
        return df

    if not sector_scores or 'sector' not in df.columns:
        return df

    # Rotation strength parameter (0 = no rotation, 1 = max rotation)
    rotation_strength = params.get('sector_rotation_strength', 0.30)

    # Map sector scores to multipliers
    # Strong sectors (score > 0.7) get 1.0 + rotation_strength
    # Weak sectors (score < 0.3) get 1.0 - rotation_strength
    # Average sectors (0.3-0.7) get 1.0

    def sector_multiplier(sector):
        score = sector_scores.get(sector, 0.5)  # Default to neutral if sector not scored

        if score >= 0.7:
            # Strong sector: boost allocation
            return 1.0 + rotation_strength
        elif score <= 0.3:
            # Weak sector: reduce allocation
            return 1.0 - rotation_strength
        else:
            # Neutral sector: slight adjustment based on exact score
            # Linear interpolation between weak and strong
            normalized = (score - 0.3) / 0.4  # Map 0.3-0.7 to 0-1
            return 1.0 + rotation_strength * (2 * normalized - 1)

    df['sector_multiplier'] = df['sector'].apply(sector_multiplier)
    df['sector_score'] = df['sector'].map(sector_scores).fillna(0.5)

    return df


# =====================================================================
# Filtering
# =====================================================================

def apply_filters(df: pd.DataFrame, params: Dict, debug_log: Optional = None) -> pd.DataFrame:
    """Apply liquidity, quality, valuation, and momentum filters."""

    def log(msg):
        if debug_log:
            debug_log(msg)

    initial_count = len(df)
    log(f"  Initial candidates: {initial_count}")
    filter_counts = [('initial', initial_count)]

    # Liquidity filters
    prev = len(df)
    df = df[df['dollar_volume'] >= params.get('min_dollar_volume', 5_000_000)]
    filter_counts.append(('dollar_volume', len(df)))
    log(f"  After dollar_volume filter: {len(df)} (removed {prev - len(df)})")

    prev = len(df)
    df = df[(df['market_cap'] >= params.get('min_market_cap', 500_000_000)) | df['market_cap'].isna()]
    filter_counts.append(('market_cap', len(df)))
    log(f"  After market_cap filter: {len(df)} (removed {prev - len(df)})")

    # Quality filters (allow NaN to pass through)
    prev = len(df)
    df = df[(df['roe'] >= params.get('min_roe', 0.10)) | df['roe'].isna()]
    filter_counts.append(('roe', len(df)))
    log(f"  After ROE filter: {len(df)} (removed {prev - len(df)})")

    prev = len(df)
    df = df[(df['profit_margins'] >= params.get('min_profit_margin', 0.05)) | df['profit_margins'].isna()]
    filter_counts.append(('profit_margins', len(df)))
    log(f"  After profit_margins filter: {len(df)} (removed {prev - len(df)})")

    prev = len(df)
    df = df[(df['debt_to_equity'] < params.get('max_debt_to_equity', 2.0)) | df['debt_to_equity'].isna()]
    filter_counts.append(('debt_to_equity', len(df)))
    log(f"  After debt_to_equity filter: {len(df)} (removed {prev - len(df)})")

    # Valuation filters (allow NaN to pass through)
    prev = len(df)
    df = df[(df['trailing_pe'] > 0) & (df['trailing_pe'] < params.get('max_pe', 50)) | df['trailing_pe'].isna()]
    filter_counts.append(('trailing_pe', len(df)))
    log(f"  After trailing_pe filter: {len(df)} (removed {prev - len(df)})")

    prev = len(df)
    df = df[(df['price_to_book'] > 0) & (df['price_to_book'] < params.get('max_pb', 10)) | df['price_to_book'].isna()]
    filter_counts.append(('price_to_book', len(df)))
    log(f"  After price_to_book filter: {len(df)} (removed {prev - len(df)})")

    # Momentum filters (allow NaN values to pass through)
    momentum_6m_min = params.get('momentum_6m_min', -0.10)
    momentum_3m_min = params.get('momentum_3m_min', -0.15)
    prev = len(df)
    df = df[(df['ret_6m'] >= momentum_6m_min) | df['ret_6m'].isna()]
    filter_counts.append(('ret_6m', len(df)))
    log(f"  After ret_6m filter: {len(df)} (removed {prev - len(df)})")

    prev = len(df)
    df = df[(df['ret_3m'] >= momentum_3m_min) | df['ret_3m'].isna()]
    filter_counts.append(('ret_3m', len(df)))
    log(f"  After ret_3m filter: {len(df)} (removed {prev - len(df)})")

    # Volatility filter
    prev = len(df)
    df = df[df['volatility'] < params.get('max_volatility', 0.80)]
    filter_counts.append(('volatility', len(df)))
    log(f"  After volatility filter: {len(df)} (removed {prev - len(df)})")

    log("  Filter progression: " + " -> ".join([f"{name}:{count}" for name, count in filter_counts]))

    return df


# =====================================================================
# Portfolio Construction
# =====================================================================

def compute_position_weights(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Compute position weights using inverse volatility with quality and sector boosts."""

    # Inverse volatility weighting
    df['inv_vol'] = 1 / (df['volatility'] + 0.01)

    # Quality multiplier based on composite score
    composite_80 = df['composite'].quantile(0.80)
    composite_20 = df['composite'].quantile(0.20)

    def quality_mult_fn(x):
        if x >= composite_80:
            return 1.2
        elif x <= composite_20:
            return 0.8
        return 1.0

    df['quality_mult'] = df['composite'].apply(quality_mult_fn)

    # Sector multiplier (already computed by apply_sector_rotation)
    if 'sector_multiplier' not in df.columns:
        df['sector_multiplier'] = 1.0

    # Raw weight with sector rotation
    df['raw_weight'] = df['inv_vol'] * df['quality_mult'] * df['sector_multiplier']

    # Normalize to sum to 1.0
    df['position_weight'] = df['raw_weight'] / df['raw_weight'].sum()

    # Cap individual positions
    position_cap = params.get('position_weight_cap', 0.10)
    df['position_weight'] = df['position_weight'].clip(upper=position_cap)

    # Re-normalize after capping
    df['position_weight'] = df['position_weight'] / df['position_weight'].sum()

    return df


def apply_sector_diversification(
    df: pd.DataFrame, target_positions: int, params: Dict, debug_log=None
) -> pd.DataFrame:
    """Select stocks respecting dynamic sector concentration limits.

    Sector limits are dynamic based on sector scores:
    - Strong sectors (score > 0.7): Can reach sector_max_tilt (e.g., 45%)
    - Weak sectors (score < 0.3): Limited to sector_min_tilt (e.g., 15%)
    - Average sectors (0.3-0.7): Base sector_max_pct (e.g., 35%)
    """
    def log(msg):
        if debug_log:
            debug_log(msg)

    selected = []
    sector_weights = {}
    log(f"  apply_sector_diversification: {len(df)} candidates, target={target_positions}")

    # Get dynamic sector limits
    base_max = params.get('sector_max_pct', 0.35)
    max_tilt = params.get('sector_max_tilt', 0.45)
    min_tilt = params.get('sector_min_tilt', 0.15)
    enable_rotation = params.get('enable_sector_rotation', True)

    # Compute sector-specific limits
    sector_limits = {}
    if enable_rotation and 'sector_score' in df.columns:
        for sector in df['sector'].unique():
            # Handle None or 'Unknown' sectors - use base limit
            if pd.isna(sector) or sector is None or sector == 'Unknown':
                sector_limits[sector] = base_max
                continue

            sector_score = df[df['sector'] == sector]['sector_score'].iloc[0]

            if sector_score >= 0.7:
                # Strong sector: allow higher concentration
                sector_limits[sector] = max_tilt
            elif sector_score <= 0.3:
                # Weak sector: restrict concentration
                sector_limits[sector] = min_tilt
            else:
                # Average sector: use base limit
                sector_limits[sector] = base_max
    else:
        # No sector rotation: use base limit for all sectors
        for sector in df['sector'].unique():
            sector_limits[sector] = base_max

    log(f"  Sector limits: {sector_limits}")
    df_sorted = df.sort_values('composite', ascending=False)

    for idx, row in df_sorted.iterrows():
        sector = row['sector']
        candidate_weight = row['position_weight']
        current_sector_weight = sector_weights.get(sector, 0.0)
        sector_limit = sector_limits.get(sector, base_max)

        log(f"    Evaluating sec_id={row['security_id']}, sector={sector}, weight={candidate_weight:.4f}, current_sector_weight={current_sector_weight:.4f}, limit={sector_limit:.4f}")

        # Check sector-specific limit, but allow at least target_positions to be selected
        # This prevents empty portfolios when position weights exceed sector limits
        if current_sector_weight + candidate_weight <= sector_limit or len(selected) < target_positions:
            selected.append(row)
            sector_weights[sector] = current_sector_weight + candidate_weight
            log(f"      ACCEPTED - new sector weight: {sector_weights[sector]:.4f}")

            if len(selected) >= target_positions:
                log(f"      Reached target positions ({target_positions})")
                break
        else:
            log(f"      REJECTED - would exceed limit ({current_sector_weight + candidate_weight:.4f} > {sector_limit:.4f})")

    log(f"  Selected {len(selected)} candidates")
    return pd.DataFrame(selected) if selected else pd.DataFrame()


# =====================================================================
# Options Strategy
# =====================================================================

def get_current_vix(market_features: List[Dict]) -> Optional[float]:
    """Extract current VIX level from market features."""
    if not market_features:
        return None

    try:
        df = pd.DataFrame(market_features)
        df['date'] = pd.to_datetime(df['date'])
        latest = df.sort_values('date').iloc[-1]
        return latest.get('vix', None)
    except Exception:
        return None


def determine_options_strategy(
    row: pd.Series, vix: Optional[float], params: Dict
) -> Optional[Dict]:
    """Determine covered call or protective put strategy."""

    iv_rank = row.get('iv_rank', np.nan)
    options_activity = row.get('options_activity', 0)

    # Skip if insufficient options liquidity
    min_oi = params.get('min_options_oi', 100)
    if pd.isna(options_activity) or options_activity < min_oi:
        return None

    # Covered call: high IV rank
    cc_threshold = params.get('covered_call_iv_threshold', 60)
    if not pd.isna(iv_rank) and iv_rank > cc_threshold:
        return {
            'strategy': 'covered_call',
            'strike_pct': 1.08,  # 8% OTM
            'dte_target': 35,
            'trigger': 'iv_rank'
        }

    # Protective put: high VIX
    vix_threshold = params.get('protective_put_vix_threshold', 25)
    if vix and vix > vix_threshold:
        return {
            'strategy': 'protective_put',
            'strike_pct': 0.92,  # 8% OTM
            'dte_target': 75,
            'trigger': 'volatility_regime'
        }

    # Protective put: low IV rank (cheap puts)
    pp_iv_threshold = params.get('protective_put_iv_threshold', 30)
    if not pd.isna(iv_rank) and iv_rank < pp_iv_threshold:
        return {
            'strategy': 'protective_put',
            'strike_pct': 0.90,  # 10% OTM
            'dte_target': 90,
            'trigger': 'iv_rank'
        }

    return None


# =====================================================================
# Signal Generation
# =====================================================================

def make_signal(row: pd.Series, params: Dict) -> Dict:
    """Generate signal from selected candidate."""

    # Build metadata
    meta = {
        # Factor scores
        'composite': float(row['composite']),
        'value_score': float(row.get('value_score', 0)),
        'momentum_score': float(row.get('momentum_score', 0)),
        'quality_score': float(row.get('quality_score', 0)),
        'risk_score': float(row.get('risk_score', 0)),
        'options_score': float(row.get('options_score', 0)),

        # Underlying metrics
        'trailing_pe': float(row.get('trailing_pe', np.nan)),
        'price_to_book': float(row.get('price_to_book', np.nan)),
        'roe': float(row.get('roe', np.nan)),
        'fcf_yield': float(row.get('fcf_yield', np.nan)),
        'ret_12m': float(row.get('ret_12m', np.nan)),
        'ret_6m': float(row.get('ret_6m', np.nan)),
        'ret_3m': float(row.get('ret_3m', np.nan)),
        'volatility': float(row.get('volatility', np.nan)),

        # Portfolio construction
        'position_weight': float(row.get('position_weight', 0)),
        'sector': str(row.get('sector', 'Unknown')),
        'sector_score': float(row.get('sector_score', 0.5)),
        'sector_multiplier': float(row.get('sector_multiplier', 1.0)),
        'sector_limit': float(row.get('sector_limit', 0.35)),

        # Risk management
        'stop_loss_pct': float(params.get('stop_loss_pct', -0.20)),
        'trailing_stop_trigger': float(params.get('trailing_stop_trigger', 0.15)),
        'trailing_stop_pct': float(params.get('trailing_stop_pct', 0.10)),
    }

    # Add options strategy if applicable
    if 'options_strategy' in row and row['options_strategy']:
        strategy = row['options_strategy']
        meta['options_strategy'] = strategy['strategy']
        meta[f"{strategy['strategy']}_strike_pct"] = strategy['strike_pct']
        meta[f"{strategy['strategy']}_dte_target"] = strategy['dte_target']
        meta[f"{strategy['strategy']}_trigger"] = strategy['trigger']

    # Add options metrics if available
    if 'iv_rank' in row and pd.notna(row['iv_rank']):
        meta['iv_rank'] = float(row['iv_rank'])
    if 'avg_iv' in row and pd.notna(row['avg_iv']):
        meta['avg_iv'] = float(row['avg_iv'])
    if 'put_call_oi_ratio' in row and pd.notna(row['put_call_oi_ratio']):
        meta['put_call_oi_ratio'] = float(row['put_call_oi_ratio'])
    if 'options_activity' in row and pd.notna(row['options_activity']):
        meta['options_activity'] = float(row['options_activity'])

    # Compute signal strength and confidence
    max_composite = row.get('_max_composite', row['composite'])
    strength = float(row['composite'] / max_composite) if max_composite > 0 else 0.5

    volatility = row.get('volatility', 0.30)
    confidence = max(0.2, min(0.95, 1 / (1 + volatility))) if not pd.isna(volatility) else 0.5

    signal = {
        'timestamp': row['timestamp'],
        'security_id': int(row['security_id']),
        'signal_type': 'long',
        'strength': strength,
        'confidence': confidence,
        'meta': meta
    }

    return signal


# =====================================================================
# Main Entry Point
# =====================================================================

def main() -> None:
    """Main model execution logic."""

    # DEBUG: File-based logging
    debug_log = Path(__file__).parent / "debug.log"
    def log(msg: str):
        with open(debug_log, "a") as f:
            f.write(f"{msg}\n")
            f.flush()

    log("=" * 80)
    log("DEBUG: Model started")

    # 1. Load input payload
    payload = json.load(sys.stdin)
    log(f"DEBUG: Loaded payload with {len(payload.get('data', {}).get('prices', []))} price payloads")
    model_id = payload.get('model_id', 'value_momentum_options_v1')
    run_id = payload.get('run_id', '')
    params = {**DEFAULT_PARAMS, **(payload.get('parameters', {}) or {})}
    log(f"DEBUG: Parameters: target_positions={params.get('target_positions')}, min_market_cap={params.get('min_market_cap')}")

    # 2. Extract data from payload
    prices_data = payload.get('data', {}).get('prices', [])
    options_data = payload.get('data', {}).get('options', {})
    market_features = payload.get('data', {}).get('market_features', [])

    # 3. Load fundamentals and securities metadata
    security_ids = [p['security_id'] for p in prices_data]
    log(f"DEBUG: Loading fundamentals for {len(security_ids)} securities")
    fundamentals_df = load_fundamentals(security_ids)
    log(f"DEBUG: Loaded {len(fundamentals_df)} fundamental records")
    securities_df = load_securities(security_ids)
    log(f"DEBUG: Loaded {len(securities_df)} security records")

    # 4. Build candidates
    candidates = []
    log(f"DEBUG: Processing {len(prices_data)} price payloads")
    for price_payload in prices_data:
        sec_id = price_payload['security_id']

        # Build price DataFrame
        price_df = build_price_dataframe(price_payload)
        if price_df.empty:
            log(f"DEBUG: Sec {sec_id}: Empty price_df - SKIPPED")
            continue

        log(f"DEBUG: Sec {sec_id}: {len(price_df)} price records")

        # Get fundamentals (if available)
        fund_row = fundamentals_df.loc[sec_id] if sec_id in fundamentals_df.index else None
        log(f"DEBUG: Sec {sec_id}: fundamentals={'present' if fund_row is not None else 'missing'}")

        # Get options metrics
        sec_options = options_data.get(str(sec_id), [])
        options_metrics = compute_options_metrics(sec_id, sec_options)

        # Get sector
        sector = securities_df.loc[sec_id, 'sector'] if sec_id in securities_df.index else None
        log(f"DEBUG: Sec {sec_id}: sector={sector}")

        # Build candidate
        candidate = build_candidate(sec_id, price_df, fund_row, options_metrics, sector)
        if candidate:
            candidates.append(candidate)
            log(f"DEBUG: Sec {sec_id}: Candidate built successfully")
        else:
            log(f"DEBUG: Sec {sec_id}: Candidate building returned None")

    log(f"DEBUG: Total candidates built: {len(candidates)}")

    if not candidates:
        # No valid candidates, return empty signals
        log("DEBUG: No candidates, exiting with empty signals")
        output = {'model_id': model_id, 'run_id': run_id, 'signals': []}
        json.dump(output, sys.stdout)
        return

    candidates_df = pd.DataFrame(candidates)
    log(f"DEBUG: Candidates DataFrame shape: {candidates_df.shape}")

    # Log first candidate details
    if len(candidates_df) > 0:
        first = candidates_df.iloc[0]
        log(f"DEBUG: First candidate values:")
        log(f"  security_id={first['security_id']}, market_cap={first.get('market_cap', 'N/A')}, dollar_volume={first.get('dollar_volume', 'N/A')}")
        log(f"  roe={first.get('roe', 'N/A')}, profit_margins={first.get('profit_margins', 'N/A')}, debt_to_equity={first.get('debt_to_equity', 'N/A')}")
        log(f"  trailing_pe={first.get('trailing_pe', 'N/A')}, price_to_book={first.get('price_to_book', 'N/A')}")
        log(f"  ret_3m={first.get('ret_3m', 'N/A')}, ret_6m={first.get('ret_6m', 'N/A')}, ret_12m={first.get('ret_12m', 'N/A')}")
        log(f"  volatility={first.get('volatility', 'N/A')}")

    # 5. Apply filters
    log("DEBUG: Applying filters")
    filtered = apply_filters(candidates_df, params, debug_log=log)
    log(f"DEBUG: After filters: {len(filtered)} candidates remain")

    if filtered.empty:
        # All candidates filtered out
        log("DEBUG: All candidates filtered out, exiting with empty signals")
        output = {'model_id': model_id, 'run_id': run_id, 'signals': []}
        json.dump(output, sys.stdout)
        return

    # 6. Compute factor scores
    log("DEBUG: Computing factor scores")
    filtered['value_score'] = compute_value_score(filtered)
    filtered['momentum_score'] = compute_momentum_score(filtered)
    filtered['quality_score'] = compute_quality_score(filtered)
    filtered['risk_score'] = compute_risk_score(filtered)
    filtered['options_score'] = compute_options_score(filtered)
    log(f"DEBUG: Factor scores computed: value={filtered['value_score'].iloc[0]:.3f}, momentum={filtered['momentum_score'].iloc[0]:.3f}, quality={filtered['quality_score'].iloc[0]:.3f}, risk={filtered['risk_score'].iloc[0]:.3f}, options={filtered['options_score'].iloc[0]:.3f}")

    # 7. Compute composite score
    filtered['composite'] = sum(
        filtered[factor] * weight for factor, weight in FACTOR_WEIGHTS.items()
    )
    log(f"DEBUG: Composite score computed: {filtered['composite'].iloc[0]:.3f}")

    # 8. Select top candidates (double the target for diversification buffer)
    target_positions = params.get('target_positions', 20)
    filtered = filtered.sort_values('composite', ascending=False).head(target_positions * 2)
    log(f"DEBUG: After selecting top {target_positions * 2}: {len(filtered)} candidates")

    # 8.5. Compute sector scores for rotation
    sector_scores = compute_sector_scores(filtered, params)
    log(f"DEBUG: Sector scores: {sector_scores}")

    # 8.6. Apply sector rotation multipliers
    filtered = apply_sector_rotation(filtered, sector_scores, params)
    log(f"DEBUG: After sector rotation: {len(filtered)} candidates")

    # 9. Compute position weights (now includes sector multiplier)
    filtered = compute_position_weights(filtered, params)
    log(f"DEBUG: After position weighting: {len(filtered)} candidates")

    # 10. Apply sector diversification with dynamic limits
    selected = apply_sector_diversification(filtered, target_positions, params, debug_log=log)
    log(f"DEBUG: After sector diversification: {len(selected)} candidates selected")

    # Portfolio size vs target + position sizing diagnostics
    actual_positions = len(selected)
    shortfall = max(0, target_positions - actual_positions)
    log(f"DEBUG: Portfolio size check: target={target_positions}, selected={actual_positions}, shortfall={shortfall}")

    if not selected.empty:
        weights = selected['position_weight']
        log(
            "DEBUG: Position weight stats: "
            f"min={weights.min():.4f}, max={weights.max():.4f}, "
            f"mean={weights.mean():.4f}, sum={weights.sum():.4f}"
        )

        top_weights = ", ".join(
            f"{int(row.security_id)}@{row.position_weight:.4f}"
            for _, row in selected.sort_values('position_weight', ascending=False).head(10).iterrows()
        )
        log(f"DEBUG: Top position weights (sec_id@weight): {top_weights}")
    else:
        log("DEBUG: No positions selected to size; skipping weight diagnostics")

    if selected.empty:
        # Sector diversification eliminated all candidates
        log("DEBUG: Sector diversification eliminated all candidates, exiting with empty signals")
        output = {'model_id': model_id, 'run_id': run_id, 'signals': []}
        json.dump(output, sys.stdout)
        return

    # 11. Determine options strategies
    log("DEBUG: Determining options strategies")
    vix = get_current_vix(market_features)
    selected['options_strategy'] = selected.apply(
        lambda row: determine_options_strategy(row, vix, params), axis=1
    )

    # 12. Store max composite for strength calculation
    max_composite = selected['composite'].max()
    selected['_max_composite'] = max_composite

    # 12.5 Trade frequency vs prior holdings (if provided)
    prior_positions = (
        payload.get('current_positions')
        or payload.get('existing_positions')
        or payload.get('positions')
    )

    if not prior_positions and isinstance(payload.get('portfolio'), dict):
        prior_positions = payload['portfolio'].get('positions')

    prior_ids = set()
    if isinstance(prior_positions, list):
        for pos in prior_positions:
            sid = None
            if isinstance(pos, dict):
                sid = pos.get('security_id') or pos.get('id')
            else:
                sid = pos

            try:
                if sid is not None:
                    prior_ids.add(int(sid))
            except (TypeError, ValueError):
                continue

    if prior_ids:
        new_ids = set(selected['security_id'].astype(int))
        buys = new_ids - prior_ids
        sells = prior_ids - new_ids
        holds = prior_ids & new_ids
        log(
            f"DEBUG: Trade frequency vs prior: prev={len(prior_ids)}, "
            f"holds={len(holds)}, buys={len(buys)}, sells={len(sells)}"
        )
    else:
        log("DEBUG: Trade frequency: no prior positions provided; logging new selection count only")

    # 13. Generate signals
    log(f"DEBUG: Generating signals for {len(selected)} selected securities")
    signals = [make_signal(row, params) for _, row in selected.iterrows()]
    log(f"DEBUG: Generated {len(signals)} signals")

    # 14. Output
    log(f"DEBUG: Outputting {len(signals)} signals")
    output = {
        'model_id': model_id,
        'run_id': run_id,
        'signals': signals
    }
    json.dump(output, sys.stdout)
    log("DEBUG: Model completed successfully")


if __name__ == '__main__':
    main()
