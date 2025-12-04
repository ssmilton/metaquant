# Value-Momentum-Options Hybrid Model v1

**Author**: shane
**Model ID**: `value_momentum_options_v1`
**Version**: 1.0.0
**Type**: Aggressive multi-factor equity model with options overlay

## Strategy Overview

This model combines five complementary factors to identify high-conviction long positions with options-enhanced risk/return characteristics:

### Core Factors (Weighted)
1. **Value (25%)**: P/E ratio, P/B ratio, FCF yield
2. **Momentum (25%)**: 3-month, 6-month, 12-month price returns
3. **Quality (20%)**: ROE, profit margins, ROA, revenue growth, leverage
4. **Risk (10%)**: Inverse volatility (prefer stable stocks)
5. **Options (20%)**: IV rank, options liquidity, put/call sentiment

### Options Overlay Strategy

The model identifies opportunities to enhance returns and manage risk through options:

**Covered Calls** (Sell):
- **Trigger**: IV rank > 60 (expensive implied volatility)
- **Strike**: 8% out-of-the-money (OTM)
- **Expiration**: 30-45 days to expiration (DTE)
- **Purpose**: Generate income on holdings when options premiums are attractive

**Protective Puts** (Buy):
- **Trigger 1**: VIX > 25 (high market volatility regime)
- **Trigger 2**: IV rank < 30 (cheap implied volatility)
- **Strike**: 8-10% OTM
- **Expiration**: 60-90 DTE
- **Purpose**: Hedge downside risk when protection is inexpensive

### Sector Rotation Strategy

The model dynamically allocates to sectors based on their relative strength:

**Sector Scoring** (computed across all candidates):
- **Momentum (60%)**: Sector-average returns (3m, 6m, 12m)
- **Fundamentals (40%)**: Sector-average ROE, profit margins, inverted P/E

**Sector Multipliers** (applied to position weights):
- **Strong sectors** (score ≥ 0.7): 1.0 + rotation_strength (default: 1.3x weight)
- **Average sectors** (0.3 < score < 0.7): 1.0x weight (linear interpolation)
- **Weak sectors** (score ≤ 0.3): 1.0 - rotation_strength (default: 0.7x weight)

**Dynamic Sector Limits**:
- **Strong sectors**: Maximum 45% allocation (vs base 35%)
- **Average sectors**: Maximum 35% allocation (base limit)
- **Weak sectors**: Maximum 15% allocation (restricted)

**Rationale**: Tilts portfolio toward sectors with strong momentum and fundamentals while maintaining diversification. This tactical overlay enhances returns during sector-driven market regimes.

### Portfolio Construction

- **Target Positions**: 15-30 stocks
- **Weighting**: Inverse volatility with quality and sector boosts (top 20% quality get 1.2x, strong sectors get 1.3x)
- **Position Cap**: 10% maximum single position
- **Sector Limits**: Dynamic (15-45%) based on sector strength scores
- **Rebalancing**: Monthly (satisfies 30-day minimum hold period)

### Risk Management

**Position-Level**:
- Hard stop loss: -20% from entry
- Trailing stop: 10% after +15% gain

**Portfolio-Level**:
- Target volatility: 18-22% annualized
- Maximum drawdown target: <35%
- Sharpe ratio target: >0.8

---

## Performance Objectives

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 0.8 |
| Max Drawdown | < 35% |
| Annualized Return | SPY + 3-5% |
| Win Rate | > 55% |
| Avg Holding Period | 30-90 days |

**Benchmark**: S&P 500 (SPY)

---

## Factor Methodology

### 1. Value Score

**Metrics**:
- Trailing P/E ratio (40% weight): Lower is better, inverted z-score
- Price-to-Book ratio (30% weight): Lower is better, inverted z-score
- FCF Yield (30% weight): Free Cash Flow / Market Cap, higher is better

**Filters**:
- P/E ratio: 0 < P/E < 50
- P/B ratio: 0 < P/B < 10
- Market cap: > $500M

**Rationale**: Identifies companies trading below intrinsic value with strong cash generation.

### 2. Momentum Score

**Metrics**:
- 12-month return (25% weight): Long-term trend
- 6-month return (50% weight): Primary momentum signal
- 3-month return (25% weight): Recent acceleration

**Filters**:
- 6-month return: > -10% (avoid falling knives)
- 3-month return: > -15% (allow minor pullbacks)

**Rationale**: Captures persistent price trends while filtering out weak names.

### 3. Quality Score

**Metrics**:
- Return on Equity (35% weight): Profitability on shareholder capital
- Profit Margins (25% weight): Operating efficiency
- Return on Assets (15% weight): Asset utilization
- Revenue Growth (15% weight): Prefer 5-30% annual growth
- Debt-to-Equity (10% weight): Financial health, lower is better

**Filters**:
- ROE: > 10%
- Profit margins: > 5%
- Debt/Equity: < 2.0

**Rationale**: Selects companies with sustainable competitive advantages and strong fundamentals.

### 4. Risk Score

**Metric**: Inverse volatility (1 / (1 + annualized volatility))

**Filter**: Volatility < 80% annualized

**Rationale**: Lower volatility stocks provide more stable returns and better risk-adjusted performance.

### 5. Options Score

**Metrics**:
- IV Rank (50% weight): Prefer moderate IV (40-70 percentile) for balanced options opportunities
- Options Activity (35% weight): Higher open interest indicates liquid options market
- Put/Call Ratio (15% weight): Neutral to bullish sentiment preferred (0.8-1.2)

**Filter**: Minimum open interest > 100 contracts

**Rationale**: Identifies stocks with active, liquid options markets suitable for overlay strategies.

---

## Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_positions` | 20 | Target number of positions (15-30 range) |
| `min_market_cap` | 500000000 | Minimum market cap ($500M) |
| `min_dollar_volume` | 5000000 | Minimum daily dollar volume ($5M) |
| `max_volatility` | 0.80 | Maximum annualized volatility (80%) |

### Quality Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_roe` | 0.10 | Minimum ROE (10%) |
| `max_pe` | 50 | Maximum P/E ratio |
| `max_pb` | 10 | Maximum P/B ratio |
| `min_profit_margin` | 0.05 | Minimum profit margin (5%) |
| `max_debt_to_equity` | 2.0 | Maximum leverage |

### Momentum Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum_6m_min` | -0.10 | Minimum 6-month return (-10%) |
| `momentum_3m_min` | -0.15 | Minimum 3-month return (-15%) |

### Portfolio Construction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sector_max_pct` | 0.35 | Base maximum sector concentration (35%) |
| `position_weight_cap` | 0.10 | Maximum single position weight (10%) |

### Sector Rotation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_sector_rotation` | true | Enable sector rotation overlay |
| `sector_rotation_strength` | 0.30 | Sector tilt strength (0=none, 1=max) |
| `sector_lookback_months` | 3 | Lookback period for sector momentum |
| `sector_max_tilt` | 0.45 | Maximum allocation for strong sectors |
| `sector_min_tilt` | 0.15 | Minimum allocation for weak sectors |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_loss_pct` | -0.20 | Hard stop loss (-20%) |
| `trailing_stop_trigger` | 0.15 | Activate trailing stop at +15% gain |
| `trailing_stop_pct` | 0.10 | Trailing stop percentage (10%) |

### Options Strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `covered_call_iv_threshold` | 60 | IV rank threshold for covered calls |
| `protective_put_vix_threshold` | 25 | VIX threshold for protective puts |
| `protective_put_iv_threshold` | 30 | IV rank threshold for protective puts |
| `min_options_oi` | 100 | Minimum options open interest |

---

## Usage

### Quick Start

```bash
# Run backtest with default parameters (20 positions)
quantfw backtest \
  --model-id value_momentum_options_v1 \
  --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,NFLX,CRM,ADBE \
  --start-date 2020-01-01 \
  --end-date 2023-12-31
```

### With Custom Parameters

```bash
# More aggressive: 30 positions, lower quality bar
quantfw backtest \
  --model-id value_momentum_options_v1 \
  --tickers-file data/sp500.txt \
  --start-date 2015-01-01 \
  --end-date 2024-12-31 \
  --params '{"target_positions": 30, "min_roe": 0.08, "min_market_cap": 250000000}'
```

### Full Universe Backtest

```bash
# All 5,000+ tickers with parameters file
quantfw backtest \
  --model-id value_momentum_options_v1 \
  --tickers-file data/tickers_list.txt \
  --start-date 2010-01-01 \
  --end-date 2024-12-31 \
  --params-file params.json
```

---

## Signal Output

Each signal includes comprehensive metadata for analysis:

```json
{
  "timestamp": "2023-10-15",
  "security_id": 101,
  "signal_type": "long",
  "strength": 0.85,
  "confidence": 0.80,
  "meta": {
    "composite": 0.72,
    "value_score": 0.65,
    "momentum_score": 0.80,
    "quality_score": 0.70,
    "risk_score": 0.75,
    "options_score": 0.68,
    "trailing_pe": 15.2,
    "price_to_book": 3.1,
    "roe": 0.22,
    "fcf_yield": 0.045,
    "ret_12m": 0.28,
    "ret_6m": 0.18,
    "ret_3m": 0.09,
    "volatility": 0.28,
    "position_weight": 0.06,
    "sector": "Technology",
    "sector_score": 0.75,
    "sector_multiplier": 1.25,
    "sector_limit": 0.42,
    "stop_loss_pct": -0.20,
    "trailing_stop_trigger": 0.15,
    "trailing_stop_pct": 0.10,
    "options_strategy": "covered_call",
    "covered_call_strike_pct": 1.08,
    "covered_call_dte_target": 35,
    "covered_call_trigger": "iv_rank",
    "iv_rank": 68.5,
    "avg_iv": 0.32,
    "put_call_oi_ratio": 0.95,
    "options_activity": 15000
  }
}
```

---

## Data Requirements

### Minimum Data

**Daily Prices** (required):
- OHLCV data for at least 252 trading days (1 year)
- Needed for momentum and volatility calculations

**Fundamentals** (recommended):
- Trailing P/E, forward P/E, price-to-book
- ROE, ROA, profit margins
- Revenue growth, earnings growth
- Free cash flow, debt-to-equity
- Market cap

**Options Quotes** (optional, enhances strategy):
- Bid, ask, last prices
- Implied volatility
- Open interest
- Delta (for call/put classification)

### Data Ingestion

```bash
# Ingest equity prices and fundamentals
python scripts/fetch_yfinance_equities.py \
  --tickers-file data/tickers_list.txt \
  --start-date 2010-01-01 \
  --end-date 2024-12-31 \
  --rate-limit-delay 0.5

# Ingest options data (subset of liquid tickers)
python scripts/fetch_yfinance_options.py \
  --tickers-file data/liquid_tickers.txt \
  --days-to-expiry-min 7 \
  --days-to-expiry-max 90
```

---

## Model Behavior

### Graceful Degradation

The model handles missing data gracefully:

- **No fundamentals**: Uses neutral scores (0) for value and quality factors
- **No options data**: Uses neutral score (0.5) for options factor
- **Insufficient price history**: Skips candidate entirely
- **Missing individual metrics**: Fills with NaN, excluded from scoring

### Factor Contribution

Expected contribution to composite score by market regime:

| Regime | Value | Momentum | Quality | Risk | Options |
|--------|-------|----------|---------|------|---------|
| **Bull Market** | Low | High | Medium | Low | Medium |
| **Bear Market** | High | Low | High | High | Medium |
| **High Volatility** | Medium | Low | High | High | High |
| **Low Volatility** | Medium | High | Medium | Low | Low |

### Sector Exposure

Typical sector concentrations (will vary by universe):
- Technology: 20-35%
- Healthcare: 10-20%
- Financials: 10-15%
- Consumer: 10-15%
- Industrials: 5-10%
- Other: 10-20%

---

## Tuning Guidelines

### Increasing Aggressiveness

To increase returns (at cost of higher drawdown):
- Increase `target_positions` to 30
- Decrease `min_roe` to 0.08
- Decrease `min_market_cap` to $250M (smaller caps)
- Increase `sector_max_pct` to 0.40
- Increase `sector_rotation_strength` to 0.40-0.50 (stronger sector bets)

### Increasing Conservatism

To reduce drawdown (at cost of lower returns):
- Decrease `target_positions` to 15
- Increase `min_roe` to 0.12
- Increase `min_market_cap` to $1B
- Decrease `max_volatility` to 0.60
- Decrease `sector_max_pct` to 0.30
- Decrease `sector_rotation_strength` to 0.15-0.20 (more sector-neutral)

### Factor Weight Tuning

Edit `FACTOR_WEIGHTS` in `run_model.py`:

```python
# More value-focused
FACTOR_WEIGHTS = {
    'value_score': 0.35,     # +10%
    'momentum_score': 0.20,  # -5%
    'quality_score': 0.25,   # +5%
    'risk_score': 0.10,
    'options_score': 0.10,   # -10%
}

# More momentum-focused
FACTOR_WEIGHTS = {
    'value_score': 0.15,     # -10%
    'momentum_score': 0.35,  # +10%
    'quality_score': 0.20,
    'risk_score': 0.10,
    'options_score': 0.20,
}
```

### Sector Rotation Tuning

Adjust sector rotation parameters to control tactical sector allocation:

**Disable Sector Rotation** (pure stock selection):
```json
{
  "enable_sector_rotation": false
}
```

**Conservative Sector Tilt** (15-20% strength):
```json
{
  "sector_rotation_strength": 0.15,
  "sector_max_tilt": 0.40,
  "sector_min_tilt": 0.20
}
```

**Moderate Sector Tilt** (default 30%):
```json
{
  "sector_rotation_strength": 0.30,
  "sector_max_tilt": 0.45,
  "sector_min_tilt": 0.15
}
```

**Aggressive Sector Tilt** (40-50% strength):
```json
{
  "sector_rotation_strength": 0.50,
  "sector_max_tilt": 0.50,
  "sector_min_tilt": 0.10,
  "sector_lookback_months": 1
}
```

**Tuning Notes**:
- Higher `sector_rotation_strength` creates larger sector bets (higher risk/return)
- Shorter `sector_lookback_months` makes sector rotation more responsive to recent trends
- Wider gap between `sector_max_tilt` and `sector_min_tilt` increases concentration in strong sectors
- In low-correlation regimes, sector rotation may underperform stock selection alone

---

## Known Limitations

1. **Fundamentals Lag**: Fundamentals are reported quarterly with delays; model uses most recent available data
2. **Options Greeks**: Greeks (delta, gamma, theta, vega) may be NULL from yfinance; model uses delta to classify calls vs puts
3. **Survivorship Bias**: Backtests may suffer from survivorship bias if using current ticker universe
4. **Transaction Costs**: Backtest engine uses default transaction costs; actual costs may vary by broker
5. **Options Execution**: Model signals options strategies but doesn't simulate actual options trades (yet)

---

## Future Enhancements

- [ ] Walk-forward optimization for out-of-sample validation
- [ ] Machine learning for dynamic factor weighting
- [ ] Macro regime detection (recession, expansion)
- [ ] Earnings surprise integration
- [ ] Short-interest analysis
- [ ] Analyst rating changes
- [ ] Insider trading activity
- [ ] Statistical arbitrage pairs
- [ ] Factor timing based on market conditions

---

## References

- **Value Investing**: Benjamin Graham, "The Intelligent Investor"
- **Momentum**: Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"
- **Quality**: Asness, Frazzini & Pedersen (2019), "Quality Minus Junk"
- **Options Overlay**: Israelov & Nielsen (2015), "Covered Call Strategies"
- **Multi-Factor**: Fama & French (2015), "A Five-Factor Asset Pricing Model"

---

## Support

For questions or issues with this model:
- Review the implementation in `run_model.py`
- Check the MetaQuant documentation at `CLAUDE.md`
- Examine signal metadata for debugging insights

---

**Disclaimer**: This model is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough backtesting and paper trading before deploying any strategy with real capital.
