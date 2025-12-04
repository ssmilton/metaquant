# Stock Screener with Options-Aware Scoring (v1.0.0)

A multi-factor stock screener that combines traditional equity metrics with options market data to identify attractive long opportunities. The model uses momentum, volatility, liquidity, and options-derived signals to rank and select top candidates.

## Author
shane

## Features

### Traditional Equity Metrics
- **Momentum**: 6-month and 3-month price returns
- **RSI (Relative Strength Index)**: Mean reversion indicator (14-day period)
- **Realized Volatility**: Annualized price volatility
- **Liquidity**: Average volume and dollar volume

### Options-Aware Metrics (New!)
- **Implied Volatility (IV) Rank**: Percentile ranking of current IV in historical distribution
- **Put/Call Open Interest Ratio**: Sentiment indicator from options market
- **Options Activity**: Total open interest as liquidity/interest measure

## Input Requirements

The model expects the following data in the input payload:

### Required
- `data.prices`: Price data for each security
  - `security_id`: Integer security identifier
  - `dates`: List of date strings (YYYY-MM-DD format)
  - `close`: List of closing prices
  - `volume`: List of volumes (optional but recommended)

### Optional (for Options-Aware Scoring)
- `data.options`: Options quotes organized by security_id
  - Structure: `{ "<security_id>": [ ...option_quotes ] }`
  - Each option quote contains:
    - `option_id`: Integer identifier
    - `date`: Date string (YYYY-MM-DD)
    - `iv`: Implied volatility (decimal, e.g., 0.25 for 25%)
    - `delta`: Delta value (positive for calls, negative for puts)
    - `open_interest`: Number of open contracts
    - Other Greeks (gamma, theta, vega, etc.) - not currently used

**Note**: The model is backward compatible. If no options data is provided, it gracefully falls back to equity-only scoring with neutral scores for options metrics.

## Scoring Methodology

The model computes a composite score for each security using weighted factors:

### Weights (with Options Data)
| Factor | Weight | Description |
|--------|--------|-------------|
| Momentum | 20% | 6-month (60%) + 3-month (40%) returns |
| RSI | 8% | Preference for RSI between 30-70 |
| Volatility | 20% | Inverse relationship (lower vol = higher score) |
| Liquidity | 8% | Average volume (60%) + dollar volume (40%) |
| **IV Rank** | **15%** | Low IV (<30) scored highest, high IV (>70) scored lowest |
| **Put/Call Ratio** | **12%** | Ratio <0.7 = bullish, 0.7-1.3 = neutral, >1.3 = bearish |
| **Options Activity** | **17%** | Higher open interest = higher score |

### Interpretation

**IV Rank Score**:
- **Low (<30%)**: High score → Cheap options, potential breakout setup
- **Moderate (30-70%)**: Neutral score → Normal volatility environment
- **High (>70%)**: Lower score → Expensive options, heightened uncertainty

**Put/Call Ratio Score**:
- **<0.7**: High score → Bullish sentiment (more calls than puts)
- **0.7-1.3**: Neutral score → Balanced sentiment
- **>1.3**: Lower score → Bearish sentiment (more puts than calls)

**Options Activity Score**:
- Relative ranking based on total open interest
- Higher = more liquid options market = higher score

## Parameters

- `max_positions` (integer, default: 8): Maximum number of positions to signal

## Output

The model returns long signals for the top-ranked securities. Each signal includes:

```json
{
  "timestamp": "2023-10-15",
  "security_id": 101,
  "signal_type": "long",
  "strength": 0.95,
  "confidence": 0.82,
  "meta": {
    "composite": 0.756,
    "ret_6m": 0.15,
    "ret_3m": 0.08,
    "vol": 0.22,
    "avg_iv": 0.28,
    "iv_rank": 45.2,
    "put_call_oi_ratio": 0.85,
    "options_activity": 15000.0
  }
}
```

### Signal Fields
- `strength`: Normalized composite score (0-1, relative to best candidate)
- `confidence`: Volatility-based confidence (lower vol = higher confidence)
- `meta`: Detailed breakdown of all scoring components

## Usage Examples

### With Options Data (Full Features)

```bash
# Prepare a universe with options data
quantfw backtest \
  --model-id stock_screener_options_v1 \
  --tickers AAPL,MSFT,GOOGL,NVDA,META \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --params '{"max_positions": 3}'
```

### Without Options Data (Equity-Only Fallback)

```bash
# Works with only price data
quantfw backtest \
  --model-id stock_screener_options_v1 \
  --security-ids 1,2,3,4,5 \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --params '{"max_positions": 3}'
```

## Data Requirements

### Minimum Historical Data
- **130 trading days** (~6 months) of price history required per security
- This allows computation of 6-month returns and proper RSI calculation

### Recommended Setup
1. Populate securities table with tickers
2. Ingest daily price data using `scripts/fetch_yfinance_equities.py`
3. (Optional) Ingest options data using `scripts/fetch_yfinance_options.py`
4. Run backtest with desired universe

## Implementation Details

### Files
- `model.json`: Model manifest and metadata
- `run_model.py`: Main scoring and signal generation logic
- `requirements.txt`: Python dependencies (pandas, numpy)

### Key Functions
- `compute_options_metrics()`: Calculates IV rank, P/C ratio, and activity
- `build_candidate()`: Computes all equity + options metrics for a security
- `score_candidates()`: Applies weighting scheme and ranks all candidates
- `make_signal()`: Formats top candidates as trading signals

### Design Philosophy
- **Graceful degradation**: Works with or without options data
- **Defensive programming**: Handles missing data, NaN values robustly
- **Transparent scoring**: All components included in signal metadata
- **Configurable**: Max positions easily adjusted via parameters

## Interpretation Guide

### High Composite Score Candidates Typically Have:
- ✓ Strong recent momentum (3-6 month returns)
- ✓ Moderate RSI (not oversold or overbought)
- ✓ Lower realized volatility
- ✓ Decent liquidity (volume/dollar volume)
- ✓ Low IV rank (cheap options)
- ✓ Bullish put/call ratio (<0.7)
- ✓ High options activity (liquid options market)

### Use Cases
1. **Long equity screening**: Find stocks with positive momentum and favorable options sentiment
2. **Options strategy setup**: Identify candidates with cheap options (low IV rank) for long volatility plays
3. **Sentiment analysis**: Use put/call ratios as contrarian or confirmation signals
4. **Multi-factor research**: Test combinations of equity and options factors

## Limitations

- **Long-only**: Model only generates long signals, no short signals
- **Naive execution**: Backtest engine uses simple 1-unit per signal execution
- **No position sizing**: Equal weighting across all signals
- **Historical IV**: Requires historical options data for accurate IV rank calculation
- **Delta proxy**: Uses delta sign to distinguish calls/puts (assumes standard conventions)

## Future Enhancements

Potential improvements for v2:
- [ ] Short signal generation based on bearish setups
- [ ] IV percentile bands for better regime detection
- [ ] Skew metrics (put vs call IV spreads)
- [ ] Term structure analysis (near-term vs far-term IV)
- [ ] Position sizing based on confidence scores
- [ ] Mean reversion signals (high IV rank + oversold RSI)
- [ ] Integration with fundamental metrics

## Version History

### v1.0.0 (Current)
- Initial release with multi-factor equity scoring
- Options-aware enhancements (IV rank, P/C ratio, activity)
- Graceful fallback for missing options data
- Configurable max positions parameter
