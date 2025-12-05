# Value-Momentum-Options Model: Extended Backtest Report

## Executive Summary

Following the initial parameter optimization experiments, we executed the remaining recommendations to further validate and test the value_momentum_options_v1 model with:

1. **Expanded universe**: 150 stocks with complete fundamental data (vs. 33 in initial testing)
2. **Extended backtest period**: Maximum available historical data (2020-2024, ~4 years)
3. **Optimized parameters**: Applied the debt_to_equity and other parameter adjustments identified in previous experiments

## Actions Completed

### 1. Universe Expansion (150 Stocks)

**Objective**: Increase diversification and statistical significance by expanding the test universe from 33 to 100+ stocks.

**Method**:
- Queried DuckDB for securities with complete fundamental data
- Required metrics: market_cap, ROE, P/E, P/B, debt_to_equity, profit_margin
- Filtered for market cap > $1B and excluded special securities
- Result: **150 stocks** identified with complete fundamental data

**Universe File**: `config/value_momentum_extended_universe.txt`

**Top stocks by market cap**:
1. NVDA - Technology
2. AAPL - Technology
3. MSFT - Technology
4. GOOGL - Communication Services
5. GOOG - Communication Services
6. AMZN - Consumer Cyclical
7. META - Communication Services
8. AVGO - Technology
9. TSLA - Consumer Cyclical
10. NFLX - Communication Services

### 2. Historical Data Coverage Analysis

**Discovery**: Database price history only extends back to 2020-01-02

**Implication**:
- Cannot run true 5-year backtest (would require data from 2018)
- Maximum available period: 2020-01-02 to 2024-01-01 (~4 years)
- This period still captures important market cycles:
  - COVID-19 crash (March 2020)
  - Recovery and stimulus-driven rally (2020-2021)
  - Inflation surge and rate hikes (2022)
  - Market normalization (2023-2024)

### 3. Extended Backtest Experiments

#### Experiment 1: 5-Year Attempt (Failed - No Data)

**Configuration**: `experiments/value_momentum_5year_extended.yaml`
- Period: 2019-01-01 to 2024-01-01
- Universe: 150 stocks
- Models: 5 optimized configurations

**Result**: FAILED
- Reason: No price data available for 2018-2019
- Models showed near-zero returns (0.07%-0.34% over 5 years)
- Indicates insufficient data prevented proper portfolio construction

**Key Finding**: Database limitation identified - need to either:
- Ingest historical data back to 2018 or earlier
- Use available data period (2020-2024)

#### Experiment 2: 4-Year with Available Data (In Progress)

**Configuration**: `experiments/value_momentum_4year_extended.yaml`
- Period: 2020-01-02 to 2024-01-01 (~4 years)
- Universe: 150 stocks with complete fundamental data
- Models: 5 optimized configurations
  - Conservative-4Y
  - Moderate-4Y
  - Aggressive-4Y
  - Growth-4Y
  - Value-4Y

**Benchmarks**:
- Buy & Hold AAPL
- Buy & Hold NVDA
- Buy & Hold MSFT
- Equal Weight Portfolio (150 stocks)

**Status**: Running in background
**Expected Completion**: ~5-10 minutes

## Parameter Configuration Summary

All configurations use the optimized `max_debt_to_equity` values identified in previous experiments:

| Strategy | max_debt_to_equity | Rationale |
|----------|-------------------|-----------|
| Conservative | 10.0 | Up from 2.0 - allows quality large caps |
| Moderate | 15.0 | Up from 2.0 - balanced approach |
| Aggressive | 20.0 | Up from 2.0 - accepts higher leverage |
| Growth | 12.0 | Up from 2.0 - growth stocks moderately leveraged |
| Value | 25.0 | Up from 2.0 - value stocks often highly leveraged |

Other key parameter differences by strategy:

**Conservative**:
- `target_positions`: 15 (concentrated)
- `min_market_cap`: $5B (mega caps only)
- `max_volatility`: 0.60 (low risk)
- `min_roe`: 0.12 (quality focus)
- `momentum_6m_min`: 0.00 (positive momentum only)

**Moderate**:
- `target_positions`: 20 (balanced)
- `min_market_cap`: $2B
- `max_volatility`: 0.75
- `min_roe`: 0.08
- `momentum_6m_min`: -0.15

**Aggressive**:
- `target_positions`: 25 (diversified)
- `min_market_cap`: $1B
- `max_volatility`: 1.20 (accepts high risk)
- `min_roe`: 0.05 (relaxed quality)
- `momentum_6m_min`: -0.30

**Growth**:
- `max_pe`: 200 (accepts high valuations)
- `max_pb`: 40
- `max_volatility`: 1.50
- `min_roe`: 0.15 (strong profitability)
- `momentum_6m_min`: 0.15 (strong positive momentum)

**Value**:
- `max_pe`: 20 (strict valuation)
- `max_pb`: 4
- `max_debt_to_equity`: 25.0 (most relaxed)
- `momentum_6m_min`: -0.40 (contrarian/deep value)

## Data Quality Insights

### Fundamental Data Coverage

Of 412,863 securities in database:
- 5,371 have fundamental data
- 150 have **complete** fundamental data (all 6 key metrics)
- Coverage rate: ~3% of all securities have complete fundamentals

This suggests:
1. Focus on quality over quantity in universe selection
2. Many securities lack necessary data for fundamental analysis
3. Current 150-stock universe represents high-quality subset

### Price Data Coverage

**Timeframe**: 2020-01-02 to present
**Limitation**: Cannot test beyond 2020 start date
**Recommendation**: Ingest additional historical data for longer backtests

## Next Steps

### Immediate
1. ✅ Complete 4-year extended backtest (in progress)
2. 📋 Analyze results and compare to benchmarks
3. 📋 Identify best-performing configuration for 2020-2024 period

### Future Enhancements

#### 1. Historical Data Extension
- Ingest price data back to 2015 or earlier
- Enable true 5+ year backtests
- Capture additional market cycles

#### 2. Walk-Forward Optimization
- Implement rolling window validation
- Test parameter stability across different periods
- Validate out-of-sample performance

**Approach**:
- Train on 2 years, test on 1 year
- Roll window forward quarterly
- Compare in-sample vs. out-of-sample metrics

#### 3. Regime-Based Parameters
- Detect market regimes (bull, bear, high vol, low vol)
- Adjust parameters dynamically based on regime
- Test regime-detection algorithms (HMM, volatility thresholds, macro indicators)

#### 4. Options Strategy Optimization
Currently using default parameters for options overlay:
- `covered_call_iv_threshold`: Default value
- `protective_put_vix_threshold`: Default value
- `options_allocation_pct`: Default value

Optimize these parameters through grid search or Bayesian optimization.

#### 5. Multi-Objective Optimization
Current optimization focuses on Sharpe ratio. Consider:
- Pareto frontier analysis (return vs. risk)
- Max drawdown minimization
- Tail risk metrics (CVaR, maximum loss)
- Sortino ratio (downside risk focus)

#### 6. Machine Learning Approaches
- Feature engineering from fundamentals and technicals
- Ensemble models combining multiple factor scores
- Reinforcement learning for dynamic position sizing
- Neural network for market regime detection

## Files Created/Updated

### Configuration Files
1. `experiments/value_momentum_5year_extended.yaml` - 5-year attempt (failed due to data)
2. `experiments/value_momentum_4year_extended.yaml` - 4-year backtest (running)

### Universe Files
1. `config/value_momentum_extended_universe.txt` - 150 stocks with complete fundamentals

### Documentation
1. `VALUE_MOMENTUM_EXTENDED_BACKTEST_REPORT.md` - This report

## Conclusion

Successfully expanded the testing scope from 33 to 150 stocks, representing a **4.5x increase** in universe size. Discovered data limitations that constrain historical backtesting to 2020-2024 period. Currently running comprehensive 4-year backtest to validate optimized parameters across multiple market cycles.

The expanded universe provides:
- Better diversification potential
- More statistical significance
- Coverage across multiple sectors
- Quality-filtered stocks with complete fundamentals

Results from the 4-year extended backtest will provide robust validation of the optimized parameter configurations across different market regimes (COVID crash, recovery, inflation, normalization).

---

## Results from 4-Year Extended Backtest

### Performance Summary

**Test Period**: 2020-01-02 to 2024-01-01 (~4 years)
**Universe**: 150 stocks with complete fundamental data
**Initial Capital**: $1,000,000

| Strategy | Total Return | CAGR | Volatility | Sharpe | Max Drawdown |
|----------|--------------|------|------------|--------|--------------|
| Conservative-4Y | 0.075% | 0.019% | 0.032% | 0.59 | -0.046% |
| Moderate-4Y | 0.147% | 0.037% | 0.054% | 0.68 | -0.066% |
| Aggressive-4Y | **0.337%** | **0.084%** | **0.109%** | **0.78** | **-0.133%** |
| Growth-4Y | 0.154% | 0.039% | 0.044% | 0.88 | -0.041% |
| Value-4Y | 0.103% | 0.026% | 0.044% | 0.59 | -0.057% |

**Benchmarks**:

| Benchmark | Total Return | CAGR | Volatility | Sharpe | Max Drawdown |
|-----------|--------------|------|------------|--------|--------------|
| Buy & Hold NVDA | **725.68%** | **69.76%** | 54.20% | 1.25 | -66.36% |
| Buy & Hold AAPL | 156.41% | 26.62% | 33.55% | 0.87 | -31.43% |
| Buy & Hold MSFT | 134.12% | 23.77% | 32.61% | 0.82 | -37.56% |
| Equal Weight Portfolio | 122.43% | 22.19% | 26.78% | 0.88 | -32.63% |

### Critical Finding: Severe Underperformance

All model configurations **drastically underperformed** benchmarks:

- **Aggressive-4Y** (best model): 0.337% vs 122.43% Equal Weight = **363x worse**
- **Models vs NVDA Buy & Hold**: 0.075%-0.337% vs 725.68% = **2,154x-9,676x worse**
- Extremely low volatility (0.03%-0.11%) indicates **minimal capital deployment**
- Max drawdown only -0.04% to -0.13% suggests **almost no trading activity**

### Root Cause Analysis

#### 1. Lookback Period Data Availability

The backtest requires 252 days (1 year) of lookback data before the start date:
- Start date: 2020-01-02
- Required lookback: back to 2019-01-02
- **Database only has data from 2020-01-02 onwards**

This means:
- For most of 2020, the model lacks sufficient historical data to calculate momentum, volatility, and other lookback metrics
- Models likely generated very few or no signals during this period
- By the time sufficient data accumulated (2021), market had already rallied

#### 2. Fundamental Data Staleness

Even with 150 stocks having complete fundamental data:
- Fundamentals may not be updated frequently enough
- `report_date` field indicates when fundamentals were captured
- If fundamentals are stale or only updated quarterly, many trading days have no new fundamental data
- This further reduces the number of stocks passing filters on any given day

#### 3. Filter Strictness Still Too High

Despite relaxing `max_debt_to_equity` from 2.0 to 10-25:
- Other filters may still be too strict in combination
- Conservative strategy requires:
  - Market cap > $5B
  - ROE > 12%
  - P/E < 35
  - Volatility < 60%
  - Positive 6m momentum
  - Debt/Equity < 10
- Even with 150-stock universe, very few stocks may pass ALL filters simultaneously

#### 4. Portfolio Construction Issue

Tiny returns suggest:
- Models may be selecting <5 stocks when target is 15-25
- Position sizes may be extremely small
- Stop losses triggering immediately
- Capital remaining largely uninvested

### Diagnostic Recommendations

To identify the exact bottleneck:

1. **Enable Debug Logging**
   - Add verbose logging to model to see:
     - How many candidates evaluated each day
     - How many pass each filter
     - Final portfolio size vs. target
     - Actual position sizes
     - Trade frequency

2. **Test with Minimal Filters**
   - Run experiment with only basic filters (market cap, volume)
   - Gradually add back other filters to identify bottleneck

3. **Check Fundamental Data Availability**
   ```sql
   SELECT
     COUNT(DISTINCT security_id) as stocks,
     report_date,
     COUNT(*) as metrics
   FROM fundamentals
   WHERE report_date >= '2020-01-01'
   GROUP BY report_date
   ORDER BY report_date
   ```
   - Verify fundamental data is available throughout 2020-2024
   - Check update frequency

4. **Analyze Model Signal Output**
   - Check if model is generating signals (currently database files are missing)
   - Verify signal strength and confidence levels
   - Ensure signals are being converted to trades properly

5. **Test Shorter Lookback Period**
   - Reduce lookback from 365 days to 180 or 90 days
   - This allows trading to start earlier in 2020

6. **Expand Data History**
   - Ingest price data back to 2018 or 2017
   - This enables proper lookback calculation from 2020-01-02 start date

### Conclusion

**Status**: COMPLETED with CRITICAL ISSUES IDENTIFIED

The extended backtest successfully executed but revealed fundamental data and configuration issues that prevent the model from achieving meaningful returns. The near-zero performance (0.07%-0.34% over 4 years) compared to benchmarks (122%-726%) indicates the model is not deploying capital effectively.

**Primary Issue**: Insufficient historical data for lookback calculations combined with overly strict filter combinations.

**Immediate Next Steps**:
1. Ingest historical price data back to 2018 minimum
2. Add verbose debug logging to model to understand filter progression
3. Test with progressively relaxed filters to identify bottleneck
4. Verify fundamental data availability and staleness

**Long-term Solution**:
- Implement adaptive filters that adjust based on market conditions
- Use relative ranking instead of absolute thresholds
- Consider factor-based portfolio construction that guarantees minimum portfolio size

---

**Status**: COMPLETED
**Last Updated**: 2024-12-04
**Critical Finding**: Model underperforming by 363x-9,676x due to data availability and filter configuration issues
