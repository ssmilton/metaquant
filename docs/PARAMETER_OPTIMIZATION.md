# Parameter Optimization Guide

This guide explains how to use MetaQuant's experiment framework to optimize model parameters systematically.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Grid Search](#grid-search)
- [Walk-Forward Optimization](#walk-forward-optimization)
- [Analyzing Results](#analyzing-results)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)

## Overview

MetaQuant provides tools for systematic parameter optimization:

1. **Grid Search**: Test all combinations of specified parameters
2. **Walk-Forward Optimization**: Robust method using rolling train/test windows
3. **Result Analysis**: Compare performance across parameter sets
4. **Stability Analysis**: Identify consistently optimal parameters

## Quick Start

### 1. Simple Grid Search

```bash
# Generate experiment config
python scripts/generate_param_sweep.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=10,20,30 top_n_sectors=2,3,5 \
  --tickers AAPL,MSFT,GOOGL,AMZN,META \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output experiments/my_sweep.yaml

# Run the experiment
quantfw run-experiment experiments/my_sweep.yaml

# View results sorted by Sharpe ratio
quantfw show-experiment experiments/my_sweep --metric sharpe

# See top 5 by CAGR
quantfw show-experiment experiments/my_sweep --metric cagr --top-n 5
```

### 2. Using a Ticker File

```bash
# Create tickers file
cat > sp500.txt << EOF
# S&P 500 Large Caps
AAPL
MSFT
GOOGL
AMZN
META
NVDA
TSLA
EOF

# Generate sweep with ticker file
python scripts/generate_param_sweep.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=20,30,60 \
  --tickers-file sp500.txt \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output experiments/sp500_sweep.yaml

quantfw run-experiment experiments/sp500_sweep.yaml
```

## Grid Search

### Parameter Types

The sweep generator automatically handles different parameter types:

```bash
# Integer parameters
--params lookback_days=10,20,30

# Float parameters
--params threshold=0.01,0.05,0.1

# Boolean parameters
--params use_filter=true,false

# String parameters
--params strategy=momentum,mean_reversion

# Multiple parameters (creates all combinations)
--params lookback=10,20 threshold=0.01,0.05 use_filter=true,false
# This creates 2 × 2 × 2 = 8 combinations
```

### Advanced Grid Search Options

```bash
python scripts/generate_param_sweep.py \
  --model-id my_model \
  --params lookback=10,20,30 threshold=0.01,0.05,0.1 \
  --tickers AAPL,MSFT \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --experiment-id my_custom_id \
  --initial-capital 500000 \
  --transaction-cost-bps 10 \
  --slippage-bps 2 \
  --benchmark QQQ \
  --metrics sharpe sortino max_drawdown cagr calmar \
  --output experiments/advanced_sweep.yaml
```

### Comparing Multiple Experiments

```bash
# Run different experiments
quantfw run-experiment experiments/sweep_2020.yaml
quantfw run-experiment experiments/sweep_2021.yaml
quantfw run-experiment experiments/sweep_2022.yaml

# Compare across experiments
quantfw compare-experiments \
  experiments/sweep_2020 \
  experiments/sweep_2021 \
  experiments/sweep_2022 \
  --metric sharpe
```

## Walk-Forward Optimization

Walk-forward optimization is more robust than simple grid search because it tests parameter stability across different time periods.

### Concept

```
Timeline: |-------- Train --------|-Test-|-------- Train --------|-Test-|
          2020-01-01      2020-12-31  →   2021-01-01      2021-12-31  →

Window 1: Train on 2020, test on Q1 2021
Window 2: Train on 2021, test on Q1 2022
Window 3: Train on 2022, test on Q1 2023
```

### Generate Walk-Forward Experiments

```bash
python scripts/generate_walk_forward.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=10,20,30,60 top_n_sectors=2,3,5 \
  --tickers AAPL,MSFT,GOOGL,AMZN \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --train-months 12 \
  --test-months 3 \
  --output-dir experiments/walk_forward
```

This creates:
- `experiments/walk_forward/train_01.yaml` (2020 data)
- `experiments/walk_forward/test_01.yaml` (Q1 2021 data)
- `experiments/walk_forward/train_02.yaml` (2021 data)
- `experiments/walk_forward/test_02.yaml` (Q1 2022 data)
- `experiments/walk_forward/manifest.yaml` (overview)

### Run Walk-Forward Analysis

```bash
# Run all training windows
for f in experiments/walk_forward/train_*.yaml; do
  quantfw run-experiment "$f"
done

# Analyze which parameters are consistently best
python scripts/analyze_param_stability.py \
  experiments/walk_forward/train_* \
  --metric sharpe \
  --top-n 3

# Based on analysis, run test windows with optimal parameters
# (or run all test windows to validate)
for f in experiments/walk_forward/test_*.yaml; do
  quantfw run-experiment "$f"
done
```

### Customizing Walk-Forward Windows

```bash
# Overlapping windows (step < test duration)
python scripts/generate_walk_forward.py \
  --model-id my_model \
  --params lookback=10,20,30 \
  --tickers AAPL,MSFT \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --train-months 12 \
  --test-months 6 \
  --step-months 3 \
  --output-dir experiments/overlapping_wf

# Larger windows for more data
python scripts/generate_walk_forward.py \
  --model-id my_model \
  --params lookback=10,20,30 \
  --tickers AAPL,MSFT \
  --start-date 2015-01-01 \
  --end-date 2023-12-31 \
  --train-months 24 \
  --test-months 6 \
  --output-dir experiments/large_wf
```

## Analyzing Results

### View Experiment Results

```bash
# Show all results
quantfw show-experiment experiments/my_sweep

# Sort by different metrics
quantfw show-experiment experiments/my_sweep --metric sharpe
quantfw show-experiment experiments/my_sweep --metric cagr
quantfw show-experiment experiments/my_sweep --metric max_drawdown

# Show only top performers
quantfw show-experiment experiments/my_sweep --metric sharpe --top-n 5
```

### Generate Performance Reports

```bash
# Generate detailed report
quantfw experiment-report experiments/my_sweep

# Save to file
quantfw experiment-report experiments/my_sweep --output-file report.txt
```

### Analyze Parameter Stability

After running walk-forward experiments:

```bash
python scripts/analyze_param_stability.py \
  experiments/walk_forward/train_* \
  --metric sharpe \
  --top-n 3
```

This shows:
- Which parameter combinations appear most frequently in top performers
- Which individual parameter values are most common in top performers
- Consistency percentages across windows

## Best Practices

### 1. Start Small

```bash
# Test with 2-3 parameter values first
--params lookback_days=20,30,60

# Then expand if needed
--params lookback_days=10,20,30,60,90
```

### 2. Use Out-of-Sample Testing

```bash
# Train on 2020-2022
python scripts/generate_param_sweep.py \
  --start-date 2020-01-01 \
  --end-date 2022-12-31 \
  --output experiments/train.yaml

# Test on 2023 (never seen during optimization)
python scripts/generate_param_sweep.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output experiments/test.yaml
```

### 3. Include Transaction Costs

Always optimize with realistic costs:

```bash
--transaction-cost-bps 5 \
--slippage-bps 1
```

### 4. Multiple Metrics

Don't optimize on returns alone:

```bash
--metrics sharpe sortino max_drawdown cagr calmar win_rate
```

Look for parameters that perform well across multiple metrics.

### 5. Parameter Ranges

Choose ranges that make economic sense:

```bash
# Good: reasonable momentum lookback periods
--params lookback_days=5,10,20,30,60,90

# Bad: arbitrary values
--params lookback_days=7,13,27,59,91
```

### 6. Use Walk-Forward for Final Validation

Grid search finds optimal parameters, walk-forward validates they're stable:

```bash
# 1. Grid search on full period
python scripts/generate_param_sweep.py ... --output experiments/grid.yaml

# 2. Identify best params
quantfw run-experiment experiments/grid.yaml
quantfw show-experiment experiments/grid --metric sharpe --top-n 5

# 3. Validate with walk-forward
python scripts/generate_walk_forward.py ... --output-dir experiments/wf
```

## Common Pitfalls

### 1. Overfitting

**Problem**: Parameters that maximize past returns rarely maximize future returns.

**Solution**:
- Use walk-forward optimization
- Reserve 20-30% of data for out-of-sample testing
- Prefer parameter regions over single points
- Simpler models with fewer parameters

### 2. Data Snooping

**Problem**: Testing multiple strategies on the same data contaminates results.

**Solution**:
- Define strategy hypothesis before testing
- Use separate data for strategy selection vs parameter optimization
- Document all strategies tested (including failures)

### 3. Survivorship Bias

**Problem**: Using current index constituents creates unrealistic backtests.

**Solution**:
- Use point-in-time universes
- Include delisted/bankrupt companies
- Be aware of index rebalancing

### 4. Ignoring Transaction Costs

**Problem**: Strategies look profitable but fail in live trading.

**Solution**:
- Always include realistic transaction costs
- Account for bid-ask spread and slippage
- Consider market impact for larger orders

### 5. Too Many Parameters

**Problem**: More parameters = more overfitting risk.

**Solution**:
- Start with 1-2 key parameters
- Only add parameters if they improve out-of-sample performance
- Use regularization or parameter constraints

### 6. Optimizing on Too Little Data

**Problem**: Not enough samples to find robust parameters.

**Solution**:
- Minimum 100-200 trades for statistical significance
- Multiple market regimes (bull, bear, sideways)
- At least 3-5 years of data

### 7. Not Checking Parameter Sensitivity

**Problem**: Optimal parameters are a narrow peak (unstable).

**Solution**:
```bash
# Look for plateaus, not peaks
quantfw show-experiment experiments/my_sweep --metric sharpe

# If sharpe=1.5 at lookback=30 but sharpe=0.8 at lookback=29,31
# → Too sensitive, likely overfit
# If sharpe=1.4-1.5 for lookback=25-35
# → More robust
```

## Example Workflow

Complete parameter optimization workflow:

```bash
# 1. Generate grid search
python scripts/generate_param_sweep.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=10,20,30,60 top_n_sectors=2,3,5 \
  --tickers-file sp500.txt \
  --start-date 2018-01-01 \
  --end-date 2022-12-31 \
  --output experiments/grid_2018_2022.yaml

# 2. Run grid search
quantfw run-experiment experiments/grid_2018_2022.yaml

# 3. Check results
quantfw show-experiment experiments/grid_2018_2022 --metric sharpe --top-n 10

# 4. Generate walk-forward with narrowed parameters
python scripts/generate_walk_forward.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=20,30 top_n_sectors=3,5 \
  --tickers-file sp500.txt \
  --start-date 2018-01-01 \
  --end-date 2022-12-31 \
  --train-months 12 \
  --test-months 3 \
  --output-dir experiments/wf_validation

# 5. Run walk-forward train windows
for f in experiments/wf_validation/train_*.yaml; do
  quantfw run-experiment "$f"
done

# 6. Analyze stability
python scripts/analyze_param_stability.py \
  experiments/wf_validation/train_* \
  --metric sharpe --top-n 3

# 7. Test on hold-out period (2023)
python scripts/generate_param_sweep.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=30 top_n_sectors=3 \
  --tickers-file sp500.txt \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output experiments/holdout_2023.yaml

quantfw run-experiment experiments/holdout_2023.yaml
quantfw show-experiment experiments/holdout_2023
```

## Additional Resources

- **Experiment Configuration**: See `experiments/daily_equities_example.yaml`
- **Model Manifests**: See `models/*/model.json` for parameter definitions
- **CLI Reference**: Run `quantfw --help` for all commands
- **Architecture**: See `CLAUDE.md` for system design

## Warning

> **Parameter optimization is inherently dangerous.** You can always find parameters that maximize past returns—that's curve-fitting, not prediction. The parameters that worked best historically rarely work best going forward.
>
> Always:
> - Use walk-forward or cross-validation
> - Test on truly out-of-sample data
> - Prefer simpler models
> - Focus on economic rationale, not statistical fit
> - Be skeptical of "too good to be true" results
