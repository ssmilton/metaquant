# Parameter Optimization Cheatsheet

Quick reference for common parameter optimization tasks.

## Grid Search

### Basic Grid Search
```bash
python scripts/generate_param_sweep.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=10,20,30 top_n_sectors=2,3,5 \
  --tickers AAPL,MSFT,GOOGL \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output experiments/my_sweep.yaml

quantfw run-experiment experiments/my_sweep.yaml
quantfw show-experiment experiments/my_sweep --metric sharpe
```

### With Ticker File
```bash
python scripts/generate_param_sweep.py \
  --model-id my_model \
  --params lookback=20,30,60 \
  --tickers-file sp500.txt \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output experiments/sweep.yaml
```

### Multiple Parameter Types
```bash
# Integers, floats, booleans, strings
--params lookback=10,20 threshold=0.01,0.05 use_filter=true,false strategy=momentum,mean_reversion
```

## Walk-Forward Optimization

### Generate Walk-Forward Experiments
```bash
python scripts/generate_walk_forward.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=10,20,30 top_n_sectors=2,3,5 \
  --tickers AAPL,MSFT,GOOGL \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --train-months 12 \
  --test-months 3 \
  --output-dir experiments/walk_forward
```

### Run All Train Windows
```bash
# Bash
for f in experiments/walk_forward/train_*.yaml; do
  quantfw run-experiment "$f"
done

# PowerShell
Get-ChildItem experiments/walk_forward/train_*.yaml | ForEach-Object {
  quantfw run-experiment $_.FullName
}
```

### Analyze Parameter Stability
```bash
python scripts/analyze_param_stability.py \
  experiments/walk_forward/train_* \
  --metric sharpe \
  --top-n 3
```

## Viewing Results

### Show All Results
```bash
quantfw show-experiment experiments/my_sweep
```

### Sort by Different Metrics
```bash
quantfw show-experiment experiments/my_sweep --metric sharpe
quantfw show-experiment experiments/my_sweep --metric cagr
quantfw show-experiment experiments/my_sweep --metric max_drawdown
```

### Show Top N
```bash
quantfw show-experiment experiments/my_sweep --metric sharpe --top-n 5
```

### Generate Report
```bash
quantfw experiment-report experiments/my_sweep
quantfw experiment-report experiments/my_sweep --output-file report.txt
```

### Compare Multiple Experiments
```bash
quantfw compare-experiments \
  experiments/sweep_2020 \
  experiments/sweep_2021 \
  experiments/sweep_2022 \
  --metric sharpe
```

## Common Workflows

### Simple Parameter Sweep
```bash
# 1. Generate and run
python scripts/generate_param_sweep.py \
  --model-id my_model --params param1=1,2,3 \
  --tickers AAPL,MSFT --start-date 2020-01-01 --end-date 2023-12-31 \
  --output experiments/sweep.yaml

quantfw run-experiment experiments/sweep.yaml

# 2. View top 5
quantfw show-experiment experiments/sweep --metric sharpe --top-n 5
```

### Train/Test Split
```bash
# 1. Train period
python scripts/generate_param_sweep.py \
  --model-id my_model --params lookback=10,20,30 \
  --tickers AAPL,MSFT --start-date 2020-01-01 --end-date 2022-12-31 \
  --output experiments/train.yaml

# 2. Test period (same params)
python scripts/generate_param_sweep.py \
  --model-id my_model --params lookback=10,20,30 \
  --tickers AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31 \
  --output experiments/test.yaml

# 3. Run both
quantfw run-experiment experiments/train.yaml
quantfw run-experiment experiments/test.yaml

# 4. Compare
quantfw compare-experiments experiments/train experiments/test --metric sharpe
```

### Full Walk-Forward Workflow
```bash
# 1. Generate walk-forward experiments
python scripts/generate_walk_forward.py \
  --model-id my_model --params lookback=10,20,30 \
  --tickers AAPL,MSFT --start-date 2020-01-01 --end-date 2023-12-31 \
  --train-months 12 --test-months 3 \
  --output-dir experiments/wf

# 2. Run all train windows
for f in experiments/wf/train_*.yaml; do quantfw run-experiment "$f"; done

# 3. Analyze stability
python scripts/analyze_param_stability.py experiments/wf/train_* --metric sharpe --top-n 3

# 4. Run test windows
for f in experiments/wf/test_*.yaml; do quantfw run-experiment "$f"; done

# 5. Compare train vs test
quantfw compare-experiments experiments/wf/train_01 experiments/wf/test_01 --metric sharpe
```

## Advanced Options

### Custom Experiment Settings
```bash
python scripts/generate_param_sweep.py \
  --model-id my_model \
  --params lookback=10,20,30 \
  --tickers AAPL,MSFT \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --experiment-id my_custom_id \
  --description "Custom experiment description" \
  --initial-capital 500000 \
  --transaction-cost-bps 10 \
  --slippage-bps 2 \
  --benchmark QQQ \
  --metrics sharpe sortino max_drawdown cagr calmar win_rate \
  --output experiments/custom.yaml
```

### Overlapping Walk-Forward Windows
```bash
python scripts/generate_walk_forward.py \
  --model-id my_model --params lookback=10,20,30 \
  --tickers AAPL,MSFT --start-date 2020-01-01 --end-date 2023-12-31 \
  --train-months 12 --test-months 6 --step-months 3 \
  --output-dir experiments/overlapping_wf
```

## Quick Tips

### Parameter Ranges
```bash
# Good ranges (economically meaningful)
--params lookback_days=5,10,20,30,60,90
--params threshold=0.01,0.02,0.05,0.1
--params top_n=2,3,5,10

# Avoid arbitrary values
--params lookback_days=7,13,27,59  # Why these specific values?
```

### Metrics to Monitor
```bash
# Don't optimize on returns alone
--metrics sharpe sortino max_drawdown cagr calmar win_rate

# Look for consistent performance across metrics
```

### Checking for Overfitting
```bash
# Large gap = overfitting
# Train Sharpe: 2.5, Test Sharpe: 0.5 → Overfit!
# Train Sharpe: 1.5, Test Sharpe: 1.3 → Reasonable

quantfw show-experiment experiments/train --metric sharpe
quantfw show-experiment experiments/test --metric sharpe
```

### Parameter Sensitivity
```bash
# After running experiment, check if optimal params are stable
quantfw show-experiment experiments/sweep --metric sharpe

# If sharpe changes drastically with small param changes → unstable/overfit
# If sharpe stays similar across nearby param values → more robust
```

## Troubleshooting

### No Results Generated
```bash
# Check experiment ran successfully
ls -la experiments/my_sweep/

# Should contain:
# - comparison.csv
# - experiment_config.yaml
# - Individual run directories
```

### Parameter Not Taking Effect
```bash
# Verify parameter name matches model's expected params
# Check model.json or model documentation
```

### Out of Memory
```bash
# Reduce parameter combinations
--params lookback=20,30  # Instead of 10,15,20,25,30,35,40,...

# Or run experiments sequentially instead of all at once
```

### PowerShell JSON Escaping
```bash
# Don't use --params with complex JSON in PowerShell
# Use the generator scripts instead
python scripts/generate_param_sweep.py ...
```

## File Locations

```
experiments/
├── my_sweep.yaml              # Experiment config
├── my_sweep/                  # Experiment output
│   ├── comparison.csv         # Results comparison
│   ├── experiment_config.yaml # Config copy
│   └── run_*/                 # Individual run data
└── walk_forward/
    ├── manifest.yaml          # WF overview
    ├── train_01.yaml          # Train configs
    ├── test_01.yaml           # Test configs
    ├── train_01/              # Train results
    └── test_01/               # Test results
```

## See Also

- Full guide: `docs/PARAMETER_OPTIMIZATION.md`
- Example scripts: `examples/parameter_optimization_example.{sh,ps1}`
- CLI help: `quantfw --help`
- Experiment config format: `experiments/daily_equities_example.yaml`
