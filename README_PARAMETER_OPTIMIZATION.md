# Parameter Optimization Tools

This directory contains tools for systematic parameter optimization in MetaQuant.

## 🚀 Quick Start

```bash
# Generate a parameter sweep
python scripts/generate_param_sweep.py \
  --model-id sector_momentum_v1 \
  --params lookback_days=10,20,30 top_n_sectors=2,3,5 \
  --tickers AAPL,MSFT,GOOGL \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output experiments/my_sweep.yaml

# Run it
quantfw run-experiment experiments/my_sweep.yaml

# View results
quantfw show-experiment experiments/my_sweep --metric sharpe --top-n 5
```

## 📁 Available Tools

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_param_sweep.py` | Generate grid search experiment configs |
| `scripts/generate_walk_forward.py` | Generate walk-forward optimization configs |
| `scripts/analyze_param_stability.py` | Analyze parameter stability across windows |

### Documentation

| Document | Purpose |
|----------|---------|
| `docs/PARAMETER_OPTIMIZATION.md` | Complete guide with examples and best practices |
| `docs/PARAMETER_OPTIMIZATION_CHEATSHEET.md` | Quick reference for common commands |
| `examples/parameter_optimization_example.{sh,ps1}` | Complete workflow examples |

### CLI Commands

| Command | Purpose |
|---------|---------|
| `quantfw run-experiment` | Execute an experiment config |
| `quantfw show-experiment` | Display experiment results |
| `quantfw compare-experiments` | Compare multiple experiments |
| `quantfw experiment-report` | Generate detailed performance report |

## 🎯 Common Use Cases

### 1. Find Optimal Parameters (Grid Search)

```bash
python scripts/generate_param_sweep.py \
  --model-id my_model \
  --params param1=1,2,3 param2=0.1,0.2 \
  --tickers AAPL,MSFT \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output experiments/grid.yaml

quantfw run-experiment experiments/grid.yaml
```

### 2. Validate Parameter Robustness (Walk-Forward)

```bash
python scripts/generate_walk_forward.py \
  --model-id my_model \
  --params param1=1,2,3 \
  --tickers AAPL,MSFT \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --train-months 12 \
  --test-months 3 \
  --output-dir experiments/wf

# Run all train windows
for f in experiments/wf/train_*.yaml; do
  quantfw run-experiment "$f"
done

# Analyze stability
python scripts/analyze_param_stability.py \
  experiments/wf/train_* \
  --metric sharpe --top-n 3
```

### 3. Out-of-Sample Testing

```bash
# Train on 2020-2022
python scripts/generate_param_sweep.py \
  --model-id my_model --params param1=1,2,3 \
  --tickers AAPL,MSFT \
  --start-date 2020-01-01 --end-date 2022-12-31 \
  --output experiments/train.yaml

# Test on 2023 (never seen during optimization)
python scripts/generate_param_sweep.py \
  --model-id my_model --params param1=2 \
  --tickers AAPL,MSFT \
  --start-date 2023-01-01 --end-date 2023-12-31 \
  --output experiments/test.yaml

quantfw run-experiment experiments/train.yaml
quantfw run-experiment experiments/test.yaml

# Compare performance
quantfw compare-experiments experiments/train experiments/test
```

## 📊 Understanding Results

After running an experiment, view results with different metrics:

```bash
# Overall comparison
quantfw show-experiment experiments/my_sweep

# Best by Sharpe ratio
quantfw show-experiment experiments/my_sweep --metric sharpe --top-n 5

# Best by CAGR
quantfw show-experiment experiments/my_sweep --metric cagr --top-n 5

# Worst by max drawdown (lower is better)
quantfw show-experiment experiments/my_sweep --metric max_drawdown
```

Available metrics:
- `sharpe`: Sharpe ratio (risk-adjusted returns)
- `sortino`: Sortino ratio (downside risk-adjusted returns)
- `cagr`: Compound annual growth rate
- `max_drawdown`: Maximum peak-to-trough decline
- `calmar`: Calmar ratio (CAGR / max drawdown)
- `win_rate`: Percentage of winning trades

## ⚠️ Critical Warning

**Parameter optimization is inherently dangerous.**

The parameters that maximize past returns rarely maximize future returns. You can always find parameters that fit historical data perfectly—that's curve-fitting, not prediction.

**Always:**
- ✅ Use walk-forward or cross-validation
- ✅ Test on truly out-of-sample data (never used during optimization)
- ✅ Prefer simpler models with fewer parameters
- ✅ Focus on economic rationale, not just statistical fit
- ✅ Include realistic transaction costs
- ✅ Look for parameter stability (regions, not peaks)

**Never:**
- ❌ Optimize on your entire dataset
- ❌ Ignore transaction costs and slippage
- ❌ Trust parameters that are too sensitive
- ❌ Add parameters just to improve backtest results
- ❌ Forget to check out-of-sample performance

## 🔄 Recommended Workflow

```
1. Define hypothesis
   ↓
2. Grid search (coarse) → Identify promising parameter ranges
   ↓
3. Grid search (fine) → Narrow down optimal region
   ↓
4. Walk-forward validation → Test parameter stability
   ↓
5. Out-of-sample test → Validate on held-out data
   ↓
6. Live paper trading → Final reality check
```

## 📚 Learn More

- **Complete Guide**: `docs/PARAMETER_OPTIMIZATION.md`
  - Detailed explanations
  - Best practices
  - Common pitfalls
  - Advanced techniques

- **Quick Reference**: `docs/PARAMETER_OPTIMIZATION_CHEATSHEET.md`
  - Common commands
  - Copy-paste examples
  - Troubleshooting

- **Example Workflow**: `examples/parameter_optimization_example.{sh,ps1}`
  - End-to-end example
  - Runnable script
  - Both Bash and PowerShell versions

## 🛠️ Script Usage

### generate_param_sweep.py

```bash
python scripts/generate_param_sweep.py --help

# Required:
--model-id MODEL_ID          Model to optimize
--params PARAM=val1,val2     Parameters to sweep
--tickers TICKER,TICKER      Or --tickers-file or --security-ids
--start-date YYYY-MM-DD      Start date
--end-date YYYY-MM-DD        End date
--output PATH.yaml           Output file

# Optional:
--experiment-id ID           Custom experiment name
--initial-capital N          Starting capital (default: 100000)
--transaction-cost-bps N     Transaction costs (default: 5)
--slippage-bps N             Slippage (default: 1)
--benchmark TICKER           Benchmark (default: SPY)
--metrics M1 M2 M3           Evaluation metrics
```

### generate_walk_forward.py

```bash
python scripts/generate_walk_forward.py --help

# Required:
--model-id MODEL_ID          Model to optimize
--params PARAM=val1,val2     Parameters to sweep
--tickers TICKER,TICKER      Or --tickers-file or --security-ids
--start-date YYYY-MM-DD      Overall start date
--end-date YYYY-MM-DD        Overall end date
--train-months N             Training window size
--test-months N              Test window size
--output-dir DIR             Output directory

# Optional:
--step-months N              Step size (default: test-months)
--experiment-id-prefix PFX   Prefix for experiment IDs
```

### analyze_param_stability.py

```bash
python scripts/analyze_param_stability.py --help

# Required:
experiment_dirs...           Paths to experiment directories

# Optional:
--metric METRIC              Metric to analyze (default: sharpe)
--top-n N                    Top N per window (default: 3)
--output FILE                Save report to file
```

## 💡 Tips

1. **Start small**: Test with 2-3 parameter values before expanding
2. **Use ticker files**: Easier than typing long ticker lists
3. **Check parameter sensitivity**: Robust parameters work in a range, not just one value
4. **Multiple metrics**: Don't optimize on returns alone
5. **Economic sense**: Parameters should have economic rationale
6. **Transaction costs**: Always include realistic costs
7. **Save everything**: Keep all experiment configs for reproducibility

## 🐛 Troubleshooting

**"No files found"**
```bash
# Experiment hasn't run yet
quantfw run-experiment experiments/my_sweep.yaml
```

**"Comparison file not found"**
```bash
# Experiment failed or incomplete
# Check experiment output directory for errors
ls -la experiments/my_sweep/
```

**"Parameter not taking effect"**
```bash
# Verify parameter name matches model's expectations
# Check model.json or model documentation
```

**PowerShell JSON issues**
```bash
# Don't use --params directly in PowerShell
# Use the generator scripts instead
python scripts/generate_param_sweep.py ...
```

## 📞 Getting Help

- Read the full guide: `docs/PARAMETER_OPTIMIZATION.md`
- Check examples: `examples/parameter_optimization_example.{sh,ps1}`
- Quick reference: `docs/PARAMETER_OPTIMIZATION_CHEATSHEET.md`
- CLI help: `quantfw --help`, `quantfw run-experiment --help`
- Script help: `python scripts/generate_param_sweep.py --help`

## 🎓 Further Reading

- **Main documentation**: `CLAUDE.md`
- **Experiment format**: `experiments/daily_equities_example.yaml`
- **Model manifests**: `models/*/model.json`
- **CLI commands**: Run `quantfw --help`
