# Experiments

This directory contains experiment configurations for batch model evaluation and comparison.

## What are Experiments?

Experiments allow you to:
- **Run multiple models in batch** - Compare different strategies on the same data
- **Parameter sweeps** - Systematically test model variations
- **Benchmark comparison** - Evaluate models against buy-and-hold or equal-weight portfolios
- **Reproducible research** - Document complete evaluation workflows in YAML
- **Comparative analysis** - Automatically rank models by performance metrics

## Quick Start

### 1. Run an Example Experiment

```bash
# Compare momentum models with different lookback periods
quantfw run-experiment experiments/example_parameter_sweep.yaml
```

### 2. View Results

```bash
# Show all results
quantfw show-experiment experiments/momentum_lookback_sweep

# Show top 5 by Sharpe ratio
quantfw show-experiment experiments/momentum_lookback_sweep --metric sharpe --top-n 5
```

### 3. Generate Performance Report

```bash
quantfw experiment-report experiments/momentum_lookback_sweep
```

## Example Configurations

This directory includes several example experiments:

### `example_model_comparison.yaml`
- Compares multiple models on the same universe
- Demonstrates individual model runs with different parameters
- Includes buy-and-hold and equal-weight benchmarks

### `example_parameter_sweep.yaml`
- Systematic parameter optimization
- Tests 7 different lookback periods for a momentum model
- Helps identify optimal parameter values

### `example_comprehensive.yaml`
- Full-featured experiment demonstrating all capabilities
- Combines individual runs and parameter sweeps
- Shows how to use tickers files for large universes
- Multiple benchmarks and custom settings

## Creating Your Own Experiment

### Basic Structure

```yaml
experiment_id: "my_experiment"
description: "Brief description of what you're testing"

universe:
  tickers: [AAPL, MSFT, GOOGL]  # Or use tickers_file or security_ids

start_date: "2020-01-01"
end_date: "2023-12-31"

models:
  - model_id: "sector_momentum_v1"
    name: "Momentum Strategy"
    parameters:
      lookback_days: 30
```

### Adding Parameter Sweeps

```yaml
parameter_sweeps:
  - model_id: "sector_momentum_v1"
    base_parameters:
      threshold: 0.05
    sweep_parameters:
      lookback_days: [10, 20, 30, 60, 90]  # Tests 5 variations
```

This automatically generates 5 model runs with different lookback periods.

### Adding Benchmarks

```yaml
benchmarks:
  # Buy and hold a single security
  - type: "buy_and_hold"
    ticker: "SPY"
    name: "S&P 500"

  # Equal weight portfolio of all securities in universe
  - type: "equal_weight"
    name: "Equal Weight"
```

### Universe Options

You can specify the universe in three ways:

```yaml
# Option 1: Explicit ticker list
universe:
  tickers: [AAPL, MSFT, GOOGL, AMZN]

# Option 2: Tickers from a file
universe:
  tickers_file: "config/sp500_tickers.txt"

# Option 3: Security IDs (if you know them)
universe:
  security_ids: [1, 2, 3, 4]
```

## Experiment Workflow

1. **Create Configuration** - Define experiment in YAML
2. **Run Experiment** - `quantfw run-experiment <config.yaml>`
3. **Review Results** - Results saved to `experiments/<experiment_id>/`
4. **Compare Models** - Use `show-experiment` or `compare-experiments`
5. **Generate Report** - Create detailed performance report

## Output Files

After running an experiment, you'll find:

```
experiments/<experiment_id>/
├── config.yaml              # Experiment configuration
├── comparison.csv           # Performance metrics table
├── results.json            # Detailed results with run IDs
├── equity_curves/          # Equity curves for each model
│   ├── Momentum_30d.csv
│   ├── Momentum_60d.csv
│   └── ...
└── performance_report.txt  # Generated report (if created)
```

## Available CLI Commands

```bash
# Run an experiment
quantfw run-experiment <config.yaml>

# View experiment results
quantfw show-experiment <experiment_dir> [--metric sharpe] [--top-n 5]

# Compare multiple experiments
quantfw compare-experiments <exp1_dir> <exp2_dir> <exp3_dir>

# Generate performance report
quantfw experiment-report <experiment_dir> [--output-file report.txt]
```

## Tips

1. **Start Small** - Test with a small universe and short date range first
2. **Use Descriptive Names** - Give models and benchmarks clear names
3. **Parameter Sweeps** - Use sweeps to systematically explore parameter space
4. **Save Configs** - Version control your experiment configs for reproducibility
5. **Compare Metrics** - Look at multiple metrics, not just returns (Sharpe, max drawdown, etc.)

## Common Use Cases

### Model Selection
Compare different strategy types to find the best approach:
```yaml
models:
  - model_id: "momentum_model"
  - model_id: "mean_reversion_model"
  - model_id: "ml_model"
```

### Parameter Optimization
Find optimal parameters for a single model:
```yaml
parameter_sweeps:
  - model_id: "momentum_model"
    sweep_parameters:
      lookback_days: [10, 20, 30, 60, 90]
      threshold: [0.01, 0.03, 0.05]
```
This creates 5 × 3 = 15 runs testing all combinations.

### Walk-Forward Testing
Create multiple experiments with different time periods:
```bash
quantfw run-experiment experiments/2020_backtest.yaml
quantfw run-experiment experiments/2021_backtest.yaml
quantfw run-experiment experiments/2022_backtest.yaml
quantfw compare-experiments experiments/2020_backtest experiments/2021_backtest experiments/2022_backtest
```

## Next Steps

- Review the example configurations in this directory
- Create your first experiment
- Check the main documentation in `CLAUDE.md` for more details
- Explore the comparison utilities in `core/experiments/comparison.py`
