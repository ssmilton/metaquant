# TODO

This file tracks pending items, bugs, and feature requests for the MetaQuant project.

## Items to Resolve

### High Priority
- [ ] Review and finalize `.gitignore` changes
- [ ] Verify FRED data fetch works correctly after recent macro model fix
- [ ] Test macro models without security-ids to ensure stability

### Medium Priority
- [ ] Add error handling for missing data in backtest engine
- [ ] Improve concurrency handling when multiple models write to different DuckDB files
- [ ] Add validation for ticker symbol resolution (handle invalid/missing tickers gracefully)
- [ ] Document the read/write separation pattern more clearly in code comments

### Low Priority
- [ ] Optimize virtual environment creation time on Windows with OneDrive
- [ ] Add cleanup script for old model run databases
- [ ] Improve logging consistency across modules

## Features to Add

### Data & Ingestion
- [ ] Add support for additional data sources (Alpha Vantage, Quandl, etc.)
- [ ] Implement data quality checks and validation pipeline
- [ ] Add support for intraday price data
- [ ] Create bulk ticker import utility
- [ ] Add fundamental data ingestion from external sources
- [ ] Support for corporate actions (splits, dividends) normalization

### Models & Strategy
- [ ] Add more example models (mean reversion, pairs trading, ML-based)
- [ ] Implement walk-forward optimization framework
- [ ] Add support for multi-asset models (stocks + options)
- [ ] Create model versioning and A/B testing framework
- [ ] Add model performance comparison dashboard

### Backtesting & Execution
- [ ] Implement realistic slippage models
- [ ] Add support for custom position sizing algorithms
- [ ] Implement stop-loss and take-profit orders
- [ ] Add support for short selling constraints
- [ ] Create risk management framework (VaR, portfolio limits)
- [ ] Add benchmark comparison (S&P 500, sector indices)

### Infrastructure
- [ ] Add Kubernetes support for distributed model execution
- [ ] Implement result caching for expensive model runs
- [ ] Add support for GPU acceleration in Docker runtime
- [ ] Create web UI for backtest visualization
- [ ] Add API server for programmatic access
- [x] Implement Experiments functionality to evaluate models in batches, evaluate parameter sweeps, and compare models against benchmarks

### Reporting & Analytics
- [ ] Generate HTML/PDF backtest reports
- [ ] Add attribution analysis (factor exposure, sector breakdown)
- [ ] Implement rolling performance metrics
- [ ] Add trade-level analytics (win rate, profit factor, etc.)
- [ ] Create portfolio analytics dashboard

### Documentation
- [ ] Add architecture diagrams
- [ ] Create tutorial for building first model
- [ ] Document best practices for model development
- [ ] Add API reference documentation
- [ ] Create troubleshooting guide

### Testing
- [ ] Increase test coverage to >80%
- [ ] Add integration tests for Docker runtime
- [ ] Add performance benchmarks
- [ ] Create test fixtures for common scenarios
- [ ] Add CI/CD pipeline configuration

## Ideas / Future Considerations
- [ ] Support for live trading integration
- [ ] Add paper trading mode
- [ ] Implement alert system for signal generation
- [ ] Add support for options pricing models
- [ ] Create model marketplace/sharing platform
- [ ] Add support for alternative data sources (sentiment, weather, etc.)

## Notes
- Keep this file updated as issues are resolved and new features are identified
- Use `[x]` to mark completed items
- Add dates when starting/completing major features
