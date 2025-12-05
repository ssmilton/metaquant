# Example: Complete parameter optimization workflow for sector momentum model
#
# This script demonstrates:
# 1. Grid search to explore parameter space
# 2. Walk-forward optimization for robustness
# 3. Result analysis and comparison
#
# Prerequisites:
# - Database initialized: quantfw init-db
# - Market data ingested for your universe
# - Model exists: models/alice/sector_momentum_v1/

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Parameter Optimization Example" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$MODEL_ID = "sector_momentum_v1"
$TICKERS = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,BAC,WMT"
$TRAIN_START = "2020-01-01"
$TRAIN_END = "2022-12-31"
$TEST_START = "2023-01-01"
$TEST_END = "2023-12-31"

Write-Host "Step 1: Generate grid search experiment" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
python scripts/generate_param_sweep.py `
  --model-id $MODEL_ID `
  --params lookback_days=10,20,30,60 top_n_sectors=2,3,5 `
  --tickers $TICKERS `
  --start-date $TRAIN_START `
  --end-date $TRAIN_END `
  --experiment-id "${MODEL_ID}_grid_train" `
  --output "experiments/${MODEL_ID}_grid_train.yaml"

Write-Host ""
Write-Host "Step 2: Run grid search" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
quantfw run-experiment "experiments/${MODEL_ID}_grid_train.yaml"

Write-Host ""
Write-Host "Step 3: View top performers" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
quantfw show-experiment "experiments/${MODEL_ID}_grid_train" --metric sharpe --top-n 5

Write-Host ""
Write-Host "Step 4: Generate walk-forward experiments" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
python scripts/generate_walk_forward.py `
  --model-id $MODEL_ID `
  --params lookback_days=20,30,60 top_n_sectors=3,5 `
  --tickers $TICKERS `
  --start-date $TRAIN_START `
  --end-date $TRAIN_END `
  --train-months 6 `
  --test-months 3 `
  --experiment-id-prefix "${MODEL_ID}_wf" `
  --output-dir "experiments/${MODEL_ID}_walk_forward"

Write-Host ""
Write-Host "Step 5: Run walk-forward train windows" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
Get-ChildItem "experiments/${MODEL_ID}_walk_forward/train_*.yaml" | ForEach-Object {
    Write-Host "Running $($_.Name)..." -ForegroundColor Gray
    quantfw run-experiment $_.FullName
}

Write-Host ""
Write-Host "Step 6: Analyze parameter stability" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
$trainDirs = Get-ChildItem "experiments/${MODEL_ID}_walk_forward" -Directory |
             Where-Object { $_.Name -match "${MODEL_ID}_wf_train_" }
python scripts/analyze_param_stability.py @($trainDirs.FullName) --metric sharpe --top-n 3

Write-Host ""
Write-Host "Step 7: Generate out-of-sample test" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
# Use the most stable parameters found (example: lookback_days=30, top_n_sectors=3)
python scripts/generate_param_sweep.py `
  --model-id $MODEL_ID `
  --params lookback_days=30 top_n_sectors=3 `
  --tickers $TICKERS `
  --start-date $TEST_START `
  --end-date $TEST_END `
  --experiment-id "${MODEL_ID}_oos_test" `
  --output "experiments/${MODEL_ID}_oos_test.yaml"

Write-Host ""
Write-Host "Step 8: Run out-of-sample test" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
quantfw run-experiment "experiments/${MODEL_ID}_oos_test.yaml"

Write-Host ""
Write-Host "Step 9: Compare train vs test performance" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
Write-Host "Training period (in-sample):" -ForegroundColor White
quantfw show-experiment "experiments/${MODEL_ID}_grid_train" --metric sharpe --top-n 3

Write-Host ""
Write-Host "Test period (out-of-sample):" -ForegroundColor White
quantfw show-experiment "experiments/${MODEL_ID}_oos_test" --metric sharpe

Write-Host ""
Write-Host "Step 10: Generate detailed reports" -ForegroundColor Yellow
Write-Host "-----------------------------------------" -ForegroundColor Yellow
quantfw experiment-report "experiments/${MODEL_ID}_grid_train" `
  --output-file "experiments/${MODEL_ID}_grid_train_report.txt"

quantfw experiment-report "experiments/${MODEL_ID}_oos_test" `
  --output-file "experiments/${MODEL_ID}_oos_test_report.txt"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Parameter Optimization Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor White
Write-Host "  - experiments/${MODEL_ID}_grid_train/" -ForegroundColor Gray
Write-Host "  - experiments/${MODEL_ID}_walk_forward/" -ForegroundColor Gray
Write-Host "  - experiments/${MODEL_ID}_oos_test/" -ForegroundColor Gray
Write-Host ""
Write-Host "Reports saved to:" -ForegroundColor White
Write-Host "  - experiments/${MODEL_ID}_grid_train_report.txt" -ForegroundColor Gray
Write-Host "  - experiments/${MODEL_ID}_oos_test_report.txt" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review parameter stability from walk-forward analysis" -ForegroundColor Gray
Write-Host "  2. Compare in-sample vs out-of-sample performance" -ForegroundColor Gray
Write-Host "  3. Check for overfitting (large performance gap = overfit)" -ForegroundColor Gray
Write-Host "  4. Consider economic rationale for optimal parameters" -ForegroundColor Gray
