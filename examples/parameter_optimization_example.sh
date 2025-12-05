#!/bin/bash
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

set -e  # Exit on error

echo "========================================="
echo "Parameter Optimization Example"
echo "========================================="
echo ""

# Configuration
MODEL_ID="sector_momentum_v1"
TICKERS="AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,BAC,WMT"
TRAIN_START="2020-01-01"
TRAIN_END="2022-12-31"
TEST_START="2023-01-01"
TEST_END="2023-12-31"

echo "Step 1: Generate grid search experiment"
echo "-----------------------------------------"
python scripts/generate_param_sweep.py \
  --model-id $MODEL_ID \
  --params lookback_days=10,20,30,60 top_n_sectors=2,3,5 \
  --tickers $TICKERS \
  --start-date $TRAIN_START \
  --end-date $TRAIN_END \
  --experiment-id "${MODEL_ID}_grid_train" \
  --output experiments/${MODEL_ID}_grid_train.yaml

echo ""
echo "Step 2: Run grid search"
echo "-----------------------------------------"
quantfw run-experiment experiments/${MODEL_ID}_grid_train.yaml

echo ""
echo "Step 3: View top performers"
echo "-----------------------------------------"
quantfw show-experiment experiments/${MODEL_ID}_grid_train --metric sharpe --top-n 5

echo ""
echo "Step 4: Generate walk-forward experiments"
echo "-----------------------------------------"
python scripts/generate_walk_forward.py \
  --model-id $MODEL_ID \
  --params lookback_days=20,30,60 top_n_sectors=3,5 \
  --tickers $TICKERS \
  --start-date $TRAIN_START \
  --end-date $TRAIN_END \
  --train-months 6 \
  --test-months 3 \
  --experiment-id-prefix "${MODEL_ID}_wf" \
  --output-dir experiments/${MODEL_ID}_walk_forward

echo ""
echo "Step 5: Run walk-forward train windows"
echo "-----------------------------------------"
for train_file in experiments/${MODEL_ID}_walk_forward/train_*.yaml; do
  echo "Running $train_file..."
  quantfw run-experiment "$train_file"
done

echo ""
echo "Step 6: Analyze parameter stability"
echo "-----------------------------------------"
python scripts/analyze_param_stability.py \
  experiments/${MODEL_ID}_walk_forward/${MODEL_ID}_wf_train_* \
  --metric sharpe \
  --top-n 3

echo ""
echo "Step 7: Generate out-of-sample test"
echo "-----------------------------------------"
# Use the most stable parameters found (example: lookback_days=30, top_n_sectors=3)
python scripts/generate_param_sweep.py \
  --model-id $MODEL_ID \
  --params lookback_days=30 top_n_sectors=3 \
  --tickers $TICKERS \
  --start-date $TEST_START \
  --end-date $TEST_END \
  --experiment-id "${MODEL_ID}_oos_test" \
  --output experiments/${MODEL_ID}_oos_test.yaml

echo ""
echo "Step 8: Run out-of-sample test"
echo "-----------------------------------------"
quantfw run-experiment experiments/${MODEL_ID}_oos_test.yaml

echo ""
echo "Step 9: Compare train vs test performance"
echo "-----------------------------------------"
echo "Training period (in-sample):"
quantfw show-experiment experiments/${MODEL_ID}_grid_train --metric sharpe --top-n 3

echo ""
echo "Test period (out-of-sample):"
quantfw show-experiment experiments/${MODEL_ID}_oos_test --metric sharpe

echo ""
echo "Step 10: Generate detailed reports"
echo "-----------------------------------------"
quantfw experiment-report experiments/${MODEL_ID}_grid_train \
  --output-file experiments/${MODEL_ID}_grid_train_report.txt

quantfw experiment-report experiments/${MODEL_ID}_oos_test \
  --output-file experiments/${MODEL_ID}_oos_test_report.txt

echo ""
echo "========================================="
echo "Parameter Optimization Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - experiments/${MODEL_ID}_grid_train/"
echo "  - experiments/${MODEL_ID}_walk_forward/"
echo "  - experiments/${MODEL_ID}_oos_test/"
echo ""
echo "Reports saved to:"
echo "  - experiments/${MODEL_ID}_grid_train_report.txt"
echo "  - experiments/${MODEL_ID}_oos_test_report.txt"
echo ""
echo "Next steps:"
echo "  1. Review parameter stability from walk-forward analysis"
echo "  2. Compare in-sample vs out-of-sample performance"
echo "  3. Check for overfitting (large performance gap = overfit)"
echo "  4. Consider economic rationale for optimal parameters"
