"""Analyze parameter stability across walk-forward windows.

This script helps identify which parameters are consistently optimal across
different time periods, indicating more robust parameter choices.

Usage:
    python scripts/analyze_param_stability.py \\
        experiments/walk_forward/train_*.yaml \\
        --metric sharpe \\
        --top-n 3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def load_experiment_results(experiment_dir: Path) -> pd.DataFrame:
    """Load comparison results from an experiment directory."""
    comparison_file = experiment_dir / "comparison.csv"
    if not comparison_file.exists():
        raise ValueError(f"Comparison file not found: {comparison_file}")
    return pd.read_csv(comparison_file)


def extract_params_from_run_id(run_id: str) -> dict:
    """Extract parameter values from run_id.

    Assumes run_id format includes params, e.g.:
    'sector_momentum_v1_uuid_lookback_days=30_top_n_sectors=3'
    """
    params = {}
    parts = run_id.split('_')

    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            # Try to convert to number
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value

    return params


def analyze_parameter_stability(
    experiment_dirs: list[Path],
    metric: str = "sharpe",
    top_n: int = 3
) -> dict:
    """Analyze which parameters appear consistently in top performers."""

    window_results = []

    for exp_dir in experiment_dirs:
        try:
            df = load_experiment_results(exp_dir)

            # Sort by metric and get top N
            df_sorted = df.sort_values(metric, ascending=False, na_last=True)
            top_runs = df_sorted.head(top_n)

            window_results.append({
                "experiment": exp_dir.name,
                "top_runs": top_runs,
                "best_value": top_runs.iloc[0][metric] if len(top_runs) > 0 else None
            })
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")

    if not window_results:
        return {}

    # Track parameter combinations that appear in top N
    param_combo_counts = Counter()
    param_value_counts = defaultdict(Counter)

    for window in window_results:
        for _, run in window["top_runs"].iterrows():
            # Extract params from run_id or model_config column
            # This is a simplified version - adjust based on your actual data format
            run_id = run.get("run_id", "")
            params = extract_params_from_run_id(run_id)

            # Count full parameter combinations
            param_tuple = tuple(sorted(params.items()))
            param_combo_counts[param_tuple] += 1

            # Count individual parameter values
            for param_name, param_value in params.items():
                param_value_counts[param_name][param_value] += 1

    return {
        "windows_analyzed": len(window_results),
        "top_n_per_window": top_n,
        "metric": metric,
        "param_combinations": param_combo_counts,
        "param_value_frequencies": dict(param_value_counts),
        "window_details": window_results
    }


def print_stability_report(analysis: dict):
    """Print a formatted stability analysis report."""

    print("=" * 80)
    print("PARAMETER STABILITY ANALYSIS")
    print("=" * 80)
    print()

    print(f"Windows analyzed: {analysis['windows_analyzed']}")
    print(f"Top N per window: {analysis['top_n_per_window']}")
    print(f"Metric: {analysis['metric']}")
    print()

    # Most consistent parameter combinations
    print("-" * 80)
    print("MOST CONSISTENT PARAMETER COMBINATIONS")
    print("-" * 80)
    print()

    if analysis['param_combinations']:
        sorted_combos = sorted(
            analysis['param_combinations'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for combo, count in sorted_combos[:10]:
            params_dict = dict(combo)
            consistency_pct = (count / analysis['windows_analyzed']) * 100
            print(f"Appeared in {count}/{analysis['windows_analyzed']} windows ({consistency_pct:.1f}%):")
            for param, value in sorted(params_dict.items()):
                print(f"  {param}: {value}")
            print()
    else:
        print("No parameter combinations found")
        print()

    # Individual parameter value frequencies
    print("-" * 80)
    print("PARAMETER VALUE FREQUENCIES")
    print("-" * 80)
    print()

    for param_name, value_counts in sorted(analysis['param_value_frequencies'].items()):
        print(f"{param_name}:")
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        for value, count in sorted_values:
            consistency_pct = (count / (analysis['windows_analyzed'] * analysis['top_n_per_window'])) * 100
            print(f"  {value}: {count} occurrences ({consistency_pct:.1f}%)")
        print()

    # Window-by-window best results
    print("-" * 80)
    print("WINDOW-BY-WINDOW BEST RESULTS")
    print("-" * 80)
    print()

    for window in analysis['window_details']:
        print(f"{window['experiment']}: {analysis['metric']} = {window['best_value']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter stability across walk-forward windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # After running all train experiments
  python scripts/analyze_param_stability.py \\
      experiments/walk_forward/train_01 \\
      experiments/walk_forward/train_02 \\
      experiments/walk_forward/train_03 \\
      --metric sharpe \\
      --top-n 3

Note: This script looks for comparison.csv files in each experiment directory.
Make sure you've run the experiments first using 'quantfw run-experiment'.
        """
    )

    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="Paths to experiment output directories"
    )

    parser.add_argument(
        "--metric",
        default="sharpe",
        help="Metric to analyze (default: sharpe)"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top performers to analyze per window (default: 3)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to file (optional)"
    )

    args = parser.parse_args()

    # Filter to valid directories
    valid_dirs = [d for d in args.experiment_dirs if d.is_dir()]
    if not valid_dirs:
        parser.error("No valid experiment directories found")

    print(f"Analyzing {len(valid_dirs)} experiment directories...")
    print()

    # Perform analysis
    analysis = analyze_parameter_stability(
        valid_dirs,
        metric=args.metric,
        top_n=args.top_n
    )

    if not analysis:
        print("No analysis results generated. Check that experiment directories contain comparison.csv files.")
        return

    # Print report
    print_stability_report(analysis)

    # Optionally save to file
    if args.output:
        # TODO: Implement formatted text export
        print(f"Note: File export not yet implemented. Report shown above.")


if __name__ == "__main__":
    main()
