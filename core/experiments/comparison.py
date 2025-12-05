"""Comparison and evaluation utilities for experiments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentComparison:
    """Utilities for comparing and analyzing experiment results."""

    @staticmethod
    def rank_by_metric(
        comparison_df: pd.DataFrame,
        metric: str,
        ascending: bool = False
    ) -> pd.DataFrame:
        """Rank models by a specific metric.

        Args:
            comparison_df: Comparison DataFrame from experiment
            metric: Metric to rank by (e.g., 'sharpe', 'total_return')
            ascending: If True, rank ascending (lower is better)

        Returns:
            Sorted DataFrame with rank column
        """
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison data")

        result = comparison_df.copy()
        result["rank"] = result[metric].rank(ascending=ascending, na_option="bottom")
        return result.sort_values("rank")

    @staticmethod
    def get_top_n(
        comparison_df: pd.DataFrame,
        metric: str,
        n: int = 5,
        ascending: bool = False
    ) -> pd.DataFrame:
        """Get top N models by a metric.

        Args:
            comparison_df: Comparison DataFrame from experiment
            metric: Metric to rank by
            n: Number of top models to return
            ascending: If True, rank ascending (lower is better)

        Returns:
            DataFrame with top N models
        """
        ranked = ExperimentComparison.rank_by_metric(comparison_df, metric, ascending)
        return ranked.head(n)

    @staticmethod
    def compute_summary_stats(comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics across all models.

        Args:
            comparison_df: Comparison DataFrame from experiment

        Returns:
            DataFrame with summary statistics (mean, median, min, max, std)
        """
        # Get numeric columns (metrics)
        numeric_cols = comparison_df.select_dtypes(include=["number"]).columns
        metric_cols = [col for col in numeric_cols if col not in ["rank"]]

        summary = comparison_df[metric_cols].describe().T
        summary["median"] = comparison_df[metric_cols].median()

        # Reorder columns
        column_order = ["mean", "median", "std", "min", "max"]
        existing_cols = [col for col in column_order if col in summary.columns]
        return summary[existing_cols]

    @staticmethod
    def compare_to_benchmark(
        comparison_df: pd.DataFrame,
        benchmark_name: str,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare all models to a specific benchmark.

        Args:
            comparison_df: Comparison DataFrame from experiment
            benchmark_name: Name of the benchmark to compare against
            metrics: Metrics to compare (defaults to all numeric metrics)

        Returns:
            DataFrame with difference from benchmark
        """
        # Find benchmark row
        benchmark_row = comparison_df[comparison_df["name"] == benchmark_name]
        if benchmark_row.empty:
            raise ValueError(f"Benchmark '{benchmark_name}' not found in comparison data")

        # Get metrics to compare
        if metrics is None:
            numeric_cols = comparison_df.select_dtypes(include=["number"]).columns
            metrics = [col for col in numeric_cols if col not in ["rank"]]

        # Calculate differences
        result = comparison_df.copy()
        for metric in metrics:
            if metric in comparison_df.columns:
                benchmark_value = benchmark_row[metric].iloc[0]
                result[f"{metric}_vs_benchmark"] = result[metric] - benchmark_value
                result[f"{metric}_outperformance"] = (
                    (result[metric] / benchmark_value - 1) * 100
                    if benchmark_value != 0 else None
                )

        return result

    @staticmethod
    def generate_leaderboard(
        comparison_df: pd.DataFrame,
        primary_metric: str = "sharpe",
        secondary_metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate a leaderboard sorted by primary metric.

        Args:
            comparison_df: Comparison DataFrame from experiment
            primary_metric: Main metric to sort by
            secondary_metrics: Additional metrics to display

        Returns:
            Formatted leaderboard DataFrame
        """
        if secondary_metrics is None:
            secondary_metrics = ["total_return", "cagr", "volatility", "max_drawdown"]

        # Select columns
        columns = ["name", "type", primary_metric] + [
            m for m in secondary_metrics if m in comparison_df.columns and m != primary_metric
        ]
        columns = [col for col in columns if col in comparison_df.columns]

        # Sort by primary metric
        leaderboard = comparison_df[columns].copy()
        leaderboard = leaderboard.sort_values(primary_metric, ascending=False, na_position="last")

        # Add rank
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

        return leaderboard

    @staticmethod
    def load_comparison(experiment_dir: str | Path) -> pd.DataFrame:
        """Load comparison data from a saved experiment.

        Args:
            experiment_dir: Path to experiment output directory

        Returns:
            Comparison DataFrame
        """
        experiment_path = Path(experiment_dir)
        comparison_path = experiment_path / "comparison.csv"

        if not comparison_path.exists():
            raise FileNotFoundError(f"Comparison file not found: {comparison_path}")

        return pd.read_csv(comparison_path)

    @staticmethod
    def create_performance_report(
        comparison_df: pd.DataFrame,
        output_path: Optional[str | Path] = None
    ) -> str:
        """Create a text performance report.

        Args:
            comparison_df: Comparison DataFrame from experiment
            output_path: Optional path to save the report

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT PERFORMANCE REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        summary = ExperimentComparison.compute_summary_stats(comparison_df)
        lines.append(summary.to_string())
        lines.append("")

        # Leaderboard
        lines.append("LEADERBOARD (by Sharpe Ratio)")
        lines.append("-" * 80)
        leaderboard = ExperimentComparison.generate_leaderboard(comparison_df)
        lines.append(leaderboard.to_string(index=False))
        lines.append("")

        # Best performers by metric
        key_metrics = ["sharpe", "total_return", "cagr"]
        for metric in key_metrics:
            if metric in comparison_df.columns:
                lines.append(f"TOP 3 BY {metric.upper()}")
                lines.append("-" * 80)
                top3 = ExperimentComparison.get_top_n(comparison_df, metric, n=3)
                display_cols = ["name", "type", metric]
                display_cols = [col for col in display_cols if col in top3.columns]
                lines.append(top3[display_cols].to_string(index=False))
                lines.append("")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("Performance report saved to %s", output_path)

        return report
