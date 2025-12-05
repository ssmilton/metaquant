"""Tests for experiment functionality."""
import pytest
from pathlib import Path

from core.experiments import (
    ExperimentConfig,
    ParameterSweep,
    ModelRunSpec,
    UniverseSpec,
    BenchmarkSpec,
)


class TestUniverseSpec:
    """Tests for UniverseSpec validation."""

    def test_validate_exclusive_security_ids(self):
        """Test that security_ids is valid."""
        spec = UniverseSpec(security_ids=[1, 2, 3])
        spec.validate_exclusive()  # Should not raise

    def test_validate_exclusive_tickers(self):
        """Test that tickers is valid."""
        spec = UniverseSpec(tickers=["AAPL", "MSFT"])
        spec.validate_exclusive()  # Should not raise

    def test_validate_exclusive_tickers_file(self):
        """Test that tickers_file is valid."""
        spec = UniverseSpec(tickers_file="tickers.txt")
        spec.validate_exclusive()  # Should not raise

    def test_validate_exclusive_none(self):
        """Test that no specification raises error."""
        spec = UniverseSpec()
        with pytest.raises(ValueError, match="Must specify one of"):
            spec.validate_exclusive()

    def test_validate_exclusive_multiple(self):
        """Test that multiple specifications raise error."""
        spec = UniverseSpec(security_ids=[1, 2], tickers=["AAPL"])
        with pytest.raises(ValueError, match="Cannot specify more than one"):
            spec.validate_exclusive()


class TestParameterSweep:
    """Tests for ParameterSweep run generation."""

    def test_generate_runs_single_param(self):
        """Test sweep with single parameter."""
        sweep = ParameterSweep(
            model_id="test_model",
            base_parameters={"param_a": 10},
            sweep_parameters={"param_b": [1, 2, 3]}
        )

        runs = sweep.generate_runs()

        assert len(runs) == 3
        assert all(r.model_id == "test_model" for r in runs)
        assert runs[0].parameters == {"param_a": 10, "param_b": 1}
        assert runs[1].parameters == {"param_a": 10, "param_b": 2}
        assert runs[2].parameters == {"param_a": 10, "param_b": 3}

    def test_generate_runs_multiple_params(self):
        """Test sweep with multiple parameters (cartesian product)."""
        sweep = ParameterSweep(
            model_id="test_model",
            base_parameters={},
            sweep_parameters={
                "param_a": [1, 2],
                "param_b": [10, 20]
            }
        )

        runs = sweep.generate_runs()

        assert len(runs) == 4  # 2 x 2 = 4 combinations
        assert all(r.model_id == "test_model" for r in runs)

        # Check all combinations exist
        param_combos = [r.parameters for r in runs]
        assert {"param_a": 1, "param_b": 10} in param_combos
        assert {"param_a": 1, "param_b": 20} in param_combos
        assert {"param_a": 2, "param_b": 10} in param_combos
        assert {"param_a": 2, "param_b": 20} in param_combos

    def test_generate_runs_names(self):
        """Test that generated runs have descriptive names."""
        sweep = ParameterSweep(
            model_id="test_model",
            sweep_parameters={"lookback": [10, 20]}
        )

        runs = sweep.generate_runs()

        assert runs[0].name == "test_model[lookback=10]"
        assert runs[1].name == "test_model[lookback=20]"

    def test_generate_runs_empty_sweep(self):
        """Test that empty sweep parameters generates single run."""
        sweep = ParameterSweep(
            model_id="test_model",
            base_parameters={"param_a": 10},
            sweep_parameters={}
        )

        runs = sweep.generate_runs()

        assert len(runs) == 1
        assert runs[0].parameters == {"param_a": 10}


class TestModelRunSpec:
    """Tests for ModelRunSpec."""

    def test_display_name_default(self):
        """Test display name defaults to model_id."""
        spec = ModelRunSpec(model_id="test_model", parameters={})
        assert spec.display_name == "test_model"

    def test_display_name_custom(self):
        """Test custom display name."""
        spec = ModelRunSpec(model_id="test_model", name="Custom Name", parameters={})
        assert spec.display_name == "Custom Name"


class TestBenchmarkSpec:
    """Tests for BenchmarkSpec."""

    def test_display_name_buy_and_hold(self):
        """Test buy and hold benchmark display name."""
        spec = BenchmarkSpec(type="buy_and_hold", ticker="AAPL")
        assert spec.display_name == "Buy & Hold (AAPL)"

    def test_display_name_equal_weight(self):
        """Test equal weight benchmark display name."""
        spec = BenchmarkSpec(type="equal_weight")
        assert spec.display_name == "Equal Weight Portfolio"

    def test_display_name_custom(self):
        """Test custom benchmark name."""
        spec = BenchmarkSpec(type="buy_and_hold", ticker="AAPL", name="Apple Benchmark")
        assert spec.display_name == "Apple Benchmark"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_get_all_runs_models_only(self):
        """Test get_all_runs with only individual models."""
        config = ExperimentConfig(
            experiment_id="test",
            description="test",
            universe=UniverseSpec(security_ids=[1, 2]),
            start_date="2020-01-01",
            end_date="2020-12-31",
            models=[
                ModelRunSpec(model_id="model_a", parameters={}),
                ModelRunSpec(model_id="model_b", parameters={})
            ]
        )

        runs = config.get_all_runs()
        assert len(runs) == 2

    def test_get_all_runs_sweeps_only(self):
        """Test get_all_runs with only parameter sweeps."""
        config = ExperimentConfig(
            experiment_id="test",
            description="test",
            universe=UniverseSpec(security_ids=[1, 2]),
            start_date="2020-01-01",
            end_date="2020-12-31",
            parameter_sweeps=[
                ParameterSweep(
                    model_id="model_a",
                    sweep_parameters={"param": [1, 2, 3]}
                )
            ]
        )

        runs = config.get_all_runs()
        assert len(runs) == 3

    def test_get_all_runs_mixed(self):
        """Test get_all_runs with both models and sweeps."""
        config = ExperimentConfig(
            experiment_id="test",
            description="test",
            universe=UniverseSpec(security_ids=[1, 2]),
            start_date="2020-01-01",
            end_date="2020-12-31",
            models=[
                ModelRunSpec(model_id="model_a", parameters={})
            ],
            parameter_sweeps=[
                ParameterSweep(
                    model_id="model_b",
                    sweep_parameters={"param": [1, 2]}
                )
            ]
        )

        runs = config.get_all_runs()
        assert len(runs) == 3  # 1 individual + 2 from sweep

    def test_validate_success(self):
        """Test validation passes for valid config."""
        config = ExperimentConfig(
            experiment_id="test",
            description="test",
            universe=UniverseSpec(security_ids=[1, 2]),
            start_date="2020-01-01",
            end_date="2020-12-31",
            models=[
                ModelRunSpec(model_id="model_a", parameters={})
            ]
        )

        config.validate()  # Should not raise

    def test_validate_no_runs_or_benchmarks(self):
        """Test validation fails with no runs or benchmarks."""
        config = ExperimentConfig(
            experiment_id="test",
            description="test",
            universe=UniverseSpec(security_ids=[1, 2]),
            start_date="2020-01-01",
            end_date="2020-12-31"
        )

        with pytest.raises(ValueError, match="at least one model run or benchmark"):
            config.validate()

    def test_validate_benchmarks_only(self):
        """Test validation passes with only benchmarks."""
        config = ExperimentConfig(
            experiment_id="test",
            description="test",
            universe=UniverseSpec(security_ids=[1, 2]),
            start_date="2020-01-01",
            end_date="2020-12-31",
            benchmarks=[
                BenchmarkSpec(type="buy_and_hold", ticker="AAPL")
            ]
        )

        config.validate()  # Should not raise
