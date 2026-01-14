"""
Unit tests for uncertainty quantification.

Tests ensemble uncertainty, Bayesian uncertainty, and calibration metrics.
"""

import pytest
import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State

# Check if torch is available
try:
    import torch
    import torch.nn as nn

    from ro_framework.multimodal.uncertainty import (
        UncertaintyEstimate,
        CalibrationMetrics,
        EnsembleUncertainty,
        BayesianUncertainty,
        compute_predictive_entropy,
        compute_mutual_information,
        temperature_scaling,
        compute_coverage,
        decompose_uncertainty,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate."""

    def test_uncertainty_estimate_creation(self):
        """Test creating an uncertainty estimate."""
        dof = PolarDoF(name="test_dof", description="Test DoF")

        estimate = UncertaintyEstimate(
            dof=dof,
            prediction=0.5,
            aleatoric_uncertainty=0.1,
            epistemic_uncertainty=0.2,
            total_uncertainty=0.22
        )

        assert estimate.dof == dof
        assert estimate.prediction == 0.5
        assert estimate.aleatoric_uncertainty == 0.1
        assert estimate.epistemic_uncertainty == 0.2
        assert estimate.total_uncertainty == 0.22

    def test_uncertainty_estimate_confidence(self):
        """Test confidence property."""
        dof = PolarDoF(name="test", description="")

        estimate = UncertaintyEstimate(
            dof=dof,
            prediction=0.5,
            aleatoric_uncertainty=0.1,
            epistemic_uncertainty=0.2,
            total_uncertainty=0.3
        )

        assert estimate.confidence == 0.7  # 1 - 0.3


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics."""

    def test_calibration_metrics_creation(self):
        """Test creating calibration metrics."""
        metrics = CalibrationMetrics(
            expected_calibration_error=0.05,
            maximum_calibration_error=0.1,
            negative_log_likelihood=0.5,
            sharpness=0.8,
            num_bins=10
        )

        assert metrics.expected_calibration_error == 0.05
        assert metrics.maximum_calibration_error == 0.1
        assert metrics.negative_log_likelihood == 0.5
        assert metrics.sharpness == 0.8

    def test_is_well_calibrated(self):
        """Test well-calibrated check."""
        # Well calibrated
        metrics1 = CalibrationMetrics(
            expected_calibration_error=0.05,
            maximum_calibration_error=0.1,
            negative_log_likelihood=0.5,
        )
        assert metrics1.is_well_calibrated(threshold=0.1)

        # Poorly calibrated
        metrics2 = CalibrationMetrics(
            expected_calibration_error=0.15,
            maximum_calibration_error=0.2,
            negative_log_likelihood=1.0,
        )
        assert not metrics2.is_well_calibrated(threshold=0.1)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEnsembleUncertainty:
    """Tests for EnsembleUncertainty."""

    def test_ensemble_uncertainty_creation(self):
        """Test creating ensemble uncertainty quantifier."""
        models = [
            nn.Sequential(nn.Linear(10, 5))
            for _ in range(3)
        ]

        quantifier = EnsembleUncertainty(models=models, device="cpu")

        assert quantifier.name == "ensemble"
        assert len(quantifier.models) == 3

    def test_ensemble_estimate_uncertainty(self):
        """Test uncertainty estimation with ensemble."""
        models = [
            nn.Sequential(nn.Linear(10, 3))
            for _ in range(5)
        ]

        quantifier = EnsembleUncertainty(models=models, device="cpu")

        # Create test input
        input_dofs = [PolarDoF(name=f"in{i}", description="") for i in range(10)]
        output_dofs = [PolarDoF(name=f"out{i}", description="") for i in range(3)]

        input_state = State(dof_values={
            dof: float(np.random.randn()) for dof in input_dofs
        })

        # Estimate uncertainty
        uncertainties = quantifier.estimate_uncertainty(
            model=None,  # Unused
            input_state=input_state,
            output_dofs=output_dofs
        )

        assert len(uncertainties) == 3
        for dof, estimate in uncertainties.items():
            assert isinstance(estimate, UncertaintyEstimate)
            assert estimate.epistemic_uncertainty >= 0.0
            assert estimate.aleatoric_uncertainty >= 0.0
            assert estimate.total_uncertainty >= 0.0

    def test_ensemble_calibrate(self):
        """Test calibration evaluation."""
        models = [nn.Sequential(nn.Linear(10, 1)) for _ in range(3)]
        quantifier = EnsembleUncertainty(models=models, device="cpu")

        # Generate synthetic data
        np.random.seed(42)
        predictions = np.random.randn(100)
        uncertainties = np.abs(np.random.randn(100)) * 0.2
        ground_truth = predictions + np.random.randn(100) * 0.1

        metrics = quantifier.calibrate(
            predictions=predictions.tolist(),
            uncertainties=uncertainties.tolist(),
            ground_truth=ground_truth.tolist(),
            num_bins=10
        )

        assert isinstance(metrics, CalibrationMetrics)
        assert 0.0 <= metrics.expected_calibration_error <= 1.0
        assert 0.0 <= metrics.maximum_calibration_error <= 1.0
        assert metrics.negative_log_likelihood > 0.0

    def test_compute_ece(self):
        """Test ECE computation."""
        models = [nn.Linear(5, 1)]
        quantifier = EnsembleUncertainty(models=models, device="cpu")

        uncertainties = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        errors = np.array([0.15, 0.25, 0.35, 0.35, 0.45])

        ece = quantifier._compute_ece(uncertainties, errors, num_bins=5)

        assert isinstance(ece, float)
        assert ece >= 0.0

    def test_compute_nll(self):
        """Test NLL computation."""
        models = [nn.Linear(5, 1)]
        quantifier = EnsembleUncertainty(models=models, device="cpu")

        predictions = np.array([0.0, 1.0, 2.0])
        uncertainties = np.array([0.1, 0.2, 0.3])
        ground_truth = np.array([0.1, 1.1, 1.9])

        nll = quantifier._compute_nll(predictions, uncertainties, ground_truth)

        assert isinstance(nll, float)
        assert nll > 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBayesianUncertainty:
    """Tests for BayesianUncertainty."""

    def test_bayesian_uncertainty_creation(self):
        """Test creating Bayesian uncertainty quantifier."""
        quantifier = BayesianUncertainty(n_samples=10, device="cpu")

        assert quantifier.name == "bayesian"
        assert quantifier.n_samples == 10

    def test_bayesian_estimate_uncertainty(self):
        """Test MC Dropout uncertainty estimation."""
        model = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 3)
        )

        quantifier = BayesianUncertainty(n_samples=5, device="cpu")

        input_dofs = [PolarDoF(name=f"in{i}", description="") for i in range(10)]
        output_dofs = [PolarDoF(name=f"out{i}", description="") for i in range(3)]

        input_state = State(dof_values={
            dof: float(np.random.randn()) for dof in input_dofs
        })

        uncertainties = quantifier.estimate_uncertainty(
            model=model,
            input_state=input_state,
            output_dofs=output_dofs
        )

        assert len(uncertainties) == 3
        for estimate in uncertainties.values():
            assert isinstance(estimate, UncertaintyEstimate)
            assert estimate.epistemic_uncertainty >= 0.0


def test_compute_predictive_entropy():
    """Test predictive entropy computation."""
    # Deterministic predictions (low entropy)
    predictions1 = np.ones(100)
    entropy1 = compute_predictive_entropy(predictions1, bins=10)
    assert entropy1 >= 0.0

    # Uniform predictions (high entropy)
    predictions2 = np.random.rand(100)
    entropy2 = compute_predictive_entropy(predictions2, bins=10)
    assert entropy2 > entropy1


def test_compute_mutual_information():
    """Test mutual information computation."""
    # Low MI (similar predictions)
    predictions1 = np.ones((50, 10)) + np.random.randn(50, 10) * 0.01
    mi1 = compute_mutual_information(predictions1)
    assert mi1 >= 0.0

    # High MI (diverse predictions)
    predictions2 = np.random.randn(50, 10)
    mi2 = compute_mutual_information(predictions2)
    assert mi2 > mi1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_temperature_scaling():
    """Test temperature scaling calibration."""
    # Create synthetic logits and labels
    logits = np.random.randn(50, 5)
    labels = np.random.randint(0, 5, size=50)

    temperature = temperature_scaling(logits, labels, max_iter=10)

    assert isinstance(temperature, float)
    assert temperature > 0.0


def test_compute_coverage():
    """Test coverage computation."""
    np.random.seed(42)

    # Generate well-calibrated predictions
    predictions = np.random.randn(100)
    uncertainties = np.ones(100) * 0.5
    ground_truth = predictions + np.random.randn(100) * 0.3

    coverage = compute_coverage(
        predictions=predictions,
        uncertainties=uncertainties,
        ground_truth=ground_truth,
        confidence_level=0.95
    )

    assert isinstance(coverage, float)
    assert 0.0 <= coverage <= 1.0


def test_decompose_uncertainty():
    """Test uncertainty decomposition."""
    # Create predictions with known variance structure
    predictions = np.random.randn(50, 10)

    decomposition = decompose_uncertainty(predictions)

    assert "epistemic" in decomposition
    assert "aleatoric" in decomposition
    assert "total" in decomposition

    assert decomposition["epistemic"] >= 0.0
    assert decomposition["aleatoric"] >= 0.0
    assert decomposition["total"] >= 0.0

    # With ground truth
    ground_truth = np.random.randn(10)
    decomposition_with_gt = decompose_uncertainty(predictions, ground_truth)

    assert "mse" in decomposition_with_gt
    assert decomposition_with_gt["mse"] >= 0.0


def test_decompose_uncertainty_perfect_predictions():
    """Test decomposition with perfect predictions."""
    # All predictions are the same (no epistemic uncertainty)
    predictions = np.ones((50, 5))

    decomposition = decompose_uncertainty(predictions)

    # Epistemic should be ~0 (all models agree)
    assert decomposition["epistemic"] < 0.01


def test_decompose_uncertainty_high_disagreement():
    """Test decomposition with high model disagreement."""
    # High variance across samples (high epistemic uncertainty)
    predictions = np.random.randn(50, 5) * 10

    decomposition = decompose_uncertainty(predictions)

    # Epistemic should be high
    assert decomposition["epistemic"] > 1.0
