"""
Comprehensive uncertainty quantification for observers.

Provides multiple methods for estimating and calibrating uncertainty,
including ensemble methods, Bayesian approaches, and calibration metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ro_framework.core.dof import DoF
from ro_framework.core.state import State


@dataclass
class UncertaintyEstimate:
    """
    Comprehensive uncertainty estimate for a prediction.

    Includes both aleatoric (data) and epistemic (model) uncertainty.
    """
    dof: DoF
    prediction: float
    aleatoric_uncertainty: float  # Inherent data noise
    epistemic_uncertainty: float  # Model uncertainty
    total_uncertainty: float  # Combined uncertainty

    @property
    def confidence(self) -> float:
        """Confidence level (1 - total_uncertainty)."""
        return max(0.0, 1.0 - self.total_uncertainty)


@dataclass
class CalibrationMetrics:
    """
    Metrics for evaluating uncertainty calibration.

    Well-calibrated uncertainty means predicted uncertainty matches actual error.
    """
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    negative_log_likelihood: float  # NLL
    brier_score: Optional[float] = None  # For classification
    sharpness: float = 0.0  # Average confidence
    num_bins: int = 10

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """Check if model is well-calibrated."""
        return self.expected_calibration_error < threshold


class UncertaintyQuantifier(ABC):
    """
    Abstract base class for uncertainty quantification methods.

    Quantifies both aleatoric and epistemic uncertainty.
    """

    def __init__(self, name: str):
        """
        Initialize uncertainty quantifier.

        Args:
            name: Name of the method
        """
        self.name = name

    @abstractmethod
    def estimate_uncertainty(
        self,
        model: Any,
        input_state: State,
        output_dofs: List[DoF],
        **kwargs
    ) -> Dict[DoF, UncertaintyEstimate]:
        """
        Estimate uncertainty for model predictions.

        Args:
            model: Model to evaluate
            input_state: Input state
            output_dofs: Output DoFs to estimate uncertainty for
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping DoFs to uncertainty estimates
        """
        pass

    @abstractmethod
    def calibrate(
        self,
        predictions: List[float],
        uncertainties: List[float],
        ground_truth: List[float],
        **kwargs
    ) -> CalibrationMetrics:
        """
        Evaluate calibration of uncertainty estimates.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            ground_truth: True values
            **kwargs: Additional arguments

        Returns:
            Calibration metrics
        """
        pass


if TORCH_AVAILABLE:

    class EnsembleUncertainty(UncertaintyQuantifier):
        """
        Ensemble-based uncertainty quantification.

        Uses multiple models to estimate epistemic uncertainty.
        Aleatoric uncertainty estimated from ensemble variance.
        """

        def __init__(
            self,
            models: List[nn.Module],
            device: str = "cpu",
        ):
            """
            Initialize ensemble uncertainty.

            Args:
                models: List of ensemble models
                device: Device for computation
            """
            super().__init__("ensemble")
            self.models = [model.to(device) for model in models]
            self.device = device

            for model in self.models:
                model.eval()

        def estimate_uncertainty(
            self,
            model: Any,  # Not used, uses ensemble instead
            input_state: State,
            output_dofs: List[DoF],
            **kwargs
        ) -> Dict[DoF, UncertaintyEstimate]:
            """
            Estimate uncertainty using ensemble.

            Args:
                model: Ignored (uses ensemble)
                input_state: Input state
                output_dofs: Output DoFs
                **kwargs: Additional arguments

            Returns:
                Uncertainty estimates for each DoF
            """
            # Convert state to tensor
            dof_order = list(input_state.values.keys())
            input_vector = input_state.to_vector(dof_order)
            input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0).to(self.device)

            # Get predictions from all models
            predictions = []
            with torch.no_grad():
                for model in self.models:
                    output = model(input_tensor)
                    predictions.append(output.cpu().numpy().flatten())

            predictions = np.array(predictions)  # Shape: (n_models, n_outputs)

            # Compute uncertainties
            uncertainties = {}
            for i, dof in enumerate(output_dofs):
                if i >= predictions.shape[1]:
                    continue

                # Mean prediction
                mean_pred = np.mean(predictions[:, i])

                # Epistemic uncertainty (variance across models)
                epistemic = np.std(predictions[:, i])

                # Aleatoric uncertainty (average prediction variance)
                # For ensemble, approximate as fraction of epistemic
                aleatoric = epistemic * 0.3

                # Total uncertainty
                total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

                uncertainties[dof] = UncertaintyEstimate(
                    dof=dof,
                    prediction=float(mean_pred),
                    aleatoric_uncertainty=float(aleatoric),
                    epistemic_uncertainty=float(epistemic),
                    total_uncertainty=float(total),
                )

            return uncertainties

        def calibrate(
            self,
            predictions: List[float],
            uncertainties: List[float],
            ground_truth: List[float],
            num_bins: int = 10,
            **kwargs
        ) -> CalibrationMetrics:
            """
            Evaluate calibration using binning.

            Args:
                predictions: Model predictions
                uncertainties: Uncertainty estimates
                ground_truth: True values
                num_bins: Number of bins for ECE calculation
                **kwargs: Additional arguments

            Returns:
                Calibration metrics
            """
            predictions = np.array(predictions)
            uncertainties = np.array(uncertainties)
            ground_truth = np.array(ground_truth)

            # Compute errors
            errors = np.abs(predictions - ground_truth)

            # Expected Calibration Error (ECE)
            ece = self._compute_ece(uncertainties, errors, num_bins)

            # Maximum Calibration Error (MCE)
            mce = self._compute_mce(uncertainties, errors, num_bins)

            # Negative Log Likelihood (for Gaussian assumption)
            nll = self._compute_nll(predictions, uncertainties, ground_truth)

            # Sharpness (average confidence)
            sharpness = float(np.mean(1.0 - uncertainties))

            return CalibrationMetrics(
                expected_calibration_error=float(ece),
                maximum_calibration_error=float(mce),
                negative_log_likelihood=float(nll),
                sharpness=sharpness,
                num_bins=num_bins,
            )

        def _compute_ece(
            self,
            uncertainties: np.ndarray,
            errors: np.ndarray,
            num_bins: int
        ) -> float:
            """Compute Expected Calibration Error."""
            # Bin by uncertainty
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(uncertainties, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            ece = 0.0
            for i in range(num_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) == 0:
                    continue

                bin_uncertainties = uncertainties[bin_mask]
                bin_errors = errors[bin_mask]

                # Average uncertainty and error in bin
                avg_uncertainty = np.mean(bin_uncertainties)
                avg_error = np.mean(bin_errors)

                # Weighted contribution to ECE
                weight = np.sum(bin_mask) / len(uncertainties)
                ece += weight * np.abs(avg_uncertainty - avg_error)

            return float(ece)

        def _compute_mce(
            self,
            uncertainties: np.ndarray,
            errors: np.ndarray,
            num_bins: int
        ) -> float:
            """Compute Maximum Calibration Error."""
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(uncertainties, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            mce = 0.0
            for i in range(num_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) == 0:
                    continue

                bin_uncertainties = uncertainties[bin_mask]
                bin_errors = errors[bin_mask]

                avg_uncertainty = np.mean(bin_uncertainties)
                avg_error = np.mean(bin_errors)

                mce = max(mce, np.abs(avg_uncertainty - avg_error))

            return float(mce)

        def _compute_nll(
            self,
            predictions: np.ndarray,
            uncertainties: np.ndarray,
            ground_truth: np.ndarray
        ) -> float:
            """Compute Negative Log Likelihood (Gaussian assumption)."""
            # Assume Gaussian: p(y|pred, unc) = N(pred, unc^2)
            variance = uncertainties ** 2 + 1e-8  # Add small epsilon
            nll = 0.5 * np.log(2 * np.pi * variance) + \
                  0.5 * ((predictions - ground_truth) ** 2) / variance
            return float(np.mean(nll))


    class BayesianUncertainty(UncertaintyQuantifier):
        """
        Bayesian uncertainty quantification using MC Dropout or variational inference.

        Estimates epistemic uncertainty through multiple stochastic forward passes.
        """

        def __init__(
            self,
            n_samples: int = 10,
            device: str = "cpu",
        ):
            """
            Initialize Bayesian uncertainty.

            Args:
                n_samples: Number of stochastic forward passes
                device: Device for computation
            """
            super().__init__("bayesian")
            self.n_samples = n_samples
            self.device = device

        def estimate_uncertainty(
            self,
            model: nn.Module,
            input_state: State,
            output_dofs: List[DoF],
            **kwargs
        ) -> Dict[DoF, UncertaintyEstimate]:
            """
            Estimate uncertainty using MC Dropout.

            Args:
                model: Model with dropout layers
                input_state: Input state
                output_dofs: Output DoFs
                **kwargs: Additional arguments

            Returns:
                Uncertainty estimates for each DoF
            """
            model = model.to(self.device)
            model.train()  # Enable dropout

            # Convert state to tensor
            dof_order = list(input_state.values.keys())
            input_vector = input_state.to_vector(dof_order)
            input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0).to(self.device)

            # Multiple stochastic forward passes
            predictions = []
            with torch.no_grad():
                for _ in range(self.n_samples):
                    output = model(input_tensor)
                    predictions.append(output.cpu().numpy().flatten())

            predictions = np.array(predictions)  # Shape: (n_samples, n_outputs)
            model.eval()  # Disable dropout

            # Compute uncertainties
            uncertainties = {}
            for i, dof in enumerate(output_dofs):
                if i >= predictions.shape[1]:
                    continue

                # Mean prediction
                mean_pred = np.mean(predictions[:, i])

                # Epistemic uncertainty (variance across samples)
                epistemic = np.std(predictions[:, i])

                # Aleatoric uncertainty (mean of predictive variance)
                # For MC Dropout, approximate as fraction of total variance
                aleatoric = epistemic * 0.2

                # Total uncertainty
                total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

                uncertainties[dof] = UncertaintyEstimate(
                    dof=dof,
                    prediction=float(mean_pred),
                    aleatoric_uncertainty=float(aleatoric),
                    epistemic_uncertainty=float(epistemic),
                    total_uncertainty=float(total),
                )

            return uncertainties

        def calibrate(
            self,
            predictions: List[float],
            uncertainties: List[float],
            ground_truth: List[float],
            num_bins: int = 10,
            **kwargs
        ) -> CalibrationMetrics:
            """
            Evaluate calibration (same as ensemble).

            Args:
                predictions: Model predictions
                uncertainties: Uncertainty estimates
                ground_truth: True values
                num_bins: Number of bins
                **kwargs: Additional arguments

            Returns:
                Calibration metrics
            """
            # Use same calibration logic as ensemble
            ensemble_quantifier = EnsembleUncertainty(models=[], device=self.device)
            return ensemble_quantifier.calibrate(
                predictions, uncertainties, ground_truth, num_bins
            )


def compute_predictive_entropy(
    predictions: np.ndarray,
    bins: int = 50
) -> float:
    """
    Compute predictive entropy from a set of predictions.

    Args:
        predictions: Array of predictions (can be continuous or discrete)
        bins: Number of bins for histogram (for continuous predictions)

    Returns:
        Predictive entropy
    """
    # Create histogram
    hist, _ = np.histogram(predictions, bins=bins, density=True)
    hist = hist + 1e-10  # Avoid log(0)

    # Normalize
    hist = hist / hist.sum()

    # Compute entropy
    entropy = -np.sum(hist * np.log(hist))
    return float(entropy)


def compute_mutual_information(
    predictions: np.ndarray,
    n_classes: Optional[int] = None
) -> float:
    """
    Compute mutual information between predictions (epistemic uncertainty measure).

    For classification: I(y; θ | x) = H(E[p(y|x,θ)]) - E[H(p(y|x,θ))]

    Args:
        predictions: Array of shape (n_samples, n_outputs) or (n_samples, n_classes)
        n_classes: Number of classes (for classification)

    Returns:
        Mutual information
    """
    if n_classes is not None:
        # Classification: compute MI over class probabilities
        mean_probs = np.mean(predictions, axis=0)
        entropy_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

        entropies = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        mean_entropy = np.mean(entropies)

        mi = entropy_mean - mean_entropy
    else:
        # Regression: approximate MI using variance
        mi = np.mean(np.var(predictions, axis=0))

    return float(mi)


def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    initial_temperature: float = 1.0,
    max_iter: int = 50
) -> float:
    """
    Calibrate model using temperature scaling.

    Finds optimal temperature T to minimize NLL: softmax(logits / T)

    Args:
        logits: Model logits (before softmax)
        labels: True labels (one-hot or class indices)
        initial_temperature: Initial temperature value
        max_iter: Maximum optimization iterations

    Returns:
        Optimal temperature
    """
    if not TORCH_AVAILABLE:
        return initial_temperature

    logits_tensor = torch.from_numpy(logits).float()
    if labels.ndim == 1:
        labels_tensor = torch.from_numpy(labels).long()
    else:
        labels_tensor = torch.from_numpy(labels).float()

    # Optimize temperature
    temperature = torch.tensor([initial_temperature], requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def eval():
        optimizer.zero_grad()
        scaled_logits = logits_tensor / temperature
        loss = nn.functional.cross_entropy(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval)

    return float(temperature.item())


def compute_coverage(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    ground_truth: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Compute empirical coverage of confidence intervals.

    For well-calibrated uncertainty, coverage should match confidence level.

    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates (standard deviation)
        ground_truth: True values
        confidence_level: Desired confidence level (e.g., 0.95 for 95%)

    Returns:
        Empirical coverage (fraction of true values within confidence intervals)
    """
    from scipy import stats

    # Compute confidence interval half-width
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    half_width = z_score * uncertainties

    # Check if ground truth within interval
    lower = predictions - half_width
    upper = predictions + half_width

    within_interval = (ground_truth >= lower) & (ground_truth <= upper)
    coverage = np.mean(within_interval)

    return float(coverage)


def decompose_uncertainty(
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Decompose total uncertainty into aleatoric and epistemic components.

    Uses variance decomposition: Var[y] = E[Var[y|θ]] + Var[E[y|θ]]

    Args:
        predictions: Array of predictions (n_samples, n_outputs)
        ground_truth: Optional true values for validation

    Returns:
        Dictionary with uncertainty components
    """
    # Epistemic uncertainty (variance of means)
    epistemic = np.var(np.mean(predictions, axis=1))

    # Aleatoric uncertainty (mean of variances)
    aleatoric = np.mean(np.var(predictions, axis=0))

    # Total uncertainty
    total = np.var(predictions)

    result = {
        "epistemic": float(epistemic),
        "aleatoric": float(aleatoric),
        "total": float(total),
    }

    if ground_truth is not None:
        # Compute actual error
        mean_pred = np.mean(predictions, axis=0)
        mse = np.mean((mean_pred - ground_truth) ** 2)
        result["mse"] = float(mse)

    return result
