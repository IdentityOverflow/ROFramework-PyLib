"""
PyTorch integration for Recursive Observer Framework.

This module provides PyTorch-specific implementations of mappings
and observers, with support for:
- Automatic tensor conversion
- MC Dropout uncertainty estimation
- GPU acceleration
- Gradient computation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ro_framework.core.dof import DoF
from ro_framework.core.state import State
from ro_framework.observer.mapping import NeuralMapping
from ro_framework.observer.observer import Observer

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type checking
    nn = Any  # type: ignore


@dataclass
class TorchNeuralMapping(NeuralMapping):
    """
    PyTorch implementation of NeuralMapping.

    Wraps PyTorch nn.Module models to work with DoF/State abstractions.
    Handles automatic tensor conversion and device management.

    Attributes:
        name: Identifier for this mapping
        input_dofs: DoFs that form the input space
        output_dofs: DoFs that form the output space
        model: PyTorch nn.Module
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        resolution: Per-DoF resolution limits

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 10))
        >>> mapping = TorchNeuralMapping(
        ...     name="encoder",
        ...     input_dofs=input_dofs,
        ...     output_dofs=output_dofs,
        ...     model=model,
        ...     device="cuda"
        ... )
    """

    device: str = "cpu"
    use_dropout_uncertainty: bool = False
    dropout_samples: int = 10

    def __post_init__(self) -> None:
        """Initialize PyTorch-specific components."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with: pip install ro-framework[torch]"
            )

        super().__post_init__()

        # Move model to device
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        # Set to eval mode by default
        if hasattr(self.model, "eval"):
            self.model.eval()

    def __call__(self, external_state: State) -> State:
        """
        Execute mapping through PyTorch model.

        Args:
            external_state: Input state with values on input_dofs

        Returns:
            Output state with values on output_dofs
        """
        # Convert state to tensor
        input_vector = external_state.to_vector(self.input_dofs)
        input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0).to(self.device)

        # Forward pass (no gradients needed for inference)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert back to numpy
        output_vector = output_tensor.squeeze(0).cpu().numpy()

        # Convert to state
        output_state = State.from_vector(output_vector, self.output_dofs)

        return output_state

    def compute_uncertainty(self, external_state: State) -> Dict[DoF, float]:
        """
        Estimate uncertainty using MC Dropout or ensemble methods.

        If the model has dropout layers and use_dropout_uncertainty=True,
        uses Monte Carlo Dropout to estimate epistemic uncertainty.

        Args:
            external_state: Input state

        Returns:
            Dictionary mapping output DoFs to uncertainty estimates
        """
        if not self.use_dropout_uncertainty:
            # Fall back to resolution-based uncertainty
            return super().compute_uncertainty(external_state)

        # Enable dropout for uncertainty estimation
        self.model.train()

        # Get input tensor
        input_vector = external_state.to_vector(self.input_dofs)
        input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0).to(self.device)

        # Multiple forward passes with dropout
        samples = []
        for _ in range(self.dropout_samples):
            with torch.no_grad():
                output = self.model(input_tensor)
                samples.append(output.squeeze(0).cpu().numpy())

        # Compute standard deviation as uncertainty
        samples_array = np.array(samples)  # Shape: (num_samples, output_dim)
        uncertainties = {}

        output_idx = 0
        for dof in self.output_dofs:
            # For now, assume each DoF corresponds to one output dimension
            # (This is simplified - actual implementation would handle categorical DoFs)
            std = np.std(samples_array[:, output_idx])
            uncertainties[dof] = float(std)
            output_idx += 1

        # Set back to eval mode
        self.model.eval()

        return uncertainties

    def forward_with_gradients(self, external_state: State) -> tuple[State, torch.Tensor]:
        """
        Execute forward pass with gradient tracking.

        Useful for training or computing saliency maps.

        Args:
            external_state: Input state

        Returns:
            Tuple of (output_state, output_tensor with gradients)
        """
        # Convert to tensor with gradients
        input_vector = external_state.to_vector(self.input_dofs)
        input_tensor = (
            torch.from_numpy(input_vector).float().unsqueeze(0).to(self.device).requires_grad_(True)
        )

        # Forward pass with gradients
        output_tensor = self.model(input_tensor)

        # Also create output state
        output_vector = output_tensor.squeeze(0).detach().cpu().numpy()
        output_state = State.from_vector(output_vector, self.output_dofs)

        return output_state, output_tensor


@dataclass
class TorchObserver(Observer):
    """
    Observer optimized for PyTorch models.

    Extends base Observer with PyTorch-specific functionality:
    - Automatic device management
    - Batch processing support
    - Gradient tracking for interpretability

    Example:
        >>> observer = TorchObserver(
        ...     name="torch_observer",
        ...     internal_dofs=internal_dofs,
        ...     external_dofs=external_dofs,
        ...     world_model=torch_world_model,
        ...     device="cuda"
        ... )
    """

    device: str = "cpu"

    def observe_batch(self, external_states: List[State]) -> List[State]:
        """
        Process a batch of states via a single batched forward pass.

        Falls back to sequential observation if the world_model is not
        a TorchNeuralMapping.

        Args:
            external_states: List of external states to observe

        Returns:
            List of internal states (each also logged in observation_log)
        """
        if not isinstance(self.world_model, TorchNeuralMapping) or not external_states:
            return [self.observe(s) for s in external_states]

        mapping: TorchNeuralMapping = self.world_model

        # Stack inputs into a single (N, D_in) tensor
        vectors = [s.to_vector(mapping.input_dofs) for s in external_states]
        batch_np = np.stack(vectors)
        batch_tensor = torch.from_numpy(batch_np).float().to(mapping.device)

        with torch.no_grad():
            out_tensor = mapping.model(batch_tensor)  # (N, D_out)

        out_np = out_tensor.cpu().numpy()

        # Convert each row back to a State and log the pair
        internal_states: List[State] = []
        for i, ext_state in enumerate(external_states):
            int_state = State.from_vector(out_np[i], mapping.output_dofs)
            self.internal_state = int_state
            from ro_framework.observer.observer import ObservationPair
            self.observation_log.append(ObservationPair(
                external_state=ext_state,
                internal_state=int_state,
                timestamp=float(len(self.observation_log)),
            ))
            internal_states.append(int_state)

        return internal_states

    def compute_saliency(self, external_state: State, target_dof: DoF) -> Dict[DoF, float]:
        """
        Compute saliency map: which input DoFs most affect target output DoF.

        Uses gradient-based attribution (|d output / d input|) to score
        each external DoF's influence on the target internal DoF.

        Args:
            external_state: Input state
            target_dof: Output DoF to compute saliency for

        Returns:
            Dictionary mapping input DoFs to importance scores (abs gradient)
        """
        if not isinstance(self.world_model, TorchNeuralMapping):
            raise ValueError("Saliency computation requires TorchNeuralMapping")

        mapping: TorchNeuralMapping = self.world_model

        # Build input tensor with gradient tracking
        input_vector = external_state.to_vector(mapping.input_dofs)
        input_tensor = (
            torch.from_numpy(input_vector)
            .float()
            .unsqueeze(0)
            .to(mapping.device)
            .requires_grad_(True)
        )

        # Forward pass
        output_tensor = mapping.model(input_tensor)

        # Get index of target DoF and backprop
        target_idx = mapping.output_dofs.index(target_dof)
        output_tensor[0, target_idx].backward()

        # Extract per-input-DoF gradient magnitudes
        grads = input_tensor.grad.squeeze(0).abs().cpu().numpy()

        saliency: Dict[DoF, float] = {}
        for i, dof in enumerate(self.external_dofs):
            saliency[dof] = float(grads[i]) if i < len(grads) else 0.0

        return saliency

    def __repr__(self) -> str:
        return (
            f"TorchObserver(name='{self.name}', "
            f"internal_dofs={len(self.internal_dofs)}, "
            f"external_dofs={len(self.external_dofs)}, "
            f"has_self_model={self.self_model is not None}, "
            f"observations={len(self.observation_log)}, "
            f"device='{self.device}')"
        )


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> nn.Module:
    """
    Create a multi-layer perceptron (MLP) for use as a mapping.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'gelu')
        dropout: Dropout probability (0.0 = no dropout)
        batch_norm: Whether to use batch normalization

    Returns:
        PyTorch nn.Module

    Example:
        >>> model = create_mlp(input_dim=10, output_dim=5, hidden_dims=[64, 32])
        >>> mapping = TorchNeuralMapping(
        ...     name="world_model",
        ...     input_dofs=input_dofs,
        ...     output_dofs=output_dofs,
        ...     model=model
        ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    layers = []
    prev_dim = input_dim

    # Activation functions
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    act_fn = activations.get(activation.lower(), nn.ReLU)

    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))

        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(act_fn())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        prev_dim = hidden_dim

    # Output layer (no activation)
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)
