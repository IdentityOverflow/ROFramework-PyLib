"""
Mapping functions for observers.

Mappings are structural relations between external and internal DoFs.
They represent the core mechanism of observation: translating external
configurations to internal representations.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

from ro_framework.core.dof import DoF
from ro_framework.core.state import State


@runtime_checkable
class MappingFunction(Protocol):
    """
    A mapping is a structural relation between external and internal DoFs.

    M: domain(d_ext) → domain(d_int)

    Properties:
    - Non-invertible (information compression)
    - Finite precision (limited by resolution)
    - Context-dependent (may vary with other internal state)
    - Possibly stochastic (distribution over internal configurations)
    """

    @abstractmethod
    def __call__(self, external_state: State) -> State:
        """
        Map external DoF configuration to internal DoF configuration.

        Args:
            external_state: Input state with values on external DoFs

        Returns:
            Output state with values on internal DoFs
        """
        ...


@dataclass
class NeuralMapping:
    """
    Neural network implementation of mapping function.

    This is the practical implementation for AI systems. It wraps
    a neural network model and handles state ↔ vector conversion.

    Attributes:
        name: Identifier for this mapping
        input_dofs: DoFs that form the input space
        output_dofs: DoFs that form the output space
        model: Neural network (framework-agnostic, can be PyTorch, JAX, etc.)
        resolution: Per-DoF resolution limits
    """

    name: str
    input_dofs: List[DoF]
    output_dofs: List[DoF]
    model: Any  # Neural network (PyTorch, JAX, etc.)
    resolution: Dict[DoF, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize resolution dict if empty."""
        if not self.resolution:
            self.resolution = {dof: 1e-3 for dof in self.output_dofs}

    def __call__(self, external_state: State) -> State:
        """
        Execute mapping through neural network.

        This is a framework-agnostic implementation. Subclasses for specific
        frameworks (PyTorch, JAX) should override this method.

        Args:
            external_state: Input state with values on input_dofs

        Returns:
            Output state with values on output_dofs

        Raises:
            NotImplementedError: If model is not callable
        """
        if not callable(self.model):
            raise NotImplementedError(
                "Base NeuralMapping requires a callable model. "
                "Use framework-specific subclasses (e.g., TorchNeuralMapping) "
                "for automatic handling."
            )

        # Convert state to vector
        input_vector = external_state.to_vector(self.input_dofs)

        # Forward pass through neural network
        output_vector = self.model(input_vector)

        # Convert back to state
        output_state = State.from_vector(output_vector, self.output_dofs)

        return output_state

    def compute_uncertainty(self, external_state: State) -> Dict[DoF, float]:
        """
        Estimate uncertainty in mapping for each output DoF.

        This can use ensemble methods, Bayesian neural networks,
        or dropout-based uncertainty estimation.

        Base implementation returns resolution-based uncertainty.
        Override in subclasses for more sophisticated methods.

        Args:
            external_state: Input state

        Returns:
            Dictionary mapping output DoFs to uncertainty estimates
        """
        # Base implementation: return resolution limits
        return self.resolution.copy()


@dataclass
class IdentityMapping:
    """
    Identity mapping: passes input directly to output.

    Useful for testing and as a baseline.
    """

    name: str = "identity"
    input_dofs: List[DoF] = field(default_factory=list)
    output_dofs: List[DoF] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that input and output DoFs match."""
        if self.input_dofs != self.output_dofs:
            raise ValueError("IdentityMapping requires input_dofs == output_dofs")

    def __call__(self, external_state: State) -> State:
        """Return input state unchanged."""
        return external_state


@dataclass
class ComposedMapping:
    """
    Composition of multiple mappings.

    Useful for building complex pipelines: f(g(h(x)))
    """

    name: str
    mappings: List[MappingFunction]

    def __post_init__(self) -> None:
        """Validate that mappings can be composed."""
        if len(self.mappings) < 2:
            raise ValueError("ComposedMapping requires at least 2 mappings")

    def __call__(self, external_state: State) -> State:
        """
        Apply mappings in sequence.

        Args:
            external_state: Initial input state

        Returns:
            Final output state after all mappings
        """
        current_state = external_state
        for mapping in self.mappings:
            current_state = mapping(current_state)
        return current_state
