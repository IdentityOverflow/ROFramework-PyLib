"""
Degrees of Freedom (DoF) implementations.

A DoF represents a dimension of variation in the Block Universe.
Different types capture different structural properties.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

T = TypeVar("T")  # Value type for the DoF


class PolarDoFType(Enum):
    """Types of polar DoFs based on domain structure."""

    CONTINUOUS_REAL = "continuous_real"  # ℝ (unbounded)
    CONTINUOUS_BOUNDED = "continuous_bounded"  # [a, b] (bounded)
    DISCRETE_ORDERED = "discrete_ordered"  # Ordered discrete values


@dataclass(eq=False)
class DoF(ABC, Generic[T]):
    """
    Abstract base class for all Degrees of Freedom.

    A DoF represents a dimension of variation in the Block Universe.
    It defines a domain of possible values and operations on those values.

    Attributes:
        name: Unique identifier for this DoF
        description: Human-readable description of what this DoF represents
    """

    name: str
    description: str = ""

    @abstractmethod
    def domain(self) -> Any:
        """
        Returns the domain of possible values for this DoF.

        Returns:
            Domain specification (type depends on DoF type)
        """
        pass

    @abstractmethod
    def validate_value(self, value: T) -> bool:
        """
        Checks if a value is valid for this DoF.

        Args:
            value: Value to validate

        Returns:
            True if value is in domain, False otherwise
        """
        pass

    @abstractmethod
    def measure(self) -> str:
        """
        Returns the natural measure structure for this DoF.

        Returns:
            Measure type name (e.g., "lebesgue", "counting")
        """
        pass

    @abstractmethod
    def distance(self, v1: T, v2: T) -> float:
        """
        Computes distance between two values on this DoF.

        Args:
            v1: First value
            v2: Second value

        Returns:
            Non-negative distance
        """
        pass

    def __hash__(self) -> int:
        """Make DoFs hashable by name for use in dicts/sets."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Two DoFs are equal if they have the same name."""
        if not isinstance(other, DoF):
            return False
        return self.name == other.name


@dataclass(eq=False)
class PolarDoF(DoF[float]):
    """
    Polar Degree of Freedom: bidirectional with gradient support.

    Essential properties:
    - Bidirectionality (two opposing poles)
    - Gradation (continuous or discrete gradients)
    - Ordering (values are comparable)
    - Traversability (relations between positions defined)
    - Measurement (quantitative distinction)

    Examples: position, velocity, temperature, charge

    Attributes:
        pole_negative: Lower pole (or -∞ for unbounded)
        pole_positive: Upper pole (or +∞ for unbounded)
        polar_type: Type of polar DoF (continuous/discrete, bounded/unbounded)
        resolution: Minimum distinguishable difference
    """

    pole_negative: float = -np.inf
    pole_positive: float = np.inf
    polar_type: PolarDoFType = PolarDoFType.CONTINUOUS_REAL
    resolution: float = 1e-6

    def __post_init__(self) -> None:
        """Validate poles."""
        if self.pole_negative >= self.pole_positive:
            raise ValueError(
                f"pole_negative ({self.pole_negative}) must be < pole_positive "
                f"({self.pole_positive})"
            )

    def domain(self) -> Tuple[float, float]:
        """
        Returns (lower_bound, upper_bound) or (-inf, inf) for unbounded.

        Returns:
            Tuple of (lower, upper) bounds
        """
        if self.polar_type == PolarDoFType.CONTINUOUS_REAL:
            return (-np.inf, np.inf)
        return (self.pole_negative, self.pole_positive)

    def validate_value(self, value: float) -> bool:
        """
        Check if value is within domain.

        Args:
            value: Value to validate

        Returns:
            True if value is in [pole_negative, pole_positive]
        """
        lower, upper = self.domain()
        return lower <= value <= upper

    def measure(self) -> str:
        """Returns measure type (Lebesgue for continuous)."""
        return "lebesgue"

    def distance(self, v1: float, v2: float) -> float:
        """
        Euclidean distance along the DoF.

        Args:
            v1: First value
            v2: Second value

        Returns:
            Absolute distance |v1 - v2|
        """
        return abs(v1 - v2)

    def normalize(self, value: float) -> float:
        """
        Normalize value to [-1, 1] range based on poles.

        Useful for neural network inputs.

        Args:
            value: Value to normalize

        Returns:
            Normalized value in [-1, 1]
        """
        if self.polar_type == PolarDoFType.CONTINUOUS_REAL:
            # Use tanh-like normalization for unbounded
            return float(np.tanh(value))
        else:
            # Linear normalization for bounded
            return (
                2 * (value - self.pole_negative) / (self.pole_positive - self.pole_negative) - 1
            )

    def denormalize(self, normalized_value: float) -> float:
        """
        Convert normalized value back to original scale.

        Args:
            normalized_value: Value in [-1, 1]

        Returns:
            Value in original domain
        """
        if self.polar_type == PolarDoFType.CONTINUOUS_REAL:
            # Inverse of tanh
            return float(np.arctanh(np.clip(normalized_value, -0.99999, 0.99999)))
        else:
            # Inverse of linear normalization
            return (normalized_value + 1) / 2 * (self.pole_positive - self.pole_negative) + self.pole_negative

    def gradient(self, v1: float, v2: float) -> float:
        """
        Compute gradient (directional difference) from v1 to v2.

        Positive means toward positive pole, negative toward negative pole.

        Args:
            v1: Starting value
            v2: Ending value

        Returns:
            Signed difference (v2 - v1)
        """
        return v2 - v1


@dataclass(eq=False)
class ScalarDoF(DoF[float]):
    """
    Scalar Degree of Freedom: magnitude-only, no inherent direction.

    Examples: mass, distance, speed, probability, energy density

    Attributes:
        min_value: Minimum value (default 0)
        max_value: Maximum value (default ∞)
        resolution: Minimum distinguishable difference
    """

    min_value: float = 0.0
    max_value: float = np.inf
    resolution: float = 1e-6

    def __post_init__(self) -> None:
        """Validate bounds."""
        if self.min_value >= self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be < max_value ({self.max_value})"
            )

    def domain(self) -> Tuple[float, float]:
        """Returns (min_value, max_value)."""
        return (self.min_value, self.max_value)

    def validate_value(self, value: float) -> bool:
        """Check if value is in [min_value, max_value]."""
        return self.min_value <= value <= self.max_value

    def measure(self) -> str:
        """Returns measure type (Lebesgue for continuous)."""
        return "lebesgue"

    def distance(self, v1: float, v2: float) -> float:
        """
        Distance is absolute difference (non-directional).

        Args:
            v1: First value
            v2: Second value

        Returns:
            Absolute distance |v1 - v2|
        """
        return abs(v1 - v2)

    def normalize(self, value: float) -> float:
        """
        Normalize value to [0, 1] range.

        Args:
            value: Value to normalize

        Returns:
            Normalized value in [0, 1]
        """
        if self.max_value == np.inf:
            # Use sigmoid-like normalization for unbounded
            return float(1.0 / (1.0 + np.exp(-value)))
        else:
            # Linear normalization for bounded
            return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, normalized_value: float) -> float:
        """
        Convert normalized value back to original scale.

        Args:
            normalized_value: Value in [0, 1]

        Returns:
            Value in original domain
        """
        if self.max_value == np.inf:
            # Inverse of sigmoid
            return float(-np.log(1.0 / normalized_value - 1.0))
        else:
            # Inverse of linear normalization
            return normalized_value * (self.max_value - self.min_value) + self.min_value


@dataclass(eq=False)
class CategoricalDoF(DoF[str]):
    """
    Categorical Degree of Freedom: discrete, unordered values.

    Examples: particle type, color names, object labels, modality type

    Attributes:
        categories: Set of valid category values
        weights: Optional weights for each category (for weighted measure)
    """

    categories: Set[str] = field(default_factory=set)
    weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Initialize uniform weights if not provided."""
        if not self.categories:
            raise ValueError("CategoricalDoF must have at least one category")

        if self.weights is None:
            # Uniform weights by default
            self.weights = {cat: 1.0 / len(self.categories) for cat in self.categories}
        else:
            # Validate weights
            if set(self.weights.keys()) != self.categories:
                raise ValueError("Weights keys must match categories")
            if not np.isclose(sum(self.weights.values()), 1.0):
                raise ValueError("Weights must sum to 1.0")

    def domain(self) -> Set[str]:
        """Returns the set of valid categories."""
        return self.categories

    def validate_value(self, value: str) -> bool:
        """Check if value is in categories."""
        return value in self.categories

    def measure(self) -> str:
        """Returns measure type."""
        if self.weights is None:
            return "counting"

        # Check if all weights are equal
        weights_list = list(self.weights.values())
        if all(np.isclose(w, weights_list[0]) for w in weights_list):
            return "counting"
        else:
            return "weighted_counting"

    def distance(self, v1: str, v2: str) -> float:
        """
        Binary distance: 0 if same, 1 if different.

        Args:
            v1: First category
            v2: Second category

        Returns:
            0.0 if v1 == v2, else 1.0
        """
        return 0.0 if v1 == v2 else 1.0

    def to_one_hot(self, value: str) -> np.ndarray:
        """
        Convert category to one-hot encoding.

        Args:
            value: Category value

        Returns:
            One-hot encoded vector

        Raises:
            ValueError: If value not in categories
        """
        if not self.validate_value(value):
            raise ValueError(f"Value '{value}' not in categories")

        categories_list = sorted(self.categories)  # Deterministic ordering
        one_hot = np.zeros(len(categories_list), dtype=np.float32)
        idx = categories_list.index(value)
        one_hot[idx] = 1.0
        return one_hot

    def from_one_hot(self, one_hot: np.ndarray) -> str:
        """
        Convert one-hot encoding back to category.

        Args:
            one_hot: One-hot encoded vector

        Returns:
            Category value

        Raises:
            ValueError: If one_hot is invalid
        """
        categories_list = sorted(self.categories)
        if len(one_hot) != len(categories_list):
            raise ValueError(
                f"One-hot vector length ({len(one_hot)}) doesn't match "
                f"number of categories ({len(categories_list)})"
            )

        idx = int(np.argmax(one_hot))
        return categories_list[idx]


@dataclass(eq=False)
class DerivedDoF(DoF[float]):
    """
    Derived Degree of Freedom: computed from other DoFs.

    Examples: velocity (from position + time), force (from mass + acceleration)

    Attributes:
        constituent_dofs: List of DoFs this is derived from
        derivation_function: Function that computes derived value
        result_type: Type hint for result (for documentation)
    """

    constituent_dofs: List[DoF] = field(default_factory=list)
    derivation_function: Optional[Callable[..., float]] = None
    result_type: type = float

    def __post_init__(self) -> None:
        """Validate derivation function."""
        if not self.constituent_dofs:
            raise ValueError("DerivedDoF must have at least one constituent DoF")

    def domain(self) -> str:
        """Domain depends on constituent DoFs and derivation."""
        return "computed"

    def validate_value(self, value: float) -> bool:
        """Basic validation for numeric values."""
        return isinstance(value, (int, float)) and not np.isnan(value)

    def measure(self) -> str:
        """Returns measure type."""
        return "derived"

    def distance(self, v1: float, v2: float) -> float:
        """Absolute distance between derived values."""
        return abs(v1 - v2)

    def compute(self, **kwargs: Any) -> float:
        """
        Compute derived value from constituent DoF values.

        Args:
            **kwargs: Named values for each constituent DoF

        Returns:
            Computed derived value

        Raises:
            ValueError: If derivation_function is None or kwargs are missing
        """
        if self.derivation_function is None:
            raise ValueError("No derivation function defined")

        # Validate that all constituent DoFs have values
        constituent_names = {dof.name for dof in self.constituent_dofs}
        provided_names = set(kwargs.keys())

        if not constituent_names.issubset(provided_names):
            missing = constituent_names - provided_names
            raise ValueError(f"Missing values for constituent DoFs: {missing}")

        return self.derivation_function(**kwargs)
