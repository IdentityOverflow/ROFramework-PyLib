"""
State implementation.

A state is a collection of values across multiple DoFs, representing
a specific location in multi-dimensional DoF-space.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ro_framework.core.dof import CategoricalDoF, DoF, PolarDoF, ScalarDoF


class State:
    """
    A state is a collection of values across multiple DoFs.

    States are locations in multi-dimensional DoF-space. They are relational
    (defined by DoF-value pairs), not substantial. States are immutable by
    convention (modifications return new State objects).

    Attributes:
        values: Mapping from DoF to value for that DoF

    Example:
        >>> position_x = PolarDoF(name="x", pole_negative=-10, pole_positive=10)
        >>> position_y = PolarDoF(name="y", pole_negative=-10, pole_positive=10)
        >>> state = State(values={position_x: 3.0, position_y: 4.0})
        >>> state.get_value(position_x)
        3.0
    """

    def __init__(self, values: Dict[DoF, Any]):
        """
        Create a new state.

        Args:
            values: Dictionary mapping DoFs to their values

        Raises:
            ValueError: If any value is invalid for its DoF
        """
        # Validate all values
        for dof, value in values.items():
            if not dof.validate_value(value):
                raise ValueError(
                    f"Invalid value {value} for DoF '{dof.name}'. "
                    f"Valid domain: {dof.domain()}"
                )

        self._values = values.copy()  # Defensive copy

    @property
    def values(self) -> Dict[DoF, Any]:
        """Get read-only view of values."""
        return self._values.copy()

    def get_value(self, dof: DoF) -> Optional[Any]:
        """
        Get value for a specific DoF.

        Args:
            dof: DoF to get value for

        Returns:
            Value for the DoF, or None if DoF not in state

        Example:
            >>> state.get_value(position_x)
            3.0
        """
        return self._values.get(dof)

    def set_value(self, dof: DoF, value: Any) -> "State":
        """
        Return new state with updated value (states are immutable).

        Args:
            dof: DoF to update
            value: New value for the DoF

        Returns:
            New State with updated value

        Example:
            >>> new_state = state.set_value(position_x, 5.0)
            >>> new_state.get_value(position_x)
            5.0
            >>> state.get_value(position_x)  # Original unchanged
            3.0
        """
        new_values = self._values.copy()
        new_values[dof] = value
        return State(values=new_values)

    def project(self, dofs: List[DoF]) -> "State":
        """
        Project state onto subset of DoFs.

        Args:
            dofs: List of DoFs to project onto

        Returns:
            New State containing only the specified DoFs

        Example:
            >>> projected = state.project([position_x])
            >>> projected.get_value(position_x)
            3.0
            >>> projected.get_value(position_y)
            None
        """
        return State(values={dof: self._values[dof] for dof in dofs if dof in self._values})

    def distance_to(self, other: "State", dofs: Optional[List[DoF]] = None) -> float:
        """
        Compute Euclidean distance to another state.

        Args:
            other: Target state
            dofs: Optional subset of DoFs to consider (default: all common DoFs)

        Returns:
            Euclidean distance in DoF-space

        Example:
            >>> state1 = State(values={position_x: 0.0, position_y: 0.0})
            >>> state2 = State(values={position_x: 3.0, position_y: 4.0})
            >>> state1.distance_to(state2)
            5.0  # 3-4-5 triangle
        """
        if dofs is None:
            # Use all common DoFs
            dofs = list(set(self._values.keys()) & set(other._values.keys()))

        if not dofs:
            return 0.0

        distances = []
        for dof in dofs:
            if dof in self._values and dof in other._values:
                d = dof.distance(self._values[dof], other._values[dof])
                distances.append(d)

        return float(np.sqrt(sum(d**2 for d in distances)))

    def to_vector(self, dof_order: List[DoF]) -> np.ndarray:
        """
        Convert state to vector representation for neural networks.

        Handles different DoF types:
        - PolarDoF: Normalized to [-1, 1]
        - ScalarDoF: Normalized to [0, 1]
        - CategoricalDoF: One-hot encoded
        - DerivedDoF: Treated as scalar

        Args:
            dof_order: Ordered list of DoFs defining vector structure

        Returns:
            NumPy array with normalized/encoded values

        Example:
            >>> vector = state.to_vector([position_x, position_y])
            >>> vector.shape
            (2,)
        """
        vector_components = []

        for dof in dof_order:
            value = self._values.get(dof)

            if value is None:
                # Missing value - use zero or special token
                if isinstance(dof, CategoricalDoF):
                    # Zero vector for missing categorical
                    vector_components.extend([0.0] * len(dof.categories))
                else:
                    vector_components.append(0.0)

            elif isinstance(dof, PolarDoF):
                # Normalize polar DoFs to [-1, 1]
                vector_components.append(dof.normalize(value))

            elif isinstance(dof, ScalarDoF):
                # Normalize scalar DoFs to [0, 1]
                vector_components.append(dof.normalize(value))

            elif isinstance(dof, CategoricalDoF):
                # One-hot encoding
                one_hot = dof.to_one_hot(value)
                vector_components.extend(one_hot.tolist())

            else:
                # Default: treat as scalar
                vector_components.append(float(value))

        return np.array(vector_components, dtype=np.float32)

    @classmethod
    def from_vector(
        cls, vector: np.ndarray, dof_order: List[DoF], missing_dofs: Optional[List[DoF]] = None
    ) -> "State":
        """
        Reconstruct state from vector representation.

        This is the inverse of to_vector().

        Args:
            vector: NumPy array of normalized/encoded values
            dof_order: Ordered list of DoFs that were used to create vector
            missing_dofs: Optional list of DoFs that were marked as missing

        Returns:
            Reconstructed State

        Raises:
            ValueError: If vector size doesn't match expected size from dof_order

        Example:
            >>> vector = np.array([0.5, -0.5])
            >>> state = State.from_vector(vector, [position_x, position_y])
        """
        missing_dofs = missing_dofs or []
        values = {}
        idx = 0

        for dof in dof_order:
            if dof in missing_dofs:
                # Skip missing DoFs
                if isinstance(dof, CategoricalDoF):
                    idx += len(dof.categories)
                else:
                    idx += 1
                continue

            if isinstance(dof, PolarDoF):
                # Denormalize from [-1, 1]
                values[dof] = dof.denormalize(vector[idx])
                idx += 1

            elif isinstance(dof, ScalarDoF):
                # Denormalize from [0, 1]
                values[dof] = dof.denormalize(vector[idx])
                idx += 1

            elif isinstance(dof, CategoricalDoF):
                # Decode one-hot
                num_categories = len(dof.categories)
                one_hot = vector[idx : idx + num_categories]
                values[dof] = dof.from_one_hot(one_hot)
                idx += num_categories

            else:
                # Default: treat as scalar
                values[dof] = float(vector[idx])
                idx += 1

        if idx != len(vector):
            raise ValueError(
                f"Vector size mismatch: expected to use {idx} elements "
                f"but vector has {len(vector)} elements"
            )

        return cls(values=values)

    def __repr__(self) -> str:
        """String representation of state."""
        dof_strs = [f"{dof.name}={value}" for dof, value in self._values.items()]
        return f"State({', '.join(dof_strs)})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on DoF values."""
        if not isinstance(other, State):
            return False
        return self._values == other._values
