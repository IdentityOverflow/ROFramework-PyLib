"""
Value implementation.

A value is a specific position on a single DoF.
"""

from dataclasses import dataclass
from typing import Any

from ro_framework.core.dof import DoF


@dataclass(frozen=True)
class Value:
    """
    A value is a specific position on a single DoF.

    Values are immutable and always associated with a DoF.
    They represent a specific location along one dimension of the Block Universe.

    Attributes:
        dof: The Degree of Freedom this value belongs to
        value: The actual value (type depends on DoF type)

    Example:
        >>> from ro_framework.core.dof import PolarDoF
        >>> position_x = PolarDoF(name="x", pole_negative=-10, pole_positive=10)
        >>> v = Value(dof=position_x, value=3.5)
        >>> print(v)
        Value(x=3.5)
    """

    dof: DoF
    value: Any

    def __post_init__(self) -> None:
        """Validate value upon creation."""
        if not self.dof.validate_value(self.value):
            raise ValueError(
                f"Invalid value {self.value} for DoF '{self.dof.name}'. "
                f"Valid domain: {self.dof.domain()}"
            )

    def __repr__(self) -> str:
        """String representation showing DoF name and value."""
        return f"Value({self.dof.name}={self.value})"

    def distance_to(self, other: "Value") -> float:
        """
        Compute distance to another value on the same DoF.

        Args:
            other: Another Value to compute distance to

        Returns:
            Distance between the two values

        Raises:
            ValueError: If values are on different DoFs

        Example:
            >>> v1 = Value(dof=position_x, value=3.0)
            >>> v2 = Value(dof=position_x, value=7.0)
            >>> v1.distance_to(v2)
            4.0
        """
        if self.dof != other.dof:
            raise ValueError(
                f"Cannot compute distance between values on different DoFs: "
                f"'{self.dof.name}' and '{other.dof.name}'"
            )
        return self.dof.distance(self.value, other.value)
