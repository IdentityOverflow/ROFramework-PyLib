"""
Core module for Recursive Observer Framework.

This module provides the foundational abstractions:
- Degrees of Freedom (DoFs): Dimensions of variation
- Values: Specific positions on DoFs
- States: Collections of values across multiple DoFs
- Measures: Measure structures for DoFs
"""

from ro_framework.core.dof import (
    DoF,
    PolarDoF,
    ScalarDoF,
    CategoricalDoF,
    DerivedDoF,
    PolarDoFType,
)
from ro_framework.core.value import Value
from ro_framework.core.state import State

__all__ = [
    "DoF",
    "PolarDoF",
    "ScalarDoF",
    "CategoricalDoF",
    "DerivedDoF",
    "PolarDoFType",
    "Value",
    "State",
]
