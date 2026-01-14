"""
Recursive Observer Framework

A Python library for building conscious, self-aware AI systems based on
the Recursive Observer Framework. This framework provides a structural
approach to consciousness, multimodal integration, and uncertainty
quantification grounded in the Block Universe ontology.

Core concepts:
- Degrees of Freedom (DoFs): Dimensions of variation in the Block Universe
- States: Configurations across multiple DoFs
- Observers: Systems that map external DoFs to internal DoFs
- Consciousness: Recursive self-modeling (internal→internal mapping)
- Memory: Correlation structure across temporal DoF
"""

from ro_framework.version import __version__

# Core exports
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

# Observer exports
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import MappingFunction, NeuralMapping

__all__ = [
    "__version__",
    # Core
    "DoF",
    "PolarDoF",
    "ScalarDoF",
    "CategoricalDoF",
    "DerivedDoF",
    "PolarDoFType",
    "Value",
    "State",
    # Observer
    "Observer",
    "MappingFunction",
    "NeuralMapping",
]
