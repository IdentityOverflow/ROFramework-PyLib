"""
Observer module for Recursive Observer Framework.

This module provides observer architecture including:
- Observer: Core observer class with boundary, mapping, resolution, memory
- Mapping functions: Structural relations between external and internal DoFs
- Boundary: Partition of DoFs into internal/external
"""

from ro_framework.observer.mapping import MappingFunction, NeuralMapping
from ro_framework.observer.observer import Observer

__all__ = [
    "Observer",
    "MappingFunction",
    "NeuralMapping",
]
