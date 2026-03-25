"""
Recursive Observer Framework

A Python library for building observers with graded knowledge assessment
and structural consciousness evaluation. Wraps any model (neural network,
function, or callable) as an Observer that maps external DoFs to internal
DoFs with finite resolution, paired observation history, and optional
recursive self-modeling.

Core concepts:
- Degrees of Freedom (DoFs): Typed dimensions of variation
- States: Configurations across multiple DoFs
- Observers: O = (B, M, R, Mem) — Boundary, Mapping, Resolution, Memory
- Knowledge: K(d_ext) = (ρ, ε, σ, C) — graded, observer-relative
- Consciousness: Recursive self-modeling with bounded error
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
from ro_framework.observer.observer import Observer, ObservationPair, ObservationLog
from ro_framework.observer.mapping import MappingFunction, NeuralMapping

# Knowledge exports
from ro_framework.knowledge.assessment import KnowledgeAssessment, compute_knowledge
from ro_framework.knowledge.tracker import KnowledgeTracker, TrajectoryPoint

# Consciousness exports
from ro_framework.consciousness.evaluation import (
    ConsciousnessEvaluator,
    ConsciousnessMetrics,
)

# Seed architecture exports
from ro_framework.seed.node import OscillatoryNode, SeedConfig
from ro_framework.seed.network import SeedNetwork
from ro_framework.seed.criticality import verify_power_law, measure_branching_ratio

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
    "ObservationPair",
    "ObservationLog",
    "MappingFunction",
    "NeuralMapping",
    # Knowledge
    "KnowledgeAssessment",
    "compute_knowledge",
    "KnowledgeTracker",
    "TrajectoryPoint",
    # Consciousness
    "ConsciousnessEvaluator",
    "ConsciousnessMetrics",
    # Seed
    "OscillatoryNode",
    "SeedConfig",
    "SeedNetwork",
    "verify_power_law",
    "measure_branching_ratio",
]
