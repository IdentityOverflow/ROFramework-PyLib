"""
Seed Architecture — A Self-Organizing Observer Module

Implements the Seed: a self-organizing neural architecture expressed in
RO Framework terms. Five local rules applied by oscillatory nodes produce
criticality, frequency bands, cross-scale coupling, memory, and consciousness
as emergent consequences.

See docs/seed_architecture.md for the full specification.
"""

from ro_framework.seed.node import OscillatoryNode, SeedConfig
from ro_framework.seed.network import SeedNetwork, SensorInterface, ActuatorInterface
from ro_framework.seed.criticality import (
    extract_cascades,
    fast_mi,
    measure_branching_ratio,
    measure_scale_distribution,
    verify_power_law,
)

__all__ = [
    "OscillatoryNode",
    "SeedConfig",
    "SeedNetwork",
    "SensorInterface",
    "ActuatorInterface",
    "extract_cascades",
    "fast_mi",
    "measure_branching_ratio",
    "measure_scale_distribution",
    "verify_power_law",
]
