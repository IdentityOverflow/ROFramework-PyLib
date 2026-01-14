"""
Correlation module for detecting structural relationships.

This module provides tools for measuring correlations between DoFs,
detecting causality, and determining knowledge structures.
"""

from ro_framework.correlation.measures import (
    CorrelationMeasure,
    pearson_correlation,
    mutual_information,
    temporal_correlation,
)

__all__ = [
    "CorrelationMeasure",
    "pearson_correlation",
    "mutual_information",
    "temporal_correlation",
]
