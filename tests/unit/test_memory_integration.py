"""
Unit tests for integrated memory system with correlation module.

Tests the proper integration of Observer memory (via ObservationLog)
with correlation measures.
"""

import pytest
import numpy as np

from ro_framework.core.dof import PolarDoF, ScalarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import IdentityMapping


class TestMemoryIntegration:
    """Tests for memory integration with correlation module."""

    def test_has_memory_with_temporal_correlation(self):
        """Test has_memory() using temporal correlation."""
        temporal_dof = ScalarDoF(name="time", min_value=0, max_value=100)
        internal_dof = PolarDoF(name="latent", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[internal_dof],
            world_model=IdentityMapping(
                input_dofs=[internal_dof],
                output_dofs=[internal_dof],
            ),
            temporal_dof=temporal_dof,
        )

        # Build memory through observations with autocorrelated pattern
        for i in range(20):
            value = np.sin(i * 0.3)
            observer.observe(State(values={internal_dof: value}))

        assert observer.has_memory(threshold=0.5)

    def test_has_memory_without_correlation(self):
        """Test has_memory() with random uncorrelated data."""
        temporal_dof = ScalarDoF(name="time", min_value=0, max_value=100)
        internal_dof = PolarDoF(name="latent", pole_negative=-10, pole_positive=10)

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[internal_dof],
            world_model=IdentityMapping(
                input_dofs=[internal_dof],
                output_dofs=[internal_dof],
            ),
            temporal_dof=temporal_dof,
        )

        np.random.seed(42)
        for _ in range(20):
            value = np.random.randn() * 5
            observer.observe(State(values={internal_dof: value}))

        assert not observer.has_memory(threshold=0.8)

    def test_has_memory_no_temporal_dof(self):
        """Test has_memory() returns False without temporal DoF."""
        internal_dof = PolarDoF(name="latent", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[internal_dof],
            world_model=IdentityMapping(
                input_dofs=[internal_dof],
                output_dofs=[internal_dof],
            ),
            temporal_dof=None,
        )

        for i in range(10):
            observer.observe(State(values={internal_dof: float(i) / 10}))

        assert not observer.has_memory()

    def test_has_memory_insufficient_data(self):
        """Test has_memory() with insufficient data."""
        temporal_dof = ScalarDoF(name="time", min_value=0, max_value=100)
        internal_dof = PolarDoF(name="latent", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[internal_dof],
            world_model=IdentityMapping(
                input_dofs=[internal_dof],
                output_dofs=[internal_dof],
            ),
            temporal_dof=temporal_dof,
        )

        # Only 2 observations (not enough)
        observer.observe(State(values={internal_dof: 0.1}))
        observer.observe(State(values={internal_dof: 0.2}))

        assert not observer.has_memory()

    def test_analyze_memory_structure(self):
        """Test analyze_memory_structure() method."""
        temporal_dof = ScalarDoF(name="time", min_value=0, max_value=100)
        dof1 = PolarDoF(name="latent1", pole_negative=-1, pole_positive=1)
        dof2 = PolarDoF(name="latent2", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[dof1, dof2],
            external_dofs=[dof1, dof2],
            world_model=IdentityMapping(
                input_dofs=[dof1, dof2],
                output_dofs=[dof1, dof2],
            ),
            temporal_dof=temporal_dof,
        )

        for i in range(15):
            val1 = np.sin(i * 0.3)
            val2 = np.cos(i * 0.3)
            observer.observe(State(values={dof1: val1, dof2: val2}))

        analysis = observer.analyze_memory_structure(max_lag=5)

        assert dof1 in analysis
        assert dof2 in analysis
        assert len(analysis[dof1]) > 0
        assert len(analysis[dof2]) > 0

    def test_analyze_memory_structure_no_temporal_dof(self):
        """Test analyze_memory_structure() without temporal DoF."""
        internal_dof = PolarDoF(name="latent", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[internal_dof],
            external_dofs=[internal_dof],
            world_model=IdentityMapping(
                input_dofs=[internal_dof],
                output_dofs=[internal_dof],
            ),
            temporal_dof=None,
        )

        analysis = observer.analyze_memory_structure()
        assert analysis == {}

    def test_get_memory_correlations(self):
        """Test get_memory_correlations() method."""
        temporal_dof = ScalarDoF(name="time", min_value=0, max_value=100)
        dof1 = PolarDoF(name="latent1", pole_negative=-5, pole_positive=5)
        dof2 = PolarDoF(name="latent2", pole_negative=-5, pole_positive=5)

        observer = Observer(
            name="test_observer",
            internal_dofs=[dof1, dof2],
            external_dofs=[dof1, dof2],
            world_model=IdentityMapping(
                input_dofs=[dof1, dof2],
                output_dofs=[dof1, dof2],
            ),
            temporal_dof=temporal_dof,
        )

        for i in range(20):
            val1 = i * 0.25
            val2 = i * 0.25 + 1
            observer.observe(State(values={dof1: val1, dof2: val2}))

        corr = observer.get_memory_correlations(dof1, dof2)
        assert corr > 0.9

    def test_get_memory_correlations_insufficient_data(self):
        """Test get_memory_correlations() with insufficient data."""
        dof1 = PolarDoF(name="latent1", pole_negative=-1, pole_positive=1)
        dof2 = PolarDoF(name="latent2", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[dof1, dof2],
            external_dofs=[dof1, dof2],
            world_model=IdentityMapping(
                input_dofs=[dof1, dof2],
                output_dofs=[dof1, dof2],
            ),
        )

        # Only 1 observation
        observer.observe(State(values={dof1: 0.1, dof2: 0.2}))

        corr = observer.get_memory_correlations(dof1, dof2)
        assert corr == 0.0

    def test_memory_integration_with_observation(self):
        """Test memory builds up correctly through observations."""
        temporal_dof = ScalarDoF(name="time", min_value=0, max_value=100)
        sensor_dof = PolarDoF(name="sensor", pole_negative=-1, pole_positive=1)

        observer = Observer(
            name="test_observer",
            internal_dofs=[sensor_dof],
            external_dofs=[sensor_dof],
            world_model=IdentityMapping(
                input_dofs=[sensor_dof],
                output_dofs=[sensor_dof],
            ),
            temporal_dof=temporal_dof,
        )

        for i in range(15):
            observer.observe(State(values={sensor_dof: np.sin(i * 0.2)}))

        assert observer.get_memory_size() == 15
        assert observer.has_memory(threshold=0.5)

        analysis = observer.analyze_memory_structure()
        assert len(analysis) > 0
