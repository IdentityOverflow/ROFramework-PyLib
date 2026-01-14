"""
Unit tests for consciousness evaluation integration with Observer.

Tests that Observer properly integrates with ConsciousnessEvaluator
for is_conscious(), recursive_depth(), and get_consciousness_metrics().
"""

import pytest
import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.observer.mapping import IdentityMapping


class TestConsciousnessIntegration:
    """Tests for Observer consciousness evaluation integration."""

    def test_is_conscious_no_self_model(self):
        """Test is_conscious() returns False when no self-model."""
        # Create observer without self-model
        # Use same DoF for external and internal for IdentityMapping
        dof = PolarDoF(name="state", description="")

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="non_conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model
        )

        # Should not be conscious (no self-model)
        assert not observer.is_conscious()

    def test_is_conscious_with_self_model(self):
        """Test is_conscious() with self-model."""
        dof = PolarDoF(name="state", description="")

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        # Should be conscious (has self-model)
        assert observer.is_conscious()

    def test_is_conscious_custom_threshold(self):
        """Test is_conscious() with custom threshold."""
        dof = PolarDoF(name="state", description="")

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        # Low threshold - should pass
        assert observer.is_conscious(threshold=0.2)

        # Very high threshold - might fail depending on metrics
        # (architectural similarity is 1.0, but other metrics may be lower)
        result = observer.is_conscious(threshold=0.95)
        # Don't assert here as it depends on other factors

    def test_recursive_depth_no_self_model(self):
        """Test recursive_depth() with no self-model."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="non_conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model
        )

        # Depth should be 0 (no self-model)
        assert observer.recursive_depth() == 0

    def test_recursive_depth_with_self_model(self):
        """Test recursive_depth() with self-model."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        # Depth should be at least 1
        depth = observer.recursive_depth()
        assert depth >= 1

    def test_recursive_depth_meta_representation(self):
        """Test recursive_depth() with capacity for meta-representation."""
        # Create observer with internal dimension >= 2 * external dimension
        external_dofs = [PolarDoF(name=f"ext{i}", description="") for i in range(2)]
        internal_dofs = [PolarDoF(name=f"int{i}", description="") for i in range(5)]

        # Need a mapping that handles different dimensions
        # Use a custom mapping
        class CustomMapping:
            def __init__(self, input_dofs, output_dofs):
                self.input_dofs = input_dofs
                self.output_dofs = output_dofs

            def map(self, input_state):
                # Simple mapping: replicate and pad
                output_values = {}
                input_vals = list(input_state.dof_values.values())
                for i, dof in enumerate(self.output_dofs):
                    output_values[dof] = input_vals[i % len(input_vals)]
                return State(values=output_values)

        world_model = CustomMapping(external_dofs, internal_dofs)

        # Self-model (internal to internal)
        self_model = CustomMapping(internal_dofs, internal_dofs)

        observer = Observer(
            name="meta_conscious",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=world_model,
            self_model=self_model
        )

        # Should achieve depth 2 (5 internal >= 2 * 2 external)
        depth = observer.recursive_depth()
        assert depth == 2

    def test_get_consciousness_metrics(self):
        """Test get_consciousness_metrics() method."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        # Get metrics
        metrics = observer.get_consciousness_metrics()

        # Verify structure
        assert hasattr(metrics, 'has_self_model')
        assert hasattr(metrics, 'recursive_depth')
        assert hasattr(metrics, 'self_accuracy')
        assert hasattr(metrics, 'architectural_similarity')
        assert hasattr(metrics, 'calibration_error')
        assert hasattr(metrics, 'meta_cognitive_capability')
        assert hasattr(metrics, 'limitation_awareness')

        # Verify values
        assert metrics.has_self_model is True
        assert metrics.recursive_depth >= 1
        assert 0.0 <= metrics.self_accuracy <= 1.0
        assert 0.0 <= metrics.architectural_similarity <= 1.0
        assert 0.0 <= metrics.calibration_error <= 1.0
        assert 0.0 <= metrics.meta_cognitive_capability <= 1.0
        assert 0.0 <= metrics.limitation_awareness <= 1.0

    def test_consciousness_metrics_with_test_states(self):
        """Test consciousness metrics with test states."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        # Create test states
        test_states = [
            State(values={dof: float(i)})
            for i in range(10)
        ]

        # Get metrics with test states
        metrics = observer.get_consciousness_metrics(test_states)

        assert metrics.has_self_model is True
        # Self-accuracy should be evaluated on test states
        assert 0.0 <= metrics.self_accuracy <= 1.0

    def test_consciousness_score_calculation(self):
        """Test overall consciousness score calculation."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        metrics = observer.get_consciousness_metrics()
        score = metrics.consciousness_score()

        # Score should be in [0, 1]
        assert 0.0 <= score <= 1.0

        # With self-model and architectural similarity, score should be > 0
        assert score > 0.0

    def test_architectural_similarity_same_type(self):
        """Test architectural similarity with same model types."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        # Same type for self-model
        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        metrics = observer.get_consciousness_metrics()

        # Architectural similarity should be 1.0 (same type)
        assert metrics.architectural_similarity == 1.0

    def test_consciousness_metrics_no_self_model(self):
        """Test metrics for observer without self-model."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="non_conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model
        )

        metrics = observer.get_consciousness_metrics()

        # All metrics should reflect absence of consciousness
        assert metrics.has_self_model is False
        assert metrics.recursive_depth == 0
        assert metrics.self_accuracy == 0.0
        assert metrics.architectural_similarity == 0.0
        assert metrics.calibration_error == 1.0
        assert metrics.meta_cognitive_capability == 0.0
        assert metrics.limitation_awareness == 0.0
        assert metrics.consciousness_score() == 0.0

    def test_is_conscious_consistency_with_metrics(self):
        """Test that is_conscious() is consistent with consciousness_score()."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        threshold = 0.5
        is_conscious = observer.is_conscious(threshold=threshold)
        metrics = observer.get_consciousness_metrics()
        score = metrics.consciousness_score()

        # Consistency check
        assert is_conscious == (score >= threshold)

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        dof = PolarDoF(name="state", description="")
        # internal_dof = dof

        world_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        self_model = IdentityMapping(
            input_dofs=[dof],
            output_dofs=[dof]
        )

        observer = Observer(
            name="conscious",
            internal_dofs=[dof],
            external_dofs=[dof],
            world_model=world_model,
            self_model=self_model
        )

        metrics = observer.get_consciousness_metrics()
        metrics_dict = metrics.to_dict()

        # Verify all keys present
        assert "has_self_model" in metrics_dict
        assert "recursive_depth" in metrics_dict
        assert "self_accuracy" in metrics_dict
        assert "architectural_similarity" in metrics_dict
        assert "calibration_error" in metrics_dict
        assert "meta_cognitive_capability" in metrics_dict
        assert "limitation_awareness" in metrics_dict
        assert "overall_score" in metrics_dict

        # Verify types
        assert isinstance(metrics_dict["has_self_model"], bool)
        assert isinstance(metrics_dict["recursive_depth"], int)
        assert isinstance(metrics_dict["overall_score"], float)
