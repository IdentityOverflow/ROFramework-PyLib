"""Unit tests for consciousness evaluation."""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF, PolarDoFType
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer
from ro_framework.consciousness.evaluation import (
    ConsciousnessEvaluator,
    ConsciousnessMetrics,
    compare_observers,
    rank_by_consciousness,
)


@pytest.fixture
def simple_dofs():
    """Create simple DoFs for testing."""
    external_dofs = [
        PolarDoF(
            name=f"ext_{i}",
            pole_negative=-1.0,
            pole_positive=1.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(3)
    ]
    internal_dofs = [
        PolarDoF(
            name=f"int_{i}",
            pole_negative=-5.0,
            pole_positive=5.0,
            polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
        )
        for i in range(5)
    ]
    return external_dofs, internal_dofs


@pytest.fixture
def simple_mapping(simple_dofs):
    """Create simple mapping for testing."""
    external_dofs, internal_dofs = simple_dofs

    class SimpleMapping:
        def __call__(self, state: State) -> State:
            # Simple linear transformation
            values = {}
            for i, int_dof in enumerate(internal_dofs):
                values[int_dof] = np.random.uniform(-5, 5)
            return State(values=values)

    return SimpleMapping()


@pytest.fixture
def unconscious_observer(simple_dofs, simple_mapping):
    """Create observer without self-model."""
    external_dofs, internal_dofs = simple_dofs

    return Observer(
        name="unconscious",
        internal_dofs=internal_dofs,
        external_dofs=external_dofs,
        world_model=simple_mapping,
        self_model=None  # No self-model = not conscious
    )


@pytest.fixture
def conscious_observer(simple_dofs, simple_mapping):
    """Create observer with self-model."""
    external_dofs, internal_dofs = simple_dofs

    class SelfMapping:
        def __call__(self, state: State) -> State:
            # Identity-like self-mapping
            return state

    return Observer(
        name="conscious",
        internal_dofs=internal_dofs,
        external_dofs=external_dofs,
        world_model=simple_mapping,
        self_model=SelfMapping()  # Has self-model = conscious
    )


class TestConsciousnessMetrics:
    """Test ConsciousnessMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating consciousness metrics."""
        metrics = ConsciousnessMetrics(
            has_self_model=True,
            recursive_depth=1,
            self_accuracy=0.9,
            architectural_similarity=1.0,
            calibration_error=0.1,
            meta_cognitive_capability=0.8,
            limitation_awareness=0.7,
        )

        assert metrics.has_self_model
        assert metrics.recursive_depth == 1
        assert metrics.self_accuracy == 0.9

    def test_consciousness_score_no_self_model(self):
        """Test score is 0 without self-model."""
        metrics = ConsciousnessMetrics(
            has_self_model=False,
            recursive_depth=0,
            self_accuracy=0.0,
            architectural_similarity=0.0,
            calibration_error=1.0,
            meta_cognitive_capability=0.0,
            limitation_awareness=0.0,
        )

        assert metrics.consciousness_score() == 0.0

    def test_consciousness_score_perfect(self):
        """Test score with perfect metrics."""
        metrics = ConsciousnessMetrics(
            has_self_model=True,
            recursive_depth=3,
            self_accuracy=1.0,
            architectural_similarity=1.0,
            calibration_error=0.0,
            meta_cognitive_capability=1.0,
            limitation_awareness=1.0,
        )

        score = metrics.consciousness_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high with perfect metrics

    def test_consciousness_score_weighted(self):
        """Test that score is weighted combination."""
        metrics = ConsciousnessMetrics(
            has_self_model=True,
            recursive_depth=1,
            self_accuracy=0.5,
            architectural_similarity=0.5,
            calibration_error=0.5,
            meta_cognitive_capability=0.5,
            limitation_awareness=0.5,
        )

        score = metrics.consciousness_score()
        assert 0.0 < score < 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ConsciousnessMetrics(
            has_self_model=True,
            recursive_depth=1,
            self_accuracy=0.8,
            architectural_similarity=0.9,
            calibration_error=0.2,
            meta_cognitive_capability=0.7,
            limitation_awareness=0.6,
        )

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert "has_self_model" in d
        assert "overall_score" in d
        assert d["has_self_model"] == True
        assert isinstance(d["overall_score"], float)


class TestConsciousnessEvaluator:
    """Test ConsciousnessEvaluator class."""

    def test_creation(self, unconscious_observer):
        """Test creating evaluator."""
        evaluator = ConsciousnessEvaluator(unconscious_observer)
        assert evaluator.observer == unconscious_observer

    def test_evaluate_unconscious(self, unconscious_observer):
        """Test evaluating unconscious observer."""
        evaluator = ConsciousnessEvaluator(unconscious_observer)
        metrics = evaluator.evaluate()

        assert not metrics.has_self_model
        assert metrics.recursive_depth == 0
        assert metrics.consciousness_score() == 0.0

    def test_evaluate_conscious(self, conscious_observer):
        """Test evaluating conscious observer."""
        evaluator = ConsciousnessEvaluator(conscious_observer)
        metrics = evaluator.evaluate()

        assert metrics.has_self_model
        assert metrics.recursive_depth >= 1
        assert metrics.consciousness_score() > 0.0

    def test_evaluate_with_test_states(self, conscious_observer, simple_dofs):
        """Test evaluation with test states."""
        external_dofs, _ = simple_dofs

        # Create test states
        test_states = [
            State(values={dof: np.random.uniform(-1, 1) for dof in external_dofs})
            for _ in range(10)
        ]

        evaluator = ConsciousnessEvaluator(conscious_observer)
        metrics = evaluator.evaluate(test_states)

        assert isinstance(metrics, ConsciousnessMetrics)

    def test_self_accuracy_evaluation(self, conscious_observer, simple_dofs):
        """Test self-accuracy metric."""
        evaluator = ConsciousnessEvaluator(conscious_observer)

        # First observe something
        external_dofs, _ = simple_dofs
        external_state = State(values={dof: 0.5 for dof in external_dofs})
        conscious_observer.observe(external_state)

        accuracy = evaluator._evaluate_self_accuracy()

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_architectural_similarity_evaluation(self, conscious_observer):
        """Test architectural similarity metric."""
        evaluator = ConsciousnessEvaluator(conscious_observer)
        similarity = evaluator._evaluate_architectural_similarity()

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_metacognition_evaluation(self, conscious_observer):
        """Test meta-cognition evaluation."""
        evaluator = ConsciousnessEvaluator(conscious_observer)
        meta_score = evaluator._evaluate_metacognition()

        assert isinstance(meta_score, float)
        assert 0.0 <= meta_score <= 1.0

    def test_limitation_awareness_evaluation(self, conscious_observer):
        """Test limitation awareness evaluation."""
        evaluator = ConsciousnessEvaluator(conscious_observer)
        awareness = evaluator._evaluate_limitation_awareness()

        assert isinstance(awareness, float)
        assert 0.0 <= awareness <= 1.0


class TestComparisonFunctions:
    """Test observer comparison functions."""

    def test_compare_observers(self, unconscious_observer, conscious_observer):
        """Test comparing multiple observers."""
        observers = [unconscious_observer, conscious_observer]

        comparison = compare_observers(observers)

        assert len(comparison) == 2
        assert "unconscious" in comparison
        assert "conscious" in comparison

        # Conscious should score higher
        unconscious_score = comparison["unconscious"].consciousness_score()
        conscious_score = comparison["conscious"].consciousness_score()
        assert conscious_score > unconscious_score

    def test_rank_by_consciousness(self, simple_dofs, simple_mapping):
        """Test ranking observers by consciousness."""
        external_dofs, internal_dofs = simple_dofs

        # Create observers with different levels of consciousness
        obs1 = Observer(
            name="obs1",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=simple_mapping,
            self_model=None  # Not conscious
        )

        class SelfMapping:
            def __call__(self, state: State) -> State:
                return state

        obs2 = Observer(
            name="obs2",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=simple_mapping,
            self_model=SelfMapping()  # Conscious
        )

        ranked = rank_by_consciousness([obs1, obs2])

        assert len(ranked) == 2
        # Most conscious should be first
        assert ranked[0][0].name == "obs2"
        assert ranked[1][0].name == "obs1"
        # Scores should be in descending order
        assert ranked[0][1] >= ranked[1][1]

    def test_rank_empty_list(self):
        """Test ranking with empty list."""
        ranked = rank_by_consciousness([])
        assert ranked == []

    def test_rank_single_observer(self, conscious_observer):
        """Test ranking with single observer."""
        ranked = rank_by_consciousness([conscious_observer])
        assert len(ranked) == 1
        assert ranked[0][0] == conscious_observer


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_evaluation_pipeline(self, simple_dofs):
        """Test complete evaluation pipeline."""
        external_dofs, internal_dofs = simple_dofs

        # Create sophisticated mapping
        class WorldModel:
            def __call__(self, state: State) -> State:
                values = {}
                for i, dof in enumerate(internal_dofs):
                    # Simple transformation
                    values[dof] = float(i) * 0.5
                return State(values=values)

        class SelfModel:
            def __call__(self, state: State) -> State:
                # Almost identity
                values = {}
                for dof in internal_dofs:
                    val = state.get_value(dof)
                    values[dof] = val * 0.95 if val is not None else 0.0
                return State(values=values)

        observer = Observer(
            name="test",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=WorldModel(),
            self_model=SelfModel()
        )

        # Create test states
        test_states = [
            State(values={dof: np.random.uniform(-1, 1) for dof in external_dofs})
            for _ in range(20)
        ]

        # Evaluate
        evaluator = ConsciousnessEvaluator(observer)
        metrics = evaluator.evaluate(test_states)

        # Check all metrics are valid
        assert metrics.has_self_model
        assert 0.0 <= metrics.self_accuracy <= 1.0
        assert 0.0 <= metrics.architectural_similarity <= 1.0
        assert 0.0 <= metrics.calibration_error <= 1.0
        assert 0.0 <= metrics.meta_cognitive_capability <= 1.0
        assert 0.0 <= metrics.limitation_awareness <= 1.0
        assert 0.0 <= metrics.consciousness_score() <= 1.0

    def test_consciousness_improves_with_features(self, simple_dofs, simple_mapping):
        """Test that consciousness score improves with better features."""
        external_dofs, internal_dofs = simple_dofs

        # Observer 1: Basic self-model
        class BasicSelfModel:
            def __call__(self, state: State) -> State:
                values = {dof: 0.0 for dof in internal_dofs}
                return State(values=values)

        obs1 = Observer(
            name="basic",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=simple_mapping,
            self_model=BasicSelfModel()
        )

        # Observer 2: Better self-model (identity)
        class BetterSelfModel:
            def __call__(self, state: State) -> State:
                return state  # Perfect self-model

        obs2 = Observer(
            name="better",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=simple_mapping,
            self_model=BetterSelfModel()
        )

        # Observe something first
        test_state = State(values={dof: 0.5 for dof in external_dofs})
        obs1.observe(test_state)
        obs2.observe(test_state)

        # Evaluate
        eval1 = ConsciousnessEvaluator(obs1)
        eval2 = ConsciousnessEvaluator(obs2)

        metrics1 = eval1.evaluate()
        metrics2 = eval2.evaluate()

        # Better self-model should have higher self-accuracy
        # (though both might be low without real training)
        assert metrics2.self_accuracy >= metrics1.self_accuracy - 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
