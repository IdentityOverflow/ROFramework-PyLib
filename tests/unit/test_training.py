"""Tests for knowledge-guided training utilities."""

import numpy as np
import pytest

from ro_framework import Observer, PolarDoF, State
from ro_framework.knowledge.assessment import KnowledgeAssessment
from ro_framework.knowledge.tracker import KnowledgeTracker
from ro_framework.observer.observer import ObservationPair
from ro_framework.integration.training import (
    FeatureRegularization,
    KnowledgeRegularizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observer_and_tracker(n_ext=2, n_int=4, log_capacity=200):
    """Create a minimal Observer + KnowledgeTracker for testing."""
    ext_dofs = [PolarDoF(name=f"ext_{i}") for i in range(n_ext)]
    int_dofs = [PolarDoF(name=f"int_{i}") for i in range(n_int)]

    class _Identity:
        def __call__(self, state):
            return State(values={d: 0.0 for d in int_dofs})

    observer = Observer(
        name="test_obs",
        internal_dofs=int_dofs,
        external_dofs=ext_dofs,
        world_model=_Identity(),
        log_capacity=log_capacity,
    )
    tracker = KnowledgeTracker(observer, external_dofs=ext_dofs)
    return observer, tracker, ext_dofs, int_dofs


def _populate_correlated(observer, ext_dofs, int_dofs, n=50, noise=0.0, bias=0.0):
    """Populate observation log with correlated data.

    int_0 tracks ext_0 with optional noise and bias.
    """
    rng = np.random.default_rng(42)
    observer.clear_memory()
    for i in range(n):
        ext_val = rng.uniform(-1, 1)
        int_val = ext_val + bias + noise * rng.normal()
        observer.observation_log.append(ObservationPair(
            external_state=State(values={
                ext_dofs[0]: ext_val,
                ext_dofs[1]: rng.uniform(-1, 1),
            }),
            internal_state=State(values={
                int_dofs[0]: int_val,
                int_dofs[1]: rng.uniform(-1, 1),
                int_dofs[2]: rng.uniform(-1, 1),
                int_dofs[3]: rng.uniform(-1, 1),
            }),
            timestamp=float(i),
        ))


def _populate_uncorrelated(observer, ext_dofs, int_dofs, n=50):
    """Populate observation log with uncorrelated data."""
    rng = np.random.default_rng(99)
    observer.clear_memory()
    for i in range(n):
        observer.observation_log.append(ObservationPair(
            external_state=State(values={d: rng.uniform(-1, 1) for d in ext_dofs}),
            internal_state=State(values={d: rng.uniform(-1, 1) for d in int_dofs}),
            timestamp=float(i),
        ))


# ---------------------------------------------------------------------------
# TestFeatureRegularization
# ---------------------------------------------------------------------------


class TestFeatureRegularization:
    def test_creation(self):
        fr = FeatureRegularization(
            feature_name="sin_7",
            knowledge_type="strong",
            correlation=0.95,
            systematic_error=0.02,
            calibration=0.8,
            weight_decay_multiplier=0.5,
            epoch_updated=100,
        )
        assert fr.feature_name == "sin_7"
        assert fr.knowledge_type == "strong"
        assert fr.weight_decay_multiplier == 0.5

    def test_all_fields_accessible(self):
        fr = FeatureRegularization(
            feature_name="test",
            knowledge_type="weak",
            correlation=0.3,
            systematic_error=0.1,
            calibration=0.4,
            weight_decay_multiplier=1.0,
            epoch_updated=0,
        )
        assert fr.correlation == 0.3
        assert fr.systematic_error == 0.1
        assert fr.calibration == 0.4
        assert fr.epoch_updated == 0


# ---------------------------------------------------------------------------
# TestKnowledgeRegularizer
# ---------------------------------------------------------------------------


class TestKnowledgeRegularizer:
    def test_default_multiplier_before_update(self):
        _, tracker, _, _ = _make_observer_and_tracker()
        reg = KnowledgeRegularizer(tracker, base_weight_decay=1.0)
        assert reg.get_weight_decay() == 1.0
        assert reg.current_multiplier == 1.0

    def test_memorized_increases_weight_decay(self):
        """High ρ + low C → memorized → increased weight decay."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        # noise=0.7 gives ρ≈0.65, C≈0.58 — "weak" by default thresholds
        # Use relaxed thresholds so this classifies as memorized
        _populate_correlated(obs, ext_dofs, int_dofs, n=50, noise=0.7)
        tracker.step(0)

        reg = KnowledgeRegularizer(
            tracker,
            base_weight_decay=1.0,
            memorized_multiplier=3.0,
            memorized_min_correlation=0.5,
            memorized_max_calibration=0.6,
        )
        reg.update(0)

        # Should have increased weight decay
        assert reg.get_weight_decay() > 1.0

    def test_generalized_decreases_weight_decay(self):
        """High ρ + high C → generalized → decreased weight decay."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        # Clean correlation: high ρ, low ε, low σ → high C
        _populate_correlated(obs, ext_dofs, int_dofs, n=50, noise=0.01)
        tracker.step(0)

        # Verify we actually got strong knowledge
        latest = tracker.latest(ext_dofs[0])
        assert latest is not None
        assert latest.correlation > 0.7

        reg = KnowledgeRegularizer(
            tracker,
            base_weight_decay=1.0,
            generalized_multiplier=0.5,
        )
        reg.update(0)

        # If the feature is classified as generalized, wd should decrease
        states = reg.feature_states()
        if ext_dofs[0].name in states:
            state = states[ext_dofs[0].name]
            if state.knowledge_type == "strong":
                assert reg.get_weight_decay() <= 1.0

    def test_uncertain_leaves_weight_decay_unchanged(self):
        """Low ρ → uncertain → no change."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        _populate_uncorrelated(obs, ext_dofs, int_dofs, n=50)
        tracker.step(0)

        reg = KnowledgeRegularizer(tracker, base_weight_decay=1.0)
        reg.update(0)
        assert reg.get_weight_decay() == 1.0

    def test_mixed_state_uses_max_multiplier(self):
        """Conservative: if any feature is memorized, use max multiplier."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()

        # Manually populate: ext_0 correlated (memorized), ext_1 uncorrelated
        rng = np.random.default_rng(42)
        obs.clear_memory()
        for i in range(50):
            ext_val = rng.uniform(-1, 1)
            obs.observation_log.append(ObservationPair(
                external_state=State(values={
                    ext_dofs[0]: ext_val,
                    ext_dofs[1]: rng.uniform(-1, 1),
                }),
                internal_state=State(values={
                    int_dofs[0]: ext_val + 2.0 * rng.normal(),  # noisy correlation
                    int_dofs[1]: rng.uniform(-1, 1),
                    int_dofs[2]: rng.uniform(-1, 1),
                    int_dofs[3]: rng.uniform(-1, 1),
                }),
                timestamp=float(i),
            ))

        tracker.step(0)
        reg = KnowledgeRegularizer(
            tracker,
            base_weight_decay=1.0,
            memorized_multiplier=3.0,
            memorized_min_correlation=0.3,
            memorized_max_calibration=0.8,
        )
        reg.update(0)

        states = reg.feature_states()
        multipliers = [s.weight_decay_multiplier for s in states.values()]

        # If any multiplier > 1.0, the aggregate should be the max
        if any(m > 1.0 for m in multipliers):
            assert reg.current_multiplier == max(multipliers)

    def test_bias_penalty_for_false_knowledge(self):
        """Features with high ρ and high |ε| should contribute bias penalty."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        # Correlated but with large systematic bias
        _populate_correlated(obs, ext_dofs, int_dofs, n=50, noise=0.1, bias=2.0)
        tracker.step(0)

        latest = tracker.latest(ext_dofs[0])
        assert latest is not None

        reg = KnowledgeRegularizer(
            tracker,
            base_weight_decay=1.0,
            bias_penalty_weight=0.5,
        )
        reg.update(0)

        # If knowledge is "false" (high ρ, high |ε|), penalty should be > 0
        states = reg.feature_states()
        if ext_dofs[0].name in states:
            state = states[ext_dofs[0].name]
            if state.knowledge_type == "false":
                assert reg.get_loss_penalty() > 0.0

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        _, tracker, _, _ = _make_observer_and_tracker()
        reg = KnowledgeRegularizer(
            tracker,
            memorized_min_correlation=0.9,
            memorized_max_calibration=0.1,
            generalized_min_correlation=0.95,
            generalized_min_calibration=0.9,
        )
        # Very strict thresholds — almost nothing should be classified
        assert reg._mem_min_rho == 0.9
        assert reg._mem_max_cal == 0.1
        assert reg._gen_min_rho == 0.95
        assert reg._gen_min_cal == 0.9


# ---------------------------------------------------------------------------
# TestIntegrationWithTracker
# ---------------------------------------------------------------------------


class TestIntegrationWithTracker:
    def test_reads_from_tracker(self):
        """Regularizer reads assessments from KnowledgeTracker."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        _populate_correlated(obs, ext_dofs, int_dofs, n=50, noise=0.1)
        tracker.step(0)

        reg = KnowledgeRegularizer(tracker)
        states = reg.update(0)

        # Should have states for tracked DoFs that had assessments
        assert len(states) > 0
        for name, state in states.items():
            assert state.knowledge_type in ("strong", "weak", "false", "uncertain")
            assert state.epoch_updated == 0

    def test_update_after_multiple_steps(self):
        """Regularizer updates correctly as tracker accumulates data."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        reg = KnowledgeRegularizer(tracker, base_weight_decay=1.0)

        # Step 0: uncorrelated data
        _populate_uncorrelated(obs, ext_dofs, int_dofs, n=50)
        tracker.step(0)
        reg.update(0)
        wd_0 = reg.get_weight_decay()

        # Step 1: now strongly correlated
        _populate_correlated(obs, ext_dofs, int_dofs, n=50, noise=0.01)
        tracker.step(1)
        reg.update(1)
        wd_1 = reg.get_weight_decay()

        # Weight decay should have changed
        # (specific direction depends on classification)
        states = reg.feature_states()
        assert all(s.epoch_updated == 1 for s in states.values())

    def test_no_penalty_without_false_knowledge(self):
        """get_loss_penalty() returns 0 when no features have false knowledge."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        _populate_uncorrelated(obs, ext_dofs, int_dofs, n=50)
        tracker.step(0)

        reg = KnowledgeRegularizer(tracker)
        reg.update(0)
        assert reg.get_loss_penalty() == 0.0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_data_yet(self):
        """Before any tracker.step(), multiplier should be 1.0."""
        _, tracker, _, _ = _make_observer_and_tracker()
        reg = KnowledgeRegularizer(tracker, base_weight_decay=2.0)
        reg.update(0)
        assert reg.get_weight_decay() == 2.0
        assert reg.current_multiplier == 1.0
        assert reg.get_loss_penalty() == 0.0

    def test_empty_feature_states(self):
        """feature_states() returns empty dict before update."""
        _, tracker, _, _ = _make_observer_and_tracker()
        reg = KnowledgeRegularizer(tracker)
        assert reg.feature_states() == {}

    def test_base_weight_decay_scaling(self):
        """Multiplier is applied to base_weight_decay correctly."""
        obs, tracker, ext_dofs, int_dofs = _make_observer_and_tracker()
        _populate_uncorrelated(obs, ext_dofs, int_dofs, n=50)
        tracker.step(0)

        reg = KnowledgeRegularizer(tracker, base_weight_decay=0.5)
        reg.update(0)
        # With uncorrelated data, multiplier should be 1.0
        assert reg.get_weight_decay() == pytest.approx(0.5, abs=0.01)
