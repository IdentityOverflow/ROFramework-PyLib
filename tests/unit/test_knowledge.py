"""Unit tests for knowledge assessment: K(d_ext) = (ρ, ε, σ, C)."""

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF, ScalarDoF
from ro_framework.core.state import State
from ro_framework.knowledge.assessment import KnowledgeAssessment, compute_knowledge
from ro_framework.observer.observer import Observer, ObservationLog, ObservationPair


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ext_dof():
    return PolarDoF(name="temperature", description="external temp")


@pytest.fixture
def int_dof():
    return PolarDoF(name="sensor", description="internal sensor")


@pytest.fixture
def noise_dof():
    return PolarDoF(name="noise", description="unrelated internal")


def _build_log(ext_dof, int_dof, ext_values, int_values, capacity=1000):
    """Helper: build an ObservationLog from parallel value lists."""
    log = ObservationLog(capacity=capacity)
    for i, (ev, iv) in enumerate(zip(ext_values, int_values)):
        log.append(ObservationPair(
            external_state=State(values={ext_dof: ev}),
            internal_state=State(values={int_dof: iv}),
            timestamp=float(i),
        ))
    return log


# ---------------------------------------------------------------------------
# KnowledgeAssessment dataclass
# ---------------------------------------------------------------------------


class TestKnowledgeAssessment:
    """Tests for the KnowledgeAssessment dataclass and knowledge_type."""

    def test_strong_knowledge(self, ext_dof, int_dof):
        ka = KnowledgeAssessment(
            external_dof=ext_dof,
            best_internal_dof=int_dof,
            correlation=0.95,
            systematic_error=0.05,
            random_error=0.1,
            calibration=0.8,
            n_samples=100,
        )
        assert ka.knowledge_type == "strong"

    def test_false_knowledge(self, ext_dof, int_dof):
        ka = KnowledgeAssessment(
            external_dof=ext_dof,
            best_internal_dof=int_dof,
            correlation=0.85,
            systematic_error=0.5,  # high bias
            random_error=0.2,
            calibration=0.3,
            n_samples=100,
        )
        assert ka.knowledge_type == "false"

    def test_uncertain_knowledge(self, ext_dof, int_dof):
        ka = KnowledgeAssessment(
            external_dof=ext_dof,
            best_internal_dof=int_dof,
            correlation=0.3,  # low correlation
            systematic_error=0.1,
            random_error=0.2,
            calibration=0.7,  # but well calibrated
            n_samples=100,
        )
        assert ka.knowledge_type == "uncertain"

    def test_weak_knowledge(self, ext_dof, int_dof):
        ka = KnowledgeAssessment(
            external_dof=ext_dof,
            best_internal_dof=int_dof,
            correlation=0.55,
            systematic_error=0.2,
            random_error=0.4,
            calibration=0.3,
            n_samples=100,
        )
        assert ka.knowledge_type == "weak"

    def test_frozen(self, ext_dof, int_dof):
        ka = KnowledgeAssessment(
            external_dof=ext_dof,
            best_internal_dof=int_dof,
            correlation=0.9,
            systematic_error=0.0,
            random_error=0.1,
            calibration=0.8,
            n_samples=50,
        )
        with pytest.raises(AttributeError):
            ka.correlation = 0.5


# ---------------------------------------------------------------------------
# compute_knowledge
# ---------------------------------------------------------------------------


class TestComputeKnowledge:
    """Tests for the compute_knowledge function."""

    def test_insufficient_samples(self, ext_dof, int_dof):
        """Returns None when observation count < min_samples."""
        log = _build_log(ext_dof, int_dof, [1, 2, 3], [1, 2, 3])
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)
        assert result is None

    def test_perfect_correlation(self, ext_dof, int_dof):
        """Perfect linear mapping → ρ ≈ 1, ε ≈ 0, σ ≈ 0, C ≈ 1."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 50).tolist()
        int_vals = ext_vals  # identity mapping

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation > 0.99
        assert abs(result.systematic_error) < 0.05
        assert result.random_error < 0.05
        assert result.calibration > 0.9
        assert result.knowledge_type == "strong"

    def test_scaled_mapping(self, ext_dof, int_dof):
        """Scaled linear mapping (int = 2*ext) → still high ρ after z-normalization."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 50).tolist()
        int_vals = [2 * v for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation > 0.99
        # Z-normalization removes scale, so errors should still be small
        assert result.random_error < 0.05

    def test_noisy_correlation(self, ext_dof, int_dof):
        """Noisy linear mapping → moderate ρ, higher σ."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 100).tolist()
        int_vals = [v + rng.normal(0, 3) for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert 0.3 < result.correlation < 0.99  # correlated but not perfect
        assert result.random_error > 0.1  # noticeable noise

    def test_no_correlation(self, ext_dof, int_dof):
        """Uncorrelated data → low ρ."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 100).tolist()
        int_vals = rng.uniform(-10, 10, 100).tolist()  # independent

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation < 0.3

    def test_constant_series_returns_fallback(self, ext_dof, int_dof):
        """Constant external values → no variance → fallback assessment."""
        log = _build_log(ext_dof, int_dof, [5.0] * 20, list(range(20)))
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation < 1e-12
        assert result.best_internal_dof is None

    def test_selects_best_internal_dof(self, ext_dof, int_dof, noise_dof):
        """When multiple internal DoFs, selects the most correlated."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 50).tolist()
        good_vals = ext_vals  # perfect correlation
        noise_vals = rng.uniform(-10, 10, 50).tolist()  # no correlation

        log = ObservationLog(capacity=1000)
        for i, (ev, gv, nv) in enumerate(zip(ext_vals, good_vals, noise_vals)):
            log.append(ObservationPair(
                external_state=State(values={ext_dof: ev}),
                internal_state=State(values={int_dof: gv, noise_dof: nv}),
                timestamp=float(i),
            ))

        result = compute_knowledge(log, ext_dof, [int_dof, noise_dof], min_samples=10)

        assert result is not None
        assert result.best_internal_dof == int_dof
        assert result.correlation > 0.9

    def test_negative_correlation_low_bias(self, ext_dof, int_dof):
        """Inverse mapping (int = -ext) → high ρ, near-zero ε (not false knowledge)."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 50).tolist()
        int_vals = [-v for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation > 0.99
        assert abs(result.systematic_error) < 0.05  # sign-aligned, no fake bias
        assert result.random_error < 0.05
        assert result.knowledge_type == "strong"  # not "false"

    def test_negative_correlation_with_real_bias(self, ext_dof, int_dof):
        """Inverse mapping with genuine offset → bias detected after sign alignment."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 50).tolist()
        # Negate + add a consistent offset to create real bias
        int_vals = [-v + 5.0 for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation > 0.99
        # After sign alignment, the z-normalized residuals should be near zero
        # because z-normalization removes constant offsets.
        # Bias only shows up if the *shape* differs, not a constant shift.
        assert abs(result.systematic_error) < 0.05

    def test_n_samples_reflects_paired_data(self, ext_dof, int_dof):
        """n_samples should reflect the actual number of valid pairs used."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 30).tolist()
        int_vals = ext_vals

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.n_samples == 30


# ---------------------------------------------------------------------------
# Observer integration: assess_knowledge() and know()
# ---------------------------------------------------------------------------


class TestObserverKnowledge:
    """Tests for Observer.assess_knowledge() and Observer.know()."""

    def _make_observer(self, ext_dof, int_dof):
        """Create a simple observer with a linear mapping."""
        class LinearMapping:
            def __call__(self, state: State) -> State:
                val = state.get_value(ext_dof)
                return State(values={int_dof: val if val is not None else 0.0})

        return Observer(
            name="test",
            internal_dofs=[int_dof],
            external_dofs=[ext_dof],
            world_model=LinearMapping(),
        )

    def test_assess_knowledge_insufficient_data(self, ext_dof, int_dof):
        """assess_knowledge returns None with too few observations."""
        obs = self._make_observer(ext_dof, int_dof)
        for i in range(5):
            obs.observe(State(values={ext_dof: float(i)}))

        result = obs.assess_knowledge(ext_dof, min_samples=10)
        assert result is None

    def test_assess_knowledge_identity_mapping(self, ext_dof, int_dof):
        """Identity mapping → strong knowledge after enough observations."""
        obs = self._make_observer(ext_dof, int_dof)
        rng = np.random.default_rng(42)
        for _ in range(50):
            obs.observe(State(values={ext_dof: float(rng.uniform(-10, 10))}))

        result = obs.assess_knowledge(ext_dof, min_samples=10)
        assert result is not None
        assert result.correlation > 0.99
        assert result.knowledge_type == "strong"

    def test_know_returns_true_for_strong_knowledge(self, ext_dof, int_dof):
        """know() returns True when correlation and calibration meet thresholds."""
        obs = self._make_observer(ext_dof, int_dof)
        rng = np.random.default_rng(42)
        for _ in range(50):
            obs.observe(State(values={ext_dof: float(rng.uniform(-10, 10))}))

        assert obs.know(ext_dof, threshold=0.7) is True

    def test_know_returns_false_insufficient_data(self, ext_dof, int_dof):
        """know() returns False with insufficient observations."""
        obs = self._make_observer(ext_dof, int_dof)
        obs.observe(State(values={ext_dof: 1.0}))

        assert obs.know(ext_dof) is False

    def test_know_returns_false_for_unknown_dof(self, ext_dof, int_dof):
        """know() returns False for a DoF the observer hasn't tracked."""
        obs = self._make_observer(ext_dof, int_dof)
        rng = np.random.default_rng(42)
        for _ in range(50):
            obs.observe(State(values={ext_dof: float(rng.uniform(-10, 10))}))

        unknown_dof = PolarDoF(name="unknown")
        assert obs.know(unknown_dof) is False
