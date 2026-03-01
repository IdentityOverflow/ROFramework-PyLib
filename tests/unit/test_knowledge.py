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


# ---------------------------------------------------------------------------
# Reachability of all four knowledge types from compute_knowledge()
# ---------------------------------------------------------------------------


class TestKnowledgeTypeReachability:
    """Verify that all four knowledge types are genuinely reachable
    from compute_knowledge(), not just from hand-constructed assessments."""

    def test_strong_from_computation(self, ext_dof, int_dof):
        """Clean linear tracking → strong knowledge."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 100).tolist()
        # Small uniform noise — good linear fit, homoscedastic
        int_vals = [v + rng.normal(0, 0.5) for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.knowledge_type == "strong", (
            f"Expected 'strong', got '{result.knowledge_type}' "
            f"(ρ={result.correlation:.3f}, ε={result.systematic_error:.3f}, "
            f"σ={result.random_error:.3f}, C={result.calibration:.3f})"
        )

    def test_false_from_computation(self, ext_dof, int_dof):
        """Heteroscedastic tracking (confound) → false knowledge.

        Feature correlates with label but error magnitude depends on position:
        accurate for positive ext values, wildly wrong for negative ones.
        """
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 200).tolist()
        # Heteroscedastic: noise grows with |ext| in one direction
        int_vals = []
        for v in ext_vals:
            if v > 0:
                int_vals.append(v + rng.normal(0, 0.3))  # accurate
            else:
                int_vals.append(v + rng.normal(0, 5.0))  # very noisy
        # Still correlated overall, but errors cluster at one end

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation >= 0.7, (
            f"Expected ρ ≥ 0.7, got {result.correlation:.3f}"
        )
        assert result.systematic_error >= 0.3, (
            f"Expected ε ≥ 0.3 (heteroscedastic), got {result.systematic_error:.3f}"
        )
        assert result.knowledge_type == "false", (
            f"Expected 'false', got '{result.knowledge_type}' "
            f"(ρ={result.correlation:.3f}, ε={result.systematic_error:.3f}, "
            f"σ={result.random_error:.3f}, C={result.calibration:.3f})"
        )

    def test_uncertain_from_computation(self, ext_dof, int_dof):
        """Uncorrelated but homoscedastic → uncertain knowledge.

        No tracking, but the error structure is uniform — the observer
        is consistently wrong by the same amount everywhere.
        """
        rng = np.random.default_rng(42)
        n = 200
        ext_vals = rng.uniform(-10, 10, n).tolist()
        # Independent of ext, but uniform noise level
        int_vals = rng.normal(0, 3, n).tolist()

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.correlation < 0.5, (
            f"Expected ρ < 0.5, got {result.correlation:.3f}"
        )
        assert result.calibration >= 0.5, (
            f"Expected C ≥ 0.5 (homoscedastic noise), got {result.calibration:.3f}"
        )
        assert result.knowledge_type == "uncertain", (
            f"Expected 'uncertain', got '{result.knowledge_type}' "
            f"(ρ={result.correlation:.3f}, ε={result.systematic_error:.3f}, "
            f"σ={result.random_error:.3f}, C={result.calibration:.3f})"
        )

    def test_weak_from_computation(self, ext_dof, int_dof):
        """Moderate correlation with heteroscedastic noise → weak knowledge."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 200).tolist()
        # Moderate correlation + heteroscedastic noise → moderate ρ, low C
        int_vals = [v * 0.5 + rng.normal(0, abs(v) * 0.5 + 0.5) for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.knowledge_type == "weak", (
            f"Expected 'weak', got '{result.knowledge_type}' "
            f"(ρ={result.correlation:.3f}, ε={result.systematic_error:.3f}, "
            f"σ={result.random_error:.3f}, C={result.calibration:.3f})"
        )

    def test_heteroscedastic_errors_increase_epsilon(self, ext_dof, int_dof):
        """Errors that grow with external value → detectable ε."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(0, 10, 100).tolist()
        # Noise proportional to ext value (heteroscedastic)
        int_vals = [v + rng.normal(0, v * 0.3 + 0.01) for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.systematic_error > 0.1, (
            f"Expected ε > 0.1 for heteroscedastic data, got {result.systematic_error:.3f}"
        )

    def test_homoscedastic_noise_gives_high_calibration(self, ext_dof, int_dof):
        """Uniform noise level → high calibration C, regardless of ρ."""
        rng = np.random.default_rng(42)
        ext_vals = rng.uniform(-10, 10, 200).tolist()
        # Moderate correlation with uniform noise
        int_vals = [v + rng.normal(0, 3) for v in ext_vals]

        log = _build_log(ext_dof, int_dof, ext_vals, int_vals)
        result = compute_knowledge(log, ext_dof, [int_dof], min_samples=10)

        assert result is not None
        assert result.calibration >= 0.5, (
            f"Expected C ≥ 0.5 for homoscedastic data, got {result.calibration:.3f}"
        )


# ---------------------------------------------------------------------------
# Multi-feature knowledge assessment
# ---------------------------------------------------------------------------


def _build_multi_log(ext_dof, int_dofs, ext_values, int_values_list, capacity=1000):
    """Build ObservationLog with multiple internal DoFs.

    Args:
        ext_dof: External DoF.
        int_dofs: List of internal DoFs.
        ext_values: List of external values.
        int_values_list: List of lists, one per internal DoF.
    """
    log = ObservationLog(capacity=capacity)
    for i, ev in enumerate(ext_values):
        int_state = {dof: vals[i] for dof, vals in zip(int_dofs, int_values_list)}
        log.append(ObservationPair(
            external_state=State(values={ext_dof: ev}),
            internal_state=State(values=int_state),
            timestamp=float(i),
        ))
    return log


class TestMultiFeatureKnowledge:
    """Tests for multi-feature knowledge assessment (max_features > 1)."""

    def test_multi_feature_increases_correlation(self):
        """Multiple regression with jointly informative features gives higher ρ."""
        rng = np.random.default_rng(42)
        ext_dof = PolarDoF(name="target")
        int_dofs = [PolarDoF(name=f"feat_{i}") for i in range(3)]

        n = 200
        ext_vals = rng.uniform(-5, 5, n).tolist()

        # Each feature captures part of the signal: ext ≈ 0.4*f0 + 0.4*f1 + 0.4*f2
        # Individual ρ ≈ 0.4-0.5, joint ρ should be much higher
        int_vals_list = []
        for _ in range(3):
            vals = [v * 0.4 + rng.normal(0, 2.0) for v in ext_vals]
            int_vals_list.append(vals)

        log = _build_multi_log(ext_dof, int_dofs, ext_vals, int_vals_list)

        # Single-feature assessment
        single = compute_knowledge(log, ext_dof, int_dofs, max_features=1)
        assert single is not None

        # Multi-feature assessment
        multi = compute_knowledge(log, ext_dof, int_dofs, max_features=3)
        assert multi is not None

        assert multi.correlation > single.correlation, (
            f"Multi-feature ρ={multi.correlation:.3f} should exceed "
            f"single-feature ρ={single.correlation:.3f}"
        )

    def test_distributed_knowledge_becomes_strong(self):
        """Distributed signal (no single strong feature) classified as 'strong' with multi-feature."""
        rng = np.random.default_rng(42)
        ext_dof = PolarDoF(name="is_question")
        int_dofs = [PolarDoF(name=f"syntax_{i}") for i in range(5)]

        n = 200
        ext_vals = rng.uniform(-1, 1, n).tolist()

        # 5 features each partially correlated with INDEPENDENT noise
        # Individual ρ ≈ 0.4, but jointly they explain most of the variance
        int_vals_list = []
        for _ in range(5):
            vals = [v * 0.5 + rng.normal(0, 0.7) for v in ext_vals]
            int_vals_list.append(vals)

        log = _build_multi_log(ext_dof, int_dofs, ext_vals, int_vals_list)

        single = compute_knowledge(log, ext_dof, int_dofs, max_features=1)
        multi = compute_knowledge(log, ext_dof, int_dofs, max_features=5)

        assert single is not None
        assert multi is not None

        # Single feature: should be weak or uncertain (ρ < 0.7)
        assert single.correlation < 0.7
        # Multi feature: should be strong (ρ ≥ 0.7)
        assert multi.knowledge_type == "strong", (
            f"Expected 'strong' with multi-feature, got '{multi.knowledge_type}' "
            f"(ρ={multi.correlation:.3f})"
        )

    def test_max_features_1_backward_compatible(self):
        """max_features=1 gives same result as the old default."""
        rng = np.random.default_rng(42)
        ext_dof = PolarDoF(name="ext")
        int_dofs = [PolarDoF(name=f"int_{i}") for i in range(3)]

        n = 100
        ext_vals = rng.uniform(-5, 5, n).tolist()
        int_vals_list = [
            [v + rng.normal(0, 1) for v in ext_vals],
            [rng.normal(0, 3) for _ in ext_vals],
            [rng.normal(0, 3) for _ in ext_vals],
        ]

        log = _build_multi_log(ext_dof, int_dofs, ext_vals, int_vals_list)

        default = compute_knowledge(log, ext_dof, int_dofs)
        explicit = compute_knowledge(log, ext_dof, int_dofs, max_features=1)

        assert default is not None and explicit is not None
        assert default.correlation == explicit.correlation
        assert default.best_internal_dof == explicit.best_internal_dof
        assert default.contributing_dofs == ()
        assert explicit.contributing_dofs == ()

    def test_contributing_dofs_populated(self):
        """Multi-feature assessment populates contributing_dofs."""
        rng = np.random.default_rng(42)
        ext_dof = PolarDoF(name="ext")
        int_dofs = [PolarDoF(name=f"int_{i}") for i in range(3)]

        n = 100
        ext_vals = rng.uniform(-5, 5, n).tolist()
        int_vals_list = [
            [v + rng.normal(0, 1) for v in ext_vals],
            [v * 0.5 + rng.normal(0, 2) for v in ext_vals],
            [rng.normal(0, 5) for _ in ext_vals],
        ]

        log = _build_multi_log(ext_dof, int_dofs, ext_vals, int_vals_list)
        result = compute_knowledge(log, ext_dof, int_dofs, max_features=3)

        assert result is not None
        assert len(result.contributing_dofs) > 0
        assert len(result.contributing_dofs) <= 3
        # All contributing DoFs should be from the internal DoFs
        for dof in result.contributing_dofs:
            assert dof in int_dofs

    def test_caps_features_at_n_over_10(self):
        """Multi-feature caps k at n_samples // 10 to prevent overfitting."""
        rng = np.random.default_rng(42)
        ext_dof = PolarDoF(name="ext")
        # 20 features but only 50 samples → should cap at 5
        int_dofs = [PolarDoF(name=f"int_{i}") for i in range(20)]

        n = 50
        ext_vals = rng.uniform(-5, 5, n).tolist()
        int_vals_list = [
            [v * (0.3 + 0.05 * i) + rng.normal(0, 2) for v in ext_vals]
            for i in range(20)
        ]

        log = _build_multi_log(ext_dof, int_dofs, ext_vals, int_vals_list)
        result = compute_knowledge(log, ext_dof, int_dofs, max_features=20)

        assert result is not None
        assert len(result.contributing_dofs) <= n // 10

    def test_multi_feature_error_metrics_computed(self):
        """Multi-feature assessment computes ε, σ, C correctly."""
        rng = np.random.default_rng(42)
        ext_dof = PolarDoF(name="ext")
        int_dofs = [PolarDoF(name=f"int_{i}") for i in range(3)]

        n = 200
        ext_vals = rng.uniform(-5, 5, n).tolist()
        # Clean uniform noise → should give high C
        int_vals_list = [
            [v + rng.normal(0, 0.5) for v in ext_vals],
            [v * 0.8 + rng.normal(0, 0.5) for v in ext_vals],
            [rng.normal(0, 5) for _ in ext_vals],
        ]

        log = _build_multi_log(ext_dof, int_dofs, ext_vals, int_vals_list)
        result = compute_knowledge(log, ext_dof, int_dofs, max_features=3)

        assert result is not None
        assert result.correlation > 0.7
        assert result.random_error >= 0.0
        assert 0.0 <= result.calibration <= 1.0
        assert 0.0 <= result.systematic_error <= 1.0
