"""Tests for criticality monitoring tools."""

import numpy as np
import pytest

from ro_framework.seed.criticality import (
    extract_cascades,
    fast_mi,
    measure_branching_ratio,
    verify_power_law,
)


# ---------------------------------------------------------------------------
# extract_cascades
# ---------------------------------------------------------------------------

class TestExtractCascades:
    def test_known_sequence(self):
        series = np.array([0.5, 0.5, 0.1, 0.5, 0.1, 0.1])
        cascades = extract_cascades(series, threshold=0.3)
        assert cascades == [2, 1]

    def test_single_long_cascade(self):
        series = np.array([0.5, 0.4, 0.6, 0.8, 0.5])
        cascades = extract_cascades(series, threshold=0.3)
        assert cascades == [5]

    def test_empty_sequence(self):
        assert extract_cascades(np.array([]), threshold=0.3) == []

    def test_all_below_threshold(self):
        series = np.array([0.1, 0.2, 0.1, 0.0])
        assert extract_cascades(series, threshold=0.3) == []

    def test_negative_activations(self):
        """Cascades count absolute value above threshold."""
        series = np.array([-0.5, -0.4, 0.1, -0.6])
        cascades = extract_cascades(series, threshold=0.3)
        assert cascades == [2, 1]


# ---------------------------------------------------------------------------
# verify_power_law
# ---------------------------------------------------------------------------

class TestVerifyPowerLaw:
    def test_synthetic_power_law(self):
        """Generate data from known discrete power law, verify detection."""
        rng = np.random.default_rng(42)
        # numpy's zipf(a) generates from P(k) = k^(-a) / ζ(a)
        alpha = 2.5
        samples = rng.zipf(alpha, size=500)

        is_pl, fitted_alpha, ks = verify_power_law(samples.tolist())
        assert is_pl, f"Should detect power law: α={fitted_alpha:.2f}, KS={ks:.3f}"
        assert 1.5 < fitted_alpha < 3.5

    def test_rejects_exponential(self):
        """Exponential data should not be classified as power law."""
        rng = np.random.default_rng(42)
        samples = rng.exponential(scale=3.0, size=500)
        samples = np.maximum(1, np.round(samples)).astype(int)

        is_pl, alpha, ks = verify_power_law(samples.tolist())
        # Should either reject or have poor KS
        # (exponential can sometimes look power-law-ish for small ranges)
        # The key test: KS for power law should be worse than for exponential
        assert not is_pl or ks > 0.1

    def test_insufficient_samples(self):
        is_pl, alpha, ks = verify_power_law([1, 2, 3], min_samples=50)
        assert not is_pl
        assert alpha == 0.0

    def test_all_same_size(self):
        """All cascades same size — not a power law."""
        is_pl, alpha, ks = verify_power_law([3] * 100)
        assert not is_pl


# ---------------------------------------------------------------------------
# measure_branching_ratio
# ---------------------------------------------------------------------------

class TestMeasureBranchingRatio:
    def test_critical_branching(self):
        """Exactly 1 neighbor activates after each node activation → σ = 1."""
        n = 100
        # Node active every other step
        node_hist = np.array([0.5, 0.0] * (n // 2))
        # Exactly 1 neighbor active at the step after node is active
        neighbor_hist = np.array([0.0, 0.5] * (n // 2))

        sigma = measure_branching_ratio(
            node_hist, {"n1": neighbor_hist}, threshold=0.3
        )
        assert abs(sigma - 1.0) < 0.01

    def test_subcritical(self):
        """No neighbors activate → σ = 0."""
        n = 100
        node_hist = np.array([0.5, 0.0] * (n // 2))
        neighbor_hist = np.zeros(n)  # never active

        sigma = measure_branching_ratio(
            node_hist, {"n1": neighbor_hist}, threshold=0.3
        )
        assert sigma < 0.01

    def test_supercritical(self):
        """Both neighbors always activate → σ = 2."""
        n = 100
        node_hist = np.array([0.5] * n)
        n1_hist = np.array([0.5] * n)
        n2_hist = np.array([0.5] * n)

        sigma = measure_branching_ratio(
            node_hist, {"n1": n1_hist, "n2": n2_hist}, threshold=0.3
        )
        assert abs(sigma - 2.0) < 0.01

    def test_empty_history(self):
        assert measure_branching_ratio(np.array([]), {}, threshold=0.3) == 0.0

    def test_no_neighbors(self):
        node_hist = np.array([0.5, 0.5, 0.5])
        assert measure_branching_ratio(node_hist, {}, threshold=0.3) == 0.0


# ---------------------------------------------------------------------------
# fast_mi
# ---------------------------------------------------------------------------

class TestFastMI:
    def test_identical_signals(self):
        """MI of identical signals should be high."""
        x = np.random.default_rng(42).standard_normal(200)
        mi = fast_mi(x, x)
        assert mi > 1.0  # high MI

    def test_independent_signals(self):
        """MI of independent signals should be near zero."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        mi = fast_mi(x, y)
        assert mi < 0.2  # near zero

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        assert fast_mi(x, y) >= 0.0

    def test_short_arrays(self):
        assert fast_mi(np.array([1.0]), np.array([2.0])) == 0.0
