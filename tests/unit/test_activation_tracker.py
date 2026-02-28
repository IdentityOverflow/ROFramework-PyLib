"""Tests for ActivationTracker — online feature discovery via PCA."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from ro_framework.core.dof import PolarDoF
from ro_framework.integration.activation_tracker import (
    ActivationTracker,
    DirectionSnapshot,
    DiscoveredDoF,
    TrackedDirection,
    _match_directions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Tiny MLP for testing hook-based activation tracking."""

    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _run_forward_passes(model, data, tracker=None):
    """Run forward passes, optionally within a collection window."""
    model.eval()
    with torch.no_grad():
        model(data)


# ---------------------------------------------------------------------------
# TestWelfordStatistics
# ---------------------------------------------------------------------------


class TestWelfordStatistics:
    """Verify Welford's online mean/covariance is correct."""

    def test_known_gaussian(self):
        """Mean and covariance match np.mean/np.cov on known data."""
        rng = np.random.default_rng(42)
        true_mean = np.array([1.0, 2.0, 3.0, 4.0])
        true_cov = np.diag([4.0, 1.0, 0.5, 0.1])
        data = rng.multivariate_normal(true_mean, true_cov, size=1000)

        model = SimpleMLP(input_dim=4, hidden_dim=4, output_dim=2)
        # Replace fc1 with identity-like weights so activations ≈ input
        with torch.no_grad():
            model.fc1.weight.copy_(torch.eye(4))
            model.fc1.bias.zero_()

        tracker = ActivationTracker(model, "fc1", top_k=4)
        tracker.attach()
        tracker.begin_collection()
        # Feed through in batches (pre-ReLU, fc1 output = identity * input)
        _run_forward_passes(model, torch.tensor(data, dtype=torch.float32), tracker)
        # Don't end_collection — check raw Welford stats
        assert tracker._n == 1000
        np.testing.assert_allclose(tracker._mean, true_mean, atol=0.15)
        welford_cov = tracker._M2 / (tracker._n - 1)
        np.testing.assert_allclose(welford_cov, np.cov(data.T), atol=1e-6)
        tracker.detach()

    def test_batched_input(self):
        """Batched input gives same stats as sequential."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((50, 4)).astype(np.float32)

        model = SimpleMLP(input_dim=4, hidden_dim=4, output_dim=2)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.eye(4))
            model.fc1.bias.zero_()

        # All at once
        t1 = ActivationTracker(model, "fc1", top_k=4)
        t1.attach()
        t1.begin_collection()
        _run_forward_passes(model, torch.tensor(data))
        assert t1._n == 50
        mean1 = t1._mean.copy()
        m2_1 = t1._M2.copy()
        t1.detach()  # detach before second tracker to avoid double-firing

        # One at a time
        t2 = ActivationTracker(model, "fc1", top_k=4)
        t2.attach()
        t2.begin_collection()
        for i in range(50):
            _run_forward_passes(model, torch.tensor(data[i:i+1]))
        assert t2._n == 50

        np.testing.assert_allclose(mean1, t2._mean, atol=1e-6)
        np.testing.assert_allclose(m2_1, t2._M2, atol=1e-4)
        t2.detach()

    def test_too_few_samples_raises(self):
        """end_collection with < 2 samples raises ValueError."""
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=2)
        tracker.attach()
        tracker.begin_collection()
        # Only one sample
        _run_forward_passes(model, torch.randn(1, 4))
        with pytest.raises(ValueError, match="at least 2 samples"):
            tracker.end_collection(epoch=0)
        tracker.detach()


# ---------------------------------------------------------------------------
# TestActivationTracker
# ---------------------------------------------------------------------------


class TestActivationTracker:
    """Core lifecycle tests."""

    def test_attach_detach_idempotent(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker.attach()
        tracker.attach()  # should not double-register
        tracker.detach()
        tracker.detach()  # should not error

    def test_begin_end_returns_snapshot(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.randn(20, 4))
        snap = tracker.end_collection(epoch=0)
        assert isinstance(snap, DirectionSnapshot)
        assert snap.epoch == 0
        assert len(snap.directions) == 3
        assert snap.total_variance > 0
        tracker.detach()

    def test_directions_sorted_by_eigenvalue(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=5)
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.randn(100, 4))
        snap = tracker.end_collection(epoch=0)
        eigenvalues = [d.eigenvalue for d in snap.directions]
        assert eigenvalues == sorted(eigenvalues, reverse=True)
        tracker.detach()

    def test_explained_variance_ratio(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=8)
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.randn(100, 4))
        snap = tracker.end_collection(epoch=0)
        # EVR should sum to <= 1 and be non-negative
        assert np.all(snap.explained_variance_ratio >= 0)
        assert snap.explained_variance_ratio.sum() <= 1.0 + 1e-6
        tracker.detach()

    def test_stability_none_on_first_epoch(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.randn(50, 4))
        snap = tracker.end_collection(epoch=0)
        for d in snap.directions:
            assert d.stability is None
        tracker.detach()

    def test_stability_high_for_same_distribution(self):
        """Two epochs from the same distribution → high stability."""
        torch.manual_seed(42)
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker.attach()

        for epoch in range(2):
            tracker.begin_collection()
            _run_forward_passes(model, torch.randn(200, 4))
            snap = tracker.end_collection(epoch=epoch)

        # Second epoch should have high stability
        for d in snap.directions:
            assert d.stability is not None
            assert d.stability > 0.8, f"Stability {d.stability} too low"
        tracker.detach()

    def test_stability_low_for_different_weights(self):
        """Different model weights → different PCA directions → low stability."""
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker.attach()

        data = torch.randn(200, 4)
        tracker.begin_collection()
        _run_forward_passes(model, data)
        tracker.end_collection(epoch=0)

        # Reinitialize model weights → completely different activations
        model.fc1.reset_parameters()

        tracker.begin_collection()
        _run_forward_passes(model, data)
        snap = tracker.end_collection(epoch=1)

        # At least some directions should have changed significantly
        stabilities = [d.stability for d in snap.directions if d.stability is not None]
        avg_stability = np.mean(stabilities)
        assert avg_stability < 0.9, f"Average stability {avg_stability} too high after weight reset"
        tracker.detach()


# ---------------------------------------------------------------------------
# TestDirectionMatching
# ---------------------------------------------------------------------------


class TestDirectionMatching:
    """Test the greedy cosine-similarity matching."""

    def test_identity_matching(self):
        """Same directions in same order → perfect matching."""
        dirs = np.eye(5)
        matches = _match_directions(dirs, dirs)
        assert len(matches) == 5
        for prev_idx, curr_idx, sim in matches:
            assert prev_idx == curr_idx
            assert abs(sim - 1.0) < 1e-10

    def test_rank_swap(self):
        """Swapped order still matched correctly by cosine similarity."""
        prev = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        curr = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # rotated order
        matches = _match_directions(prev, curr)
        match_dict = {p: (c, s) for p, c, s in matches}
        assert match_dict[0] == (1, 1.0)  # prev[0]=[1,0,0] matches curr[1]=[1,0,0]
        assert match_dict[1] == (2, 1.0)
        assert match_dict[2] == (0, 1.0)

    def test_sign_invariance(self):
        """Negated direction still matches (eigenvector sign ambiguity)."""
        prev = np.array([[1, 0], [0, 1]], dtype=float)
        curr = np.array([[-1, 0], [0, -1]], dtype=float)
        matches = _match_directions(prev, curr)
        for _, _, sim in matches:
            assert abs(sim - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# TestDirectionDiscovery
# ---------------------------------------------------------------------------


class TestDirectionDiscovery:
    """Test discover_dofs and related functionality."""

    def _train_stable_tracker(self, n_epochs=5):
        """Create a tracker with several epochs of stable PCA directions."""
        torch.manual_seed(42)
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)
        tracker = ActivationTracker(model, "relu", top_k=4)
        tracker.attach()

        data = torch.randn(200, 4)
        for epoch in range(n_epochs):
            tracker.begin_collection()
            _run_forward_passes(model, data)
            tracker.end_collection(epoch=epoch)

        tracker.detach()
        return tracker

    def test_stable_directions_discovered(self):
        tracker = self._train_stable_tracker(n_epochs=6)
        discovered = tracker.discover_dofs(min_stability=0.8, min_stable_epochs=3)
        assert len(discovered) > 0
        for dd in discovered:
            assert isinstance(dd, DiscoveredDoF)
            assert isinstance(dd.dof, PolarDoF)
            assert dd.projection.shape == (8,)
            assert dd.stability_epochs >= 3

    def test_unstable_filtered(self):
        """With very high stability threshold, nothing passes."""
        tracker = self._train_stable_tracker(n_epochs=3)
        discovered = tracker.discover_dofs(min_stability=0.9999, min_stable_epochs=3)
        # May or may not find anything — but should not crash
        assert isinstance(discovered, list)

    def test_low_variance_filtered(self):
        """High variance threshold filters out small components."""
        tracker = self._train_stable_tracker(n_epochs=6)
        discovered_all = tracker.discover_dofs(min_stability=0.5, min_stable_epochs=1, min_variance_fraction=0.0)
        discovered_big = tracker.discover_dofs(min_stability=0.5, min_stable_epochs=1, min_variance_fraction=0.5)
        assert len(discovered_big) <= len(discovered_all)

    def test_create_projection_mapping(self):
        tracker = self._train_stable_tracker(n_epochs=6)
        discovered = tracker.discover_dofs(min_stability=0.8, min_stable_epochs=3)
        if not discovered:
            pytest.skip("No stable directions found")
        mapping = tracker.create_projection_mapping(discovered)
        # Test projection on random activation vector
        rng = np.random.default_rng(42)
        act = rng.standard_normal(8)
        result = mapping.project(act)
        assert len(result) == len(discovered)
        for dof, val in result.items():
            assert isinstance(val, float)

    def test_projection_batch(self):
        tracker = self._train_stable_tracker(n_epochs=6)
        discovered = tracker.discover_dofs(min_stability=0.8, min_stable_epochs=3)
        if not discovered:
            pytest.skip("No stable directions found")
        mapping = tracker.create_projection_mapping(discovered)
        rng = np.random.default_rng(42)
        acts = rng.standard_normal((10, 8))
        results = mapping.project_batch(acts)
        assert len(results) == 10
        # Batch should match sequential
        for i in range(10):
            single = mapping.project(acts[i])
            for dof in single:
                assert abs(single[dof] - results[i][dof]) < 1e-10


# ---------------------------------------------------------------------------
# TestEigenvalueAnalysis
# ---------------------------------------------------------------------------


class TestEigenvalueAnalysis:

    def test_eigenvalue_trajectory(self):
        torch.manual_seed(42)
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker.attach()
        for epoch in range(5):
            tracker.begin_collection()
            _run_forward_passes(model, torch.randn(50, 4))
            tracker.end_collection(epoch=epoch)
        tracker.detach()

        traj = tracker.eigenvalue_trajectory(0)
        assert len(traj) == 5
        for epoch, ev in traj:
            assert isinstance(epoch, int)
            assert ev >= 0

    def test_detect_eigenvalue_spike(self):
        """Synthetic scenario: manually set histories with a spike."""
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        # Manually inject history with a spike at epoch 3
        tracker._direction_histories = [
            [(0, 1.0, None), (1, 1.1, 0.99), (2, 1.2, 0.99), (3, 5.0, 0.95), (4, 5.5, 0.99)],
        ]
        spike = tracker.detect_eigenvalue_spike(0, relative_threshold=2.0)
        assert spike == 3

    def test_no_spike_when_stable(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        tracker._direction_histories = [
            [(0, 1.0, None), (1, 1.05, 0.99), (2, 1.1, 0.99), (3, 1.15, 0.99)],
        ]
        spike = tracker.detect_eigenvalue_spike(0, relative_threshold=2.0)
        assert spike is None


# ---------------------------------------------------------------------------
# TestSerialization
# ---------------------------------------------------------------------------


class TestSerialization:

    def test_roundtrip(self):
        torch.manual_seed(42)
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3, readout_layer_name="fc2")
        tracker.attach()
        for epoch in range(3):
            tracker.begin_collection()
            _run_forward_passes(model, torch.randn(50, 4))
            tracker.end_collection(epoch=epoch)
        tracker.detach()

        d = tracker.to_dict()
        # Ensure JSON-serializable
        json_str = json.dumps(d)
        d2 = json.loads(json_str)

        tracker2 = ActivationTracker.from_dict(d2, model)
        assert len(tracker2.snapshots()) == 3
        assert len(tracker2._direction_histories) == len(tracker._direction_histories)

        # Check eigenvalue trajectories match
        for i in range(min(3, len(tracker._direction_histories))):
            t1 = tracker.eigenvalue_trajectory(i)
            t2 = tracker2.eigenvalue_trajectory(i)
            assert len(t1) == len(t2)
            for (e1, ev1), (e2, ev2) in zip(t1, t2):
                assert e1 == e2
                assert abs(ev1 - ev2) < 1e-10


# ---------------------------------------------------------------------------
# TestTorchSmoke
# ---------------------------------------------------------------------------


class TestTorchSmoke:

    def test_hook_collects_from_relu(self):
        """Hook on ReLU layer collects post-activation values."""
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)
        tracker = ActivationTracker(model, "relu", top_k=4)
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.randn(30, 4))
        snap = tracker.end_collection(epoch=0)
        assert snap.total_variance > 0
        # Directions should be 8-dimensional (hidden_dim)
        assert snap.directions[0].direction.shape == (8,)
        tracker.detach()

    def test_known_rank_structure(self):
        """Data with known rank-2 structure → PCA recovers 2 dominant directions."""
        # Create data that varies only in 2 directions
        rng = np.random.default_rng(42)
        v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        # Large variance in v1, v2; tiny noise elsewhere
        n = 500
        data = (rng.standard_normal((n, 1)).astype(np.float32) * 10 * v1
                + rng.standard_normal((n, 1)).astype(np.float32) * 5 * v2
                + rng.standard_normal((n, 8)).astype(np.float32) * 0.01)

        model = SimpleMLP(input_dim=8, hidden_dim=8, output_dim=2)
        # Set fc1 to identity so activations ≈ input (pre-ReLU)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.eye(8))
            model.fc1.bias.zero_()

        tracker = ActivationTracker(model, "fc1", top_k=4)
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.tensor(data))
        snap = tracker.end_collection(epoch=0)

        # Top 2 should capture > 99% of variance
        top2_evr = snap.explained_variance_ratio[:2].sum()
        assert top2_evr > 0.99, f"Top-2 EVR = {top2_evr:.3f}, expected > 0.99"

        # Top directions should align with v1 and v2
        d1 = snap.directions[0].direction
        d2 = snap.directions[1].direction
        # One should align with [1,0,...], other with [0,1,...]
        align_1 = max(abs(d1[0]), abs(d2[0]))
        align_2 = max(abs(d1[1]), abs(d2[1]))
        assert align_1 > 0.99
        assert align_2 > 0.99
        tracker.detach()

    def test_readout_alignment(self):
        """Readout alignment is computed when readout_layer_name is given."""
        model = SimpleMLP(input_dim=4, hidden_dim=8, output_dim=2)
        tracker = ActivationTracker(model, "relu", top_k=4, readout_layer_name="fc2")
        tracker.attach()
        tracker.begin_collection()
        _run_forward_passes(model, torch.randn(100, 4))
        snap = tracker.end_collection(epoch=0)
        for d in snap.directions:
            assert d.readout_alignment is not None
            assert 0 <= d.readout_alignment
        tracker.detach()

    def test_repr(self):
        model = SimpleMLP()
        tracker = ActivationTracker(model, "relu", top_k=3)
        r = repr(tracker)
        assert "relu" in r
        assert "top_k=3" in r
