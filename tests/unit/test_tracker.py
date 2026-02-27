"""Unit tests for KnowledgeTracker — trajectory tracking and phase detection."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.integration.wrappers import create_dofs_for_vector, wrap_callable
from ro_framework.knowledge.tracker import KnowledgeTracker, TrajectoryPoint
from ro_framework.observer.observer import Observer, ObservationPair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_observer(scale: float = 1.0):
    """Create an observer wrapping y = scale * x."""
    in_dofs = create_dofs_for_vector(1, prefix="x")
    out_dofs = create_dofs_for_vector(1, prefix="y")

    def fn(x: np.ndarray) -> np.ndarray:
        return x * scale

    obs = wrap_callable(fn, in_dofs, out_dofs, name="linear")
    return obs, in_dofs, out_dofs


def _feed_observations(observer, in_dofs, n, rng, noise_std=0.0):
    """Feed n random observations into the observer."""
    for _ in range(n):
        val = rng.uniform(-5, 5)
        observer.observe(State(values={in_dofs[0]: val}))


# ---------------------------------------------------------------------------
# Basic trajectory tracking
# ---------------------------------------------------------------------------


class TestTrajectoryTracking:
    """Tests for step(), trajectory(), latest()."""

    def test_step_records_trajectory(self):
        """step() should record one TrajectoryPoint per call."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs)
        tracker.step(0)
        tracker.step(1)

        traj = tracker.trajectory(in_dofs[0])
        assert len(traj) == 2
        assert traj[0].epoch == 0
        assert traj[1].epoch == 1

    def test_step_returns_assessments(self):
        """step() should return a dict of assessments."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs)
        results = tracker.step(0)

        assert in_dofs[0] in results
        assert results[in_dofs[0]] is not None
        assert results[in_dofs[0]].correlation > 0.5

    def test_assess_interval(self):
        """With interval=2, only every other step should record."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, assess_interval=2)

        # step 1 (first call) always records
        result0 = tracker.step(0)
        assert len(result0) > 0

        # step 2 is skipped (step_count=2, 2%2==0 → records)
        result1 = tracker.step(1)
        assert len(result1) > 0

        # step 3 is skipped (step_count=3, 3%2!=0 and step_count!=1)
        result2 = tracker.step(2)
        assert len(result2) == 0

        # step 4 records (step_count=4, 4%2==0)
        result3 = tracker.step(3)
        assert len(result3) > 0

        traj = tracker.trajectory(in_dofs[0])
        assert len(traj) == 3  # epochs 0, 1, 3

    def test_latest(self):
        """latest() returns the most recent assessment."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs)
        assert tracker.latest(in_dofs[0]) is None  # no steps yet

        tracker.step(0)
        tracker.step(1)
        latest = tracker.latest(in_dofs[0])

        assert latest is not None
        traj = tracker.trajectory(in_dofs[0])
        assert latest is traj[-1].assessment

    def test_default_external_dofs(self):
        """Without explicit dofs, uses observer.external_dofs."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs)  # no explicit dofs
        tracker.step(0)

        traj = tracker.trajectory(in_dofs[0])
        assert len(traj) == 1

    def test_trajectory_empty_for_unknown_dof(self):
        """trajectory() returns empty list for untracked DoF."""
        obs, in_dofs, _ = _make_linear_observer()
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs)

        unknown = PolarDoF(name="unknown")
        assert tracker.trajectory(unknown) == []

    def test_insufficient_data_not_recorded(self):
        """If assess_knowledge returns None, no point is recorded."""
        obs, in_dofs, _ = _make_linear_observer()
        # Don't feed any observations — too few for assessment
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, min_samples=10)
        results = tracker.step(0)

        assert results[in_dofs[0]] is None
        assert len(tracker.trajectory(in_dofs[0])) == 0


# ---------------------------------------------------------------------------
# Phase transition detection
# ---------------------------------------------------------------------------


class TestPhaseDetection:
    """Tests for detect_grokking, detect_resonance, detect_forgetting."""

    def test_detect_grokking(self):
        """Detect transition from weak to strong knowledge."""
        obs, in_dofs, out_dofs = _make_linear_observer()
        rng = np.random.default_rng(42)
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, min_samples=10)

        # Phase 1: Feed noisy data → weak knowledge
        for _ in range(15):
            val = rng.uniform(-5, 5)
            # Noisy: output doesn't correlate well with input
            obs.observation_log.append(
                __import__("ro_framework.observer.observer", fromlist=["ObservationPair"]).ObservationPair(
                    external_state=State(values={in_dofs[0]: val}),
                    internal_state=State(values={out_dofs[0]: rng.uniform(-5, 5)}),
                    timestamp=float(len(obs.observation_log)),
                )
            )
        tracker.step(0)

        # Phase 2: Clear and feed clean data → strong knowledge
        obs.clear_memory()
        for _ in range(20):
            val = rng.uniform(-5, 5)
            obs.observe(State(values={in_dofs[0]: val}))
        tracker.step(1)

        grok_epoch = tracker.detect_grokking(in_dofs[0])
        assert grok_epoch == 1

    def test_no_grokking_returns_none(self):
        """Consistently weak knowledge → no grokking detected."""
        in_dofs = create_dofs_for_vector(1, prefix="x")
        out_dofs = create_dofs_for_vector(1, prefix="y")

        # Random function: no correlation
        def random_fn(x):
            return np.random.default_rng(hash(x.tobytes()) % 2**32).uniform(-1, 1, size=x.shape)

        obs = wrap_callable(random_fn, in_dofs, out_dofs, name="random")
        rng = np.random.default_rng(42)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, min_samples=10)

        for epoch in range(5):
            for _ in range(20):
                obs.observe(State(values={in_dofs[0]: rng.uniform(-5, 5)}))
            tracker.step(epoch)

        assert tracker.detect_grokking(in_dofs[0]) is None

    def test_detect_forgetting(self):
        """Detect when correlation drops from a peak."""
        obs, in_dofs, out_dofs = _make_linear_observer()
        rng = np.random.default_rng(42)
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, min_samples=10)

        # Phase 1: Good data → high ρ
        for _ in range(20):
            obs.observe(State(values={in_dofs[0]: rng.uniform(-5, 5)}))
        tracker.step(0)

        # Phase 2: Replace with noisy data → ρ drops
        obs.clear_memory()
        from ro_framework.observer.observer import ObservationPair
        for _ in range(20):
            val = rng.uniform(-5, 5)
            obs.observation_log.append(
                ObservationPair(
                    external_state=State(values={in_dofs[0]: val}),
                    internal_state=State(values={out_dofs[0]: rng.uniform(-5, 5)}),
                    timestamp=float(len(obs.observation_log)),
                )
            )
        tracker.step(1)

        forgetting = tracker.detect_forgetting(in_dofs[0], rho_drop=0.2)
        assert len(forgetting) > 0
        assert forgetting[0] == 1

    def test_detect_resonance(self):
        """Detect rising correlation with high noise (pre-grokking)."""
        obs, in_dofs, out_dofs = _make_linear_observer()
        rng = np.random.default_rng(42)
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, min_samples=10)

        from ro_framework.observer.observer import ObservationPair

        # Epoch 0: Low correlation, high noise
        for _ in range(20):
            val = rng.uniform(-5, 5)
            obs.observation_log.append(
                ObservationPair(
                    external_state=State(values={in_dofs[0]: val}),
                    internal_state=State(values={out_dofs[0]: rng.uniform(-5, 5)}),
                    timestamp=float(len(obs.observation_log)),
                )
            )
        tracker.step(0)

        # Epoch 1: Moderate correlation, still high noise (resonance)
        obs.clear_memory()
        for _ in range(20):
            val = rng.uniform(-5, 5)
            obs.observation_log.append(
                ObservationPair(
                    external_state=State(values={in_dofs[0]: val}),
                    internal_state=State(values={out_dofs[0]: val + rng.normal(0, 3)}),
                    timestamp=float(len(obs.observation_log)),
                )
            )
        tracker.step(1)

        resonance = tracker.detect_resonance(in_dofs[0], rho_threshold=0.3, sigma_threshold=0.3)
        # Epoch 1 should show resonance: ρ rising, σ still high
        assert 1 in resonance

    def test_no_forgetting_when_stable(self):
        """Stable high correlation → no forgetting detected."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, min_samples=10)

        for epoch in range(3):
            obs.clear_memory()
            for _ in range(20):
                obs.observe(State(values={in_dofs[0]: rng.uniform(-5, 5)}))
            tracker.step(epoch)

        assert tracker.detect_forgetting(in_dofs[0]) == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestTrackerSerialization:
    """Tests for save/load and to_dict/from_dict."""

    def test_serialization_roundtrip(self):
        """Save and load should preserve trajectory data."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs)
        tracker.step(0)
        tracker.step(1)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = Path(f.name)
        tracker.save(save_path)

        # Load
        loaded = KnowledgeTracker.load(save_path, observer=obs)
        save_path.unlink()

        # Verify
        orig_traj = tracker.trajectory(in_dofs[0])
        loaded_traj = loaded.trajectory(in_dofs[0])

        assert len(loaded_traj) == len(orig_traj)
        for orig, loaded_pt in zip(orig_traj, loaded_traj):
            assert orig.epoch == loaded_pt.epoch
            assert abs(orig.assessment.correlation - loaded_pt.assessment.correlation) < 1e-10
            assert abs(orig.assessment.systematic_error - loaded_pt.assessment.systematic_error) < 1e-10
            assert abs(orig.assessment.random_error - loaded_pt.assessment.random_error) < 1e-10
            assert abs(orig.assessment.calibration - loaded_pt.assessment.calibration) < 1e-10
            assert orig.assessment.n_samples == loaded_pt.assessment.n_samples

    def test_to_dict_from_dict(self):
        """to_dict/from_dict roundtrip."""
        obs, in_dofs, _ = _make_linear_observer()
        rng = np.random.default_rng(42)
        _feed_observations(obs, in_dofs, 20, rng)

        tracker = KnowledgeTracker(obs, external_dofs=in_dofs, assess_interval=3, min_samples=5)
        tracker.step(0)

        d = tracker.to_dict()
        loaded = KnowledgeTracker.from_dict(d, observer=obs)

        assert loaded.assess_interval == 3
        assert loaded.min_samples == 5
        assert len(loaded.trajectory(in_dofs[0])) == 1

    def test_serialization_preserves_config(self):
        """Config (interval, min_samples) should survive serialization."""
        obs, in_dofs, _ = _make_linear_observer()

        tracker = KnowledgeTracker(obs, assess_interval=5, min_samples=15)
        d = tracker.to_dict()
        loaded = KnowledgeTracker.from_dict(d, observer=obs)

        assert loaded.assess_interval == 5
        assert loaded.min_samples == 15


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestTrackerRepr:
    def test_repr(self):
        obs, in_dofs, _ = _make_linear_observer()
        tracker = KnowledgeTracker(obs, external_dofs=in_dofs)
        r = repr(tracker)
        assert "KnowledgeTracker" in r
        assert "dofs=1" in r
        assert "points=0" in r


# ---------------------------------------------------------------------------
# PyTorch integration smoke test
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class TestTrackerTorchSmoke:
    """Smoke test: tracker + real PyTorch model (few epochs, no grokking expected)."""

    def test_torch_training_pipeline(self):
        """Tracker records trajectory during a short PyTorch training run."""
        import torch.nn as nn

        p = 7
        hidden_dim = 16

        # Model
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.ea = nn.Embedding(p, 16)
                self.eb = nn.Embedding(p, 16)
                self.fc1 = nn.Linear(32, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, p)

            def forward(self, a, b):
                return self.fc2(torch.relu(self.fc1(
                    torch.cat([self.ea(a), self.eb(b)], -1))))

            def get_hidden(self, a, b):
                with torch.no_grad():
                    return torch.relu(self.fc1(
                        torch.cat([self.ea(a), self.eb(b)], -1)))

        torch.manual_seed(0)
        model = MLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()

        # Data
        all_a = torch.arange(p).repeat_interleave(p)
        all_b = torch.arange(p).repeat(p)
        all_y = (all_a + all_b) % p

        # Observer + Tracker
        fourier_dofs = [PolarDoF(name="sin_1"), PolarDoF(name="cos_1")]
        neuron_dofs = [PolarDoF(name=f"h_{i}") for i in range(hidden_dim)]

        class Dummy:
            def __call__(self, state):
                return State(values={d: 0.0 for d in neuron_dofs})

        obs = Observer(
            name="smoke",
            internal_dofs=neuron_dofs,
            external_dofs=fourier_dofs,
            world_model=Dummy(),
            log_capacity=p * p + 1,
        )
        tracker = KnowledgeTracker(obs, external_dofs=fourier_dofs)

        # Short training: 100 epochs, eval every 50
        for epoch in range(100):
            loss = crit(model(all_a, all_b), all_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % 50 == 0:
                obs.clear_memory()
                hidden = model.get_hidden(all_a, all_b).numpy()
                sums = ((all_a + all_b) % p).numpy()

                for idx in range(len(all_a)):
                    s = sums[idx]
                    h = hidden[idx]
                    angle = 2 * np.pi * s / p
                    ext = {fourier_dofs[0]: float(np.sin(angle)),
                           fourier_dofs[1]: float(np.cos(angle))}
                    internal = {neuron_dofs[i]: float(h[i])
                                for i in range(hidden_dim)}
                    obs.observation_log.append(ObservationPair(
                        external_state=State(values=ext),
                        internal_state=State(values=internal),
                        timestamp=float(len(obs.observation_log)),
                    ))
                tracker.step(epoch)

        # Verify tracker recorded data
        traj_sin = tracker.trajectory(fourier_dofs[0])
        traj_cos = tracker.trajectory(fourier_dofs[1])
        assert len(traj_sin) == 2  # epochs 0 and 50
        assert len(traj_cos) == 2
        assert traj_sin[0].epoch == 0
        assert traj_sin[1].epoch == 50
        assert traj_sin[0].assessment.n_samples == p * p

        # Phase detection doesn't crash
        tracker.detect_grokking(fourier_dofs[0])
        tracker.detect_resonance(fourier_dofs[0])
        tracker.detect_forgetting(fourier_dofs[0])
