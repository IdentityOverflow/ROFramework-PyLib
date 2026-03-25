"""Tests for OscillatoryNode — the unit observer primitive of the Seed."""

import math
from collections import deque

import numpy as np
import pytest

from ro_framework.seed.node import OscillatoryNode, SeedConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return SeedConfig()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_node(config, node_id="n0", frequency=0.1, is_seed=False):
    return OscillatoryNode(
        node_id=node_id,
        frequency=frequency,
        phase=0.0,
        is_seed_node=is_seed,
        _config=config,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self, config):
        node = _make_node(config)
        assert node.node_id == "n0"
        assert node.frequency == 0.1
        assert node.activation == 0.0
        assert node.branching_ratio == 0.0
        assert node.noise_floor == config.noise_floor
        assert node.drive_amplitude == config.drive_amplitude
        assert isinstance(node.activation_history, deque)
        assert len(node.coupling_weights) == 0

    def test_memory_window_scales_with_frequency(self, config):
        slow = _make_node(config, frequency=0.01)
        fast = _make_node(config, frequency=1.0)
        # Slow node should have much longer window
        assert slow.memory_window > fast.memory_window
        assert slow.memory_window >= 100
        assert fast.memory_window >= 10

    def test_seed_node_flag(self, config):
        node = _make_node(config, is_seed=True)
        assert node.is_seed_node is True

    def test_frequency_clamped_to_range(self, config):
        node = _make_node(config, frequency=999.0)
        assert node.frequency == config.freq_range[1]
        node2 = _make_node(config, frequency=-1.0)
        assert node2.frequency == config.freq_range[0]

    def test_activation_history_maxlen_matches_memory_window(self, config):
        node = _make_node(config, frequency=0.05)
        assert node.activation_history.maxlen == node.memory_window


# ---------------------------------------------------------------------------
# Step (Rule 1 + Rule 3)
# ---------------------------------------------------------------------------

class TestStep:
    def test_bounded_activation(self, config, rng):
        node = _make_node(config, frequency=0.1)
        for _ in range(200):
            act = node.step({}, 0.0, rng)
            assert -1.0 <= act <= 1.0

    def test_oscillates_without_neighbors(self, config, rng):
        """With no coupling, node should oscillate at its natural frequency."""
        node = _make_node(config, frequency=0.1)
        node.noise_floor = 0.0  # suppress noise to see clean oscillation
        activations = []
        for _ in range(200):
            activations.append(node.step({}, 0.0, rng))
        activations = np.array(activations)
        # Should have sign changes (oscillation)
        sign_changes = np.sum(np.diff(np.sign(activations)) != 0)
        assert sign_changes > 5, "Node should oscillate"

    def test_coupling_influences_activation(self, config, rng):
        """Strong positive neighbor input should push activation positive."""
        node = _make_node(config, frequency=0.1)
        node.noise_floor = 0.0
        node.drive_amplitude = 0.0  # suppress drive to isolate coupling
        node.coupling_weights = {"n1": 2.0}

        # Strong positive neighbor
        acts_pos = []
        for _ in range(50):
            acts_pos.append(node.step({"n1": 1.0}, 0.0, rng))

        # Strong negative neighbor
        node2 = _make_node(config, frequency=0.1)
        node2.noise_floor = 0.0
        node2.drive_amplitude = 0.0
        node2.coupling_weights = {"n1": 2.0}
        acts_neg = []
        for _ in range(50):
            acts_neg.append(node2.step({"n1": -1.0}, 0.0, rng))

        assert np.mean(acts_pos) > np.mean(acts_neg)

    def test_noise_floor_present(self, config, rng):
        """With zero drive and no neighbors, activation should still vary."""
        node = _make_node(config, frequency=0.1)
        node.drive_amplitude = 0.0
        activations = []
        for _ in range(100):
            activations.append(node.step({}, 0.0, rng))
        # Should not be all zeros
        assert np.std(activations) > 0.01

    def test_external_drive_influences_activation(self, config, rng):
        """External drive should push activation."""
        node = _make_node(config, frequency=0.1)
        node.noise_floor = 0.0
        node.drive_amplitude = 0.0
        acts = []
        for _ in range(50):
            acts.append(node.step({}, 5.0, rng))
        # Strong positive drive → mostly positive activations
        assert np.mean(acts) > 0.5

    def test_step_count_increments(self, config, rng):
        node = _make_node(config, frequency=0.1)
        assert node._step_count == 0
        node.step({}, 0.0, rng)
        assert node._step_count == 1
        node.step({}, 0.0, rng)
        assert node._step_count == 2

    def test_activation_history_fills(self, config, rng):
        node = _make_node(config, frequency=0.5)
        for _ in range(20):
            node.step({}, 0.0, rng)
        assert len(node.activation_history) == 20


# ---------------------------------------------------------------------------
# Branching ratio
# ---------------------------------------------------------------------------

class TestBranchingRatio:
    def test_updates_when_node_active(self, config, rng):
        """Branching ratio should update when the node is activated."""
        node = _make_node(config, frequency=0.1)
        node.noise_floor = 0.0  # suppress noise for determinism
        node.drive_amplitude = 0.0  # suppress drive
        node.coupling_weights = {"n1": 1.0, "n2": 1.0}

        # Both neighbors strongly active + strong external drive → node
        # will be active each step, and both neighbors count as active
        for _ in range(500):
            node.step({"n1": 0.8, "n2": 0.8}, 2.0, rng)

        # With 2 active neighbors each step, σ should converge toward 2
        assert node.branching_ratio > 1.0

    def test_subcritical_increases_weights(self, config, rng):
        """When σ < 1, co-active pairs should strengthen."""
        node = _make_node(config, frequency=0.1)
        node.coupling_weights = {"n1": 0.5}
        node.branching_ratio = 0.3  # subcritical
        node.activation = 0.8
        node._last_neighborhood = {"n1": 0.8}
        old_w = node.coupling_weights["n1"]
        node.adjust_couplings()
        assert node.coupling_weights["n1"] > old_w

    def test_supercritical_decreases_weights(self, config, rng):
        """When σ > 1, co-active pairs should weaken."""
        node = _make_node(config, frequency=0.1)
        node.coupling_weights = {"n1": 0.5}
        node.branching_ratio = 2.0  # supercritical
        node.activation = 0.8
        node._last_neighborhood = {"n1": 0.8}
        old_w = node.coupling_weights["n1"]
        node.adjust_couplings()
        assert node.coupling_weights["n1"] < old_w


# ---------------------------------------------------------------------------
# Connections (Rules 2a, 2b)
# ---------------------------------------------------------------------------

class TestConnections:
    def test_form_connection(self, config):
        node = _make_node(config)
        node.form_connection("n5", initial_weight=0.01)
        assert "n5" in node.coupling_weights
        assert abs(node.coupling_weights["n5"] - 0.01) < 1e-10

    def test_form_connection_idempotent(self, config):
        node = _make_node(config)
        node.form_connection("n5", initial_weight=0.01)
        node.coupling_weights["n5"] = 0.5  # evolved
        node.form_connection("n5", initial_weight=0.01)
        # Should not overwrite existing connection
        assert abs(node.coupling_weights["n5"] - 0.5) < 1e-10

    def test_remove_connection(self, config):
        node = _make_node(config)
        node.form_connection("n5")
        node.neighbor_coactivation[("n3", "n5")] = 10
        node.remove_connection("n5")
        assert "n5" not in node.coupling_weights
        assert ("n3", "n5") not in node.neighbor_coactivation

    def test_pruning_removes_low_weight_connections(self, config, rng):
        """Connections persistently below threshold get pruned."""
        cfg = SeedConfig(
            prune_weight_threshold=0.1,
            prune_weight_window=5,
            learning_rate=0.0,  # disable learning to keep weight static
        )
        node = _make_node(cfg, frequency=0.1)
        node.coupling_weights = {"n1": 0.001}  # below threshold
        node._last_neighborhood = {"n1": 0.0}

        for _ in range(10):
            pruned = node.adjust_couplings()
            if pruned:
                break
        assert "n1" not in node.coupling_weights

    def test_propose_introductions_detects_coactive_pairs(self, config):
        """High co-activation rate between neighbors triggers proposal."""
        cfg = SeedConfig(connect_threshold=0.2, connect_window=10)
        node = _make_node(cfg, frequency=0.1)
        node._step_count = 100
        # 40% co-activation rate (above 0.2 threshold)
        node.neighbor_coactivation[("a", "b")] = 40
        proposals = node.propose_introductions()
        assert ("a", "b") in proposals

    def test_propose_introductions_respects_window(self, config):
        """No proposals before minimum step count."""
        cfg = SeedConfig(connect_window=50)
        node = _make_node(cfg, frequency=0.1)
        node._step_count = 10  # below window
        node.neighbor_coactivation[("a", "b")] = 100
        assert node.propose_introductions() == []


# ---------------------------------------------------------------------------
# Frequency entrainment
# ---------------------------------------------------------------------------

class TestFrequencyEntrainment:
    def test_entrainment_direction(self, config):
        """Frequency should drift toward neighbors."""
        node = _make_node(config, frequency=0.1)
        node.coupling_weights = {"n1": 1.0}
        old_freq = node.frequency

        # Neighbor at higher frequency
        node.update_frequency({"n1": 0.5})
        assert node.frequency > old_freq

    def test_entrainment_weighted_by_coupling(self, config):
        """Stronger connections have more entrainment influence."""
        node = _make_node(config, frequency=0.1)
        node.coupling_weights = {"n1": 0.01, "n2": 2.0}
        node.update_frequency({"n1": 0.9, "n2": 0.2})
        # Should drift more toward n2 (weight=2.0) at freq 0.2
        assert node.frequency < 0.3  # closer to n2's frequency

    def test_memory_window_updates_after_frequency_change(self, config):
        node = _make_node(config, frequency=0.1)
        old_window = node.memory_window
        node.coupling_weights = {"n1": 2.0}
        # Shift to much higher frequency
        for _ in range(100):
            node.update_frequency({"n1": 0.8})
        # Higher frequency → shorter window
        assert node.memory_window < old_window

    def test_frequency_stays_in_range(self, config):
        node = _make_node(config, frequency=0.1)
        node.coupling_weights = {"n1": 2.0}
        for _ in range(10000):
            node.update_frequency({"n1": 999.0})
        assert node.frequency <= config.freq_range[1]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self, config, rng):
        node = _make_node(config, frequency=0.15, is_seed=True)
        node.coupling_weights = {"n1": 0.5, "n2": -0.3}
        node.neighbor_coactivation[("n1", "n2")] = 42
        # Run a few steps to populate history
        for _ in range(10):
            node.step({"n1": 0.5, "n2": -0.2}, 0.1, rng)

        d = node.to_dict()
        restored = OscillatoryNode.from_dict(d, config)

        assert restored.node_id == node.node_id
        assert abs(restored.frequency - node.frequency) < 1e-10
        assert abs(restored.phase - node.phase) < 1e-10
        assert abs(restored.activation - node.activation) < 1e-10
        assert restored.is_seed_node == node.is_seed_node
        assert restored.coupling_weights == node.coupling_weights
        assert abs(restored.branching_ratio - node.branching_ratio) < 1e-10
        assert restored._step_count == node._step_count
        assert len(restored.activation_history) == len(node.activation_history)
        assert restored.neighbor_coactivation == node.neighbor_coactivation

    def test_config_to_dict_round_trip(self):
        cfg = SeedConfig(n_init=32, freq_range=(0.05, 2.0))
        d = cfg.to_dict()
        restored = SeedConfig.from_dict(d)
        assert restored.n_init == 32
        assert restored.freq_range == (0.05, 2.0)
