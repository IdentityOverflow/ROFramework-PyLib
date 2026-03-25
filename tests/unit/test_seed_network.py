"""Tests for SeedNetwork — the collective observer."""

import numpy as np
import pytest

from ro_framework.seed.network import SeedNetwork, SensorInterface, ActuatorInterface
from ro_framework.seed.node import OscillatoryNode, SeedConfig


# ---------------------------------------------------------------------------
# Minimal sensor/actuator for testing
# ---------------------------------------------------------------------------

class SimpleSensor:
    """Maps external input uniformly to all nodes."""

    def __call__(self, external_input, node_frequencies):
        val = float(np.mean(external_input)) if len(external_input) > 0 else 0.0
        return {nid: val for nid in node_frequencies}


class SimpleActuator:
    """Returns mean activation as a 1-D array."""

    def __call__(self, node_activations):
        if not node_activations:
            return np.array([0.0])
        return np.array([np.mean(list(node_activations.values()))])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return SeedConfig(n_init=16, k_neighbors=4, n_seed_nodes=4)


@pytest.fixture
def small_config():
    return SeedConfig(n_init=8, k_neighbors=4, n_seed_nodes=2, max_nodes=12)


@pytest.fixture
def sensor():
    return SimpleSensor()


@pytest.fixture
def actuator():
    return SimpleActuator()


def _make_network(config, sensor, actuator, seed=42):
    return SeedNetwork(config, sensor, actuator, seed=seed)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_node_count(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        assert net.node_count == config.n_init

    def test_seed_nodes_flagged(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        seed_count = sum(1 for n in net.nodes.values() if n.is_seed_node)
        assert seed_count == config.n_seed_nodes

    def test_ring_lattice_connectivity(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        for node in net.nodes.values():
            # Each node should have approximately k_neighbors connections
            assert len(node.coupling_weights) >= config.k_neighbors // 2
            assert len(node.coupling_weights) <= config.k_neighbors + 2

    def test_frequency_spread(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        freqs = [n.frequency for n in net.nodes.values()]
        freq_lo, freq_hi = config.freq_range
        assert min(freqs) >= freq_lo
        assert max(freqs) <= freq_hi
        # Should span a reasonable range
        assert max(freqs) / min(freqs) > 5.0

    def test_initial_weights_near_zero(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        for node in net.nodes.values():
            for w in node.coupling_weights.values():
                assert abs(w) < 0.1

    def test_connections_are_symmetric(self, config, sensor, actuator):
        """If A→B exists, B→A should also exist."""
        net = _make_network(config, sensor, actuator)
        for nid, node in net.nodes.items():
            for other_id in node.coupling_weights:
                assert nid in net.nodes[other_id].coupling_weights


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_output(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        output = net.step(np.array([0.5]))
        assert isinstance(output, np.ndarray)
        assert output.shape == (1,)

    def test_activations_bounded(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        for _ in range(50):
            net.step(np.array([0.5]))
        for node in net.nodes.values():
            assert -1.0 <= node.activation <= 1.0

    def test_step_count_increments(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        assert net._step_count == 0
        net.step(np.array([0.0]))
        assert net._step_count == 1
        net.step(np.array([0.0]))
        assert net._step_count == 2

    def test_multi_step_stability(self, config, sensor, actuator):
        """100 steps without NaN or crash."""
        net = _make_network(config, sensor, actuator)
        for _ in range(100):
            output = net.step(np.array([1.0]))
            assert np.all(np.isfinite(output))

    def test_branching_ratios_update(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        for _ in range(200):
            net.step(np.array([1.0]))
        # At least some nodes should have non-zero branching ratio
        ratios = net.get_branching_ratios()
        assert any(r > 0.0 for r in ratios.values())


# ---------------------------------------------------------------------------
# Recruit / Release
# ---------------------------------------------------------------------------

class TestRecruit:
    def test_recruit_adds_node(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        initial_count = net.node_count
        node = net.recruit_node(near_frequency=0.1)
        assert node is not None
        assert net.node_count == initial_count + 1
        assert not node.is_seed_node

    def test_recruit_respects_upper_bound(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        # Fill to max
        while net.node_count < small_config.max_nodes:
            net.recruit_node()
        assert net.node_count == small_config.max_nodes
        # Should return None when at max
        result = net.recruit_node()
        assert result is None
        assert net.node_count == small_config.max_nodes

    def test_recruited_node_has_connections(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        node = net.recruit_node(near_frequency=0.5)
        assert len(node.coupling_weights) > 0

    def test_recruited_connections_symmetric(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        node = net.recruit_node(near_frequency=0.5)
        for other_id in node.coupling_weights:
            assert node.node_id in net.nodes[other_id].coupling_weights


class TestRelease:
    def test_release_removes_node(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        # Find a non-seed node
        non_seed = [nid for nid, n in net.nodes.items() if not n.is_seed_node]
        assert len(non_seed) > 0
        nid = non_seed[0]
        initial_count = net.node_count
        released = net.release_node(nid)
        assert released is True
        assert net.node_count == initial_count - 1
        assert nid not in net.nodes

    def test_release_protects_seed_nodes(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        seed_nodes = [nid for nid, n in net.nodes.items() if n.is_seed_node]
        assert len(seed_nodes) > 0
        released = net.release_node(seed_nodes[0])
        assert released is False

    def test_release_cleans_connections(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        non_seed = [nid for nid, n in net.nodes.items() if not n.is_seed_node]
        nid = non_seed[0]
        # Get neighbors before release
        neighbors = list(net.nodes[nid].coupling_weights.keys())
        net.release_node(nid)
        # Neighbors should no longer reference the released node
        for neighbor_id in neighbors:
            if neighbor_id in net.nodes:
                assert nid not in net.nodes[neighbor_id].coupling_weights

    def test_release_nonexistent_node(self, small_config, sensor, actuator):
        net = _make_network(small_config, sensor, actuator)
        assert net.release_node("nonexistent") is False


# ---------------------------------------------------------------------------
# Introductions (Rule 2b at network level)
# ---------------------------------------------------------------------------

class TestIntroductions:
    def test_introductions_form_connections(self, config, sensor, actuator):
        """Manually trigger introductions and verify they form."""
        net = _make_network(config, sensor, actuator)
        # Find two nodes that are NOT connected
        nodes = list(net.nodes.keys())
        a, b = None, None
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[j] not in net.nodes[nodes[i]].coupling_weights:
                    a, b = nodes[i], nodes[j]
                    break
            if a:
                break

        if a and b:
            formed = net._process_introductions([(a, b)])
            assert formed == 1
            assert b in net.nodes[a].coupling_weights
            assert a in net.nodes[b].coupling_weights


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------

class TestQueryMethods:
    def test_frequency_distribution(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        fd = net.frequency_distribution()
        assert len(fd) == config.n_init
        for freq in fd.values():
            assert isinstance(freq, float)

    def test_get_activations(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        net.step(np.array([1.0]))
        acts = net.get_activations()
        assert len(acts) == config.n_init

    def test_get_branching_ratios(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        ratios = net.get_branching_ratios()
        assert len(ratios) == config.n_init


# ---------------------------------------------------------------------------
# as_observer
# ---------------------------------------------------------------------------

class TestAsObserver:
    def test_returns_observer(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        obs = net.as_observer()
        from ro_framework.observer.observer import Observer
        assert isinstance(obs, Observer)

    def test_internal_dofs_match_nodes(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        obs = net.as_observer()
        assert len(obs.internal_dofs) == config.n_init

    def test_subset_observer(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        subset = set(list(net.nodes.keys())[:4])
        obs = net.as_observer(node_subset=subset)
        assert len(obs.internal_dofs) == 4


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        # Run a few steps
        for _ in range(10):
            net.step(np.array([0.5]))

        d = net.to_dict()
        restored = SeedNetwork.from_dict(d, sensor, actuator, seed=42)

        assert restored.node_count == net.node_count
        assert restored._step_count == net._step_count
        assert restored._next_node_id == net._next_node_id

        # Check node frequencies preserved
        for nid in net.nodes:
            assert nid in restored.nodes
            assert abs(restored.nodes[nid].frequency - net.nodes[nid].frequency) < 1e-10

    def test_round_trip_preserves_connections(self, config, sensor, actuator):
        net = _make_network(config, sensor, actuator)
        for _ in range(5):
            net.step(np.array([0.5]))

        d = net.to_dict()
        restored = SeedNetwork.from_dict(d, sensor, actuator)

        for nid in net.nodes:
            orig = net.nodes[nid].coupling_weights
            rest = restored.nodes[nid].coupling_weights
            assert set(orig.keys()) == set(rest.keys())
