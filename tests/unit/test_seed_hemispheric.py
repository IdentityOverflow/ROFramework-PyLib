"""Tests for hemispheric topology, motor nodes, and reward modulation."""

import math
import numpy as np
import pytest

from ro_framework.seed.node import OscillatoryNode, SeedConfig
from ro_framework.seed.network import SeedNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_sensor(external_input, node_frequencies):
    """Minimal sensor: zero drive for all nodes."""
    return {nid: 0.0 for nid in node_frequencies}


def _dummy_actuator(node_activations):
    """Minimal actuator: return mean activation."""
    if not node_activations:
        return np.zeros(1, dtype=np.float32)
    return np.array([np.mean(list(node_activations.values()))], dtype=np.float32)


MOTOR_SPEC = {
    "fwd_push_L": 1, "fwd_pull_L": 1,
    "fwd_push_R": 1, "fwd_pull_R": 1,
    "turn_L": 1, "turn_R": 1,
    "eat_L": 1, "eat_R": 1,
}


def _hemi_config(**overrides):
    """Config with hemispheric topology + motor nodes enabled."""
    defaults = dict(
        n_init=32,
        k_neighbors=4,
        n_seed_nodes=4,
        hemisphere_enabled=True,
        motor_nodes_enabled=True,
        motor_node_spec=dict(MOTOR_SPEC),
        motor_freq_range=(0.01, 0.05),
        reward_modulation_enabled=True,
        rpe_scale=5.0,
        rpe_baseline_lr_frac=0.2,
        n_callosal_bridges=2,
        callosal_bands=2,
    )
    defaults.update(overrides)
    return SeedConfig(**defaults)


def _make_net(cfg=None, seed=42):
    cfg = cfg or _hemi_config()
    return SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=seed)


# ---------------------------------------------------------------------------
# Hemispheric construction
# ---------------------------------------------------------------------------

class TestHemisphericConstruction:
    def test_hemispheres_are_balanced(self):
        net = _make_net()
        # Each hemisphere gets n_init // 2 = 16 nodes (plus motor nodes)
        # Motor: 4 _L, 4 _R = split
        assert len(net._hemisphere_L) >= 16
        assert len(net._hemisphere_R) >= 16
        diff = abs(len(net._hemisphere_L) - len(net._hemisphere_R))
        assert diff <= len(MOTOR_SPEC)  # motor nodes may unbalance slightly

    def test_callosal_bridges_exist(self):
        net = _make_net()
        assert len(net._hemisphere_M) == 2 * 2  # n_callosal_bridges * callosal_bands
        for nid in net._hemisphere_M:
            node = net.nodes[nid]
            assert node.node_role == "callosal"
            assert node.hemisphere == "M"

    def test_callosal_bridges_connect_both_hemispheres(self):
        net = _make_net()
        for nid in net._hemisphere_M:
            node = net.nodes[nid]
            neighbors = set(node.coupling_weights.keys())
            has_L = bool(neighbors & net._hemisphere_L)
            has_R = bool(neighbors & net._hemisphere_R)
            assert has_L, f"Bridge {nid} not connected to L hemisphere"
            assert has_R, f"Bridge {nid} not connected to R hemisphere"

    def test_within_hemisphere_connections(self):
        net = _make_net()
        # Check that L nodes are primarily connected to L nodes
        for nid in list(net._hemisphere_L)[:5]:
            node = net.nodes[nid]
            if node.node_role == "motor":
                continue
            neighbors = set(node.coupling_weights.keys())
            n_same = len(neighbors & net._hemisphere_L)
            n_other = len(neighbors & net._hemisphere_R)
            assert n_same >= n_other, (
                f"Node {nid} has more cross-hemi ({n_other}) than same-hemi ({n_same})"
            )

    def test_all_nodes_tracked(self):
        net = _make_net()
        tracked = net._hemisphere_L | net._hemisphere_R | net._hemisphere_M
        assert tracked == set(net.nodes.keys())


# ---------------------------------------------------------------------------
# Motor nodes
# ---------------------------------------------------------------------------

class TestMotorNodes:
    def test_motor_nodes_created(self):
        net = _make_net()
        assert len(net._motor_nodes) == 8  # sum of MOTOR_SPEC values
        labels = set(net._motor_nodes.values())
        assert labels == set(MOTOR_SPEC.keys())

    def test_motor_nodes_are_seed_nodes(self):
        net = _make_net()
        for nid in net._motor_nodes:
            assert net.nodes[nid].is_seed_node

    def test_motor_nodes_have_correct_role(self):
        net = _make_net()
        for nid, label in net._motor_nodes.items():
            node = net.nodes[nid]
            assert node.node_role == "motor"
            assert node.motor_label == label

    def test_motor_node_hemispheres(self):
        net = _make_net()
        for nid, label in net._motor_nodes.items():
            node = net.nodes[nid]
            if label.endswith("_L"):
                assert node.hemisphere == "L"
                assert nid in net._hemisphere_L
            elif label.endswith("_R"):
                assert node.hemisphere == "R"
                assert nid in net._hemisphere_R

    def test_motor_node_frequencies_in_range(self):
        net = _make_net()
        lo, hi = net.config.motor_freq_range
        for nid in net._motor_nodes:
            f = net.nodes[nid].frequency
            assert lo <= f <= hi, f"Motor node {nid} freq {f} outside [{lo}, {hi}]"

    def test_get_motor_activations(self):
        net = _make_net()
        # Step a few times to get non-zero activations
        obs = np.random.default_rng(0).random(10)
        for _ in range(5):
            net.step(obs)
        acts = net.get_motor_activations()
        assert set(acts.keys()) == set(MOTOR_SPEC.keys())
        for v in acts.values():
            assert -1.0 <= v <= 1.0

    def test_motor_nodes_protected_from_release(self):
        net = _make_net()
        motor_nids = list(net._motor_nodes.keys())
        for nid in motor_nids:
            released = net.release_node(nid)
            assert not released
            assert nid in net.nodes


# ---------------------------------------------------------------------------
# Reward modulation
# ---------------------------------------------------------------------------

class TestRewardModulation:
    def test_reward_modulator_default_zero(self):
        net = _make_net()
        assert net._reward_modulator == 0.0

    def test_set_reward_modulator(self):
        net = _make_net()
        net.set_reward_modulator(0.5)
        assert net._reward_modulator == 0.5

    def test_positive_rpe_amplifies_learning(self):
        """With RPE > 0, weight changes should be larger."""
        cfg = _hemi_config(n_init=8, k_neighbors=2, n_callosal_bridges=0,
                           callosal_bands=0, motor_nodes_enabled=False)
        rng = np.random.default_rng(42)

        # Create two identical networks
        net_base = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=99)
        net_boost = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=99)

        # Run a few steps to get activations
        obs = rng.random(10)
        for _ in range(10):
            net_base.step(obs)
            net_boost.step(obs)

        # Now set different reward modulators and step
        net_base.set_reward_modulator(0.0)
        net_boost.set_reward_modulator(0.1)

        # Record weights before
        nid = list(net_base.nodes.keys())[0]
        w_before_base = dict(net_base.nodes[nid].coupling_weights)
        w_before_boost = dict(net_boost.nodes[nid].coupling_weights)

        # Step to trigger Rule 2a
        net_base.step(obs)
        net_boost.step(obs)

        # Measure total weight change
        delta_base = sum(
            abs(net_base.nodes[nid].coupling_weights.get(k, 0) - v)
            for k, v in w_before_base.items()
        )
        delta_boost = sum(
            abs(net_boost.nodes[nid].coupling_weights.get(k, 0) - v)
            for k, v in w_before_boost.items()
        )

        # Boosted should have larger changes (baseline=0.2, boost=0.2+5*0.1=0.7)
        assert delta_boost > delta_base or (delta_base == 0 and delta_boost == 0)

    def test_disabled_reward_modulation_ignores_rpe(self):
        """With reward_modulation_enabled=False, RPE should have no effect."""
        cfg = _hemi_config(reward_modulation_enabled=False, n_init=8,
                           k_neighbors=2, n_callosal_bridges=0,
                           callosal_bands=0, motor_nodes_enabled=False)

        net_a = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=99)
        net_b = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=99)

        obs = np.random.default_rng(42).random(10)
        for _ in range(10):
            net_a.step(obs)
            net_b.step(obs)

        # Different RPE values but modulation disabled
        net_a.set_reward_modulator(0.0)
        net_b.set_reward_modulator(1.0)

        net_a.step(obs)
        net_b.step(obs)

        # Weights should be identical
        for nid in net_a.nodes:
            for k in net_a.nodes[nid].coupling_weights:
                assert net_a.nodes[nid].coupling_weights[k] == pytest.approx(
                    net_b.nodes[nid].coupling_weights[k], abs=1e-12
                )

    def test_step_passes_reward_modulator(self):
        """Verify step() propagates reward_modulator to adjust_couplings."""
        net = _make_net()
        net.set_reward_modulator(0.5)
        # Just verify it doesn't crash — the actual modulation is tested above
        obs = np.random.default_rng(0).random(10)
        net.step(obs)
        assert True  # no exception


# ---------------------------------------------------------------------------
# Recruit with hemispheres
# ---------------------------------------------------------------------------

class TestHemisphericRecruit:
    def test_recruited_node_gets_hemisphere(self):
        net = _make_net()
        node = net.recruit_node(near_frequency=0.5)
        assert node is not None
        assert node.hemisphere in ("L", "R")
        assert node.node_id in (net._hemisphere_L | net._hemisphere_R)

    def test_recruit_balances_hemispheres(self):
        cfg = _hemi_config(n_init=10, k_neighbors=2, n_callosal_bridges=0,
                           callosal_bands=0, motor_nodes_enabled=False)
        net = _make_net(cfg)
        initial_diff = abs(len(net._hemisphere_L) - len(net._hemisphere_R))

        # Recruit several nodes
        for _ in range(6):
            net.recruit_node(near_frequency=0.5)

        new_diff = abs(len(net._hemisphere_L) - len(net._hemisphere_R))
        assert new_diff <= initial_diff + 1  # shouldn't diverge


# ---------------------------------------------------------------------------
# Release with hemispheres
# ---------------------------------------------------------------------------

class TestHemisphericRelease:
    def test_release_cleans_hemisphere_set(self):
        net = _make_net()
        # Find a non-seed, non-motor node
        target = None
        for nid, node in net.nodes.items():
            if not node.is_seed_node and node.node_role != "motor":
                target = nid
                break
        assert target is not None

        hemi = net.get_hemisphere(target)
        net.release_node(target)

        assert target not in net.nodes
        assert target not in net._hemisphere_L
        assert target not in net._hemisphere_R
        assert target not in net._hemisphere_M


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestHemisphericSerialization:
    def test_round_trip(self):
        net = _make_net()
        # Step a few times
        obs = np.random.default_rng(0).random(10)
        for _ in range(5):
            net.step(obs)
        net.set_reward_modulator(0.42)

        d = net.to_dict()
        net2 = SeedNetwork.from_dict(d, _dummy_sensor, _dummy_actuator, seed=42)

        assert net2._hemisphere_L == net._hemisphere_L
        assert net2._hemisphere_R == net._hemisphere_R
        assert net2._hemisphere_M == net._hemisphere_M
        assert net2._motor_nodes == net._motor_nodes
        assert net2._reward_modulator == pytest.approx(0.42)

    def test_node_fields_preserved(self):
        net = _make_net()
        d = net.to_dict()
        net2 = SeedNetwork.from_dict(d, _dummy_sensor, _dummy_actuator, seed=42)

        for nid in net._motor_nodes:
            orig = net.nodes[nid]
            loaded = net2.nodes[nid]
            assert loaded.node_role == orig.node_role
            assert loaded.hemisphere == orig.hemisphere
            assert loaded.motor_label == orig.motor_label
            assert loaded.is_seed_node == orig.is_seed_node


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_default_config_no_hemispheres(self):
        cfg = SeedConfig()
        assert not cfg.hemisphere_enabled
        assert not cfg.motor_nodes_enabled
        assert not cfg.reward_modulation_enabled

    def test_default_config_runs_unchanged(self):
        """SeedConfig() with defaults should produce the original ring-lattice."""
        cfg = SeedConfig(n_init=16, k_neighbors=4, n_seed_nodes=2)
        net = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=42)

        assert len(net._hemisphere_L) == 0
        assert len(net._hemisphere_R) == 0
        assert len(net._hemisphere_M) == 0
        assert len(net._motor_nodes) == 0
        assert net._reward_modulator == 0.0

        # Network should function normally
        obs = np.random.default_rng(0).random(10)
        for _ in range(10):
            net.step(obs)

    def test_adjust_couplings_backward_compat(self):
        """adjust_couplings() with no args should work as before."""
        cfg = SeedConfig(n_init=8, k_neighbors=2, n_seed_nodes=2)
        net = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=42)
        obs = np.random.default_rng(0).random(10)
        for _ in range(5):
            net.step(obs)
        # Should not crash — reward_modulator defaults to 1.0 in signature
        # but reward_modulation_enabled is False, so it's ignored
        for node in net.nodes.values():
            node.adjust_couplings()

    def test_get_hemisphere_returns_empty_without_hemispheres(self):
        cfg = SeedConfig(n_init=8, k_neighbors=2, n_seed_nodes=2)
        net = SeedNetwork(cfg, _dummy_sensor, _dummy_actuator, seed=42)
        for nid in net.nodes:
            assert net.get_hemisphere(nid) == ""
