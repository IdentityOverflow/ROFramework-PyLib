"""
SeedNetwork — the collective observer O_seed = (B_seed, M_seed, R_seed, Mem_seed).

The internal DoF set D_internal is the set of activation PolarDoFs contributed
by all currently active nodes. It expands via Rule 4 (recruit) and contracts
via Rule 5 (release).

See docs/seed_architecture.md Section 5 and Section 8.2.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from ro_framework.seed.criticality import fast_mi
from ro_framework.seed.node import OscillatoryNode, SeedConfig


# ---------------------------------------------------------------------------
# Interface protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SensorInterface(Protocol):
    """Maps external input to per-node drive signals.

    Args:
        external_input: Raw environment observation (1-D array).
        node_frequencies: {node_id: frequency} for all active nodes.

    Returns:
        {node_id: drive_value} — external drive for each node.
    """

    def __call__(
        self,
        external_input: np.ndarray,
        node_frequencies: Dict[str, float],
    ) -> Dict[str, float]: ...


@runtime_checkable
class ActuatorInterface(Protocol):
    """Maps node activations to external action.

    Args:
        node_activations: {node_id: activation} for all active nodes.

    Returns:
        Action vector (1-D array).
    """

    def __call__(
        self,
        node_activations: Dict[str, float],
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# SeedNetwork
# ---------------------------------------------------------------------------

class SeedNetwork:
    """The Seed as collective observer.

    Orchestrates all five rules across the node population each timestep.
    Environment-agnostic: sensor and actuator interfaces decouple the
    architecture from any specific environment.
    """

    def __init__(
        self,
        config: SeedConfig,
        sensor: SensorInterface,
        actuator: ActuatorInterface,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.sensor = sensor
        self.actuator = actuator
        self.rng = np.random.default_rng(seed)
        self._step_count: int = 0
        self._next_node_id: int = 0

        # MI estimation buffers (for Rules 4/5)
        self._mi_buffer_ext: deque = deque(maxlen=config.recruit_window)
        self._mi_buffer_int: deque = deque(maxlen=config.recruit_window)

        # Build initial network
        self.nodes: Dict[str, OscillatoryNode] = {}
        self._build_initial_network()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _new_node_id(self) -> str:
        nid = f"n{self._next_node_id}"
        self._next_node_id += 1
        return nid

    def _build_initial_network(self) -> None:
        """Create N_init nodes with log-uniform frequency, ring-lattice topology."""
        cfg = self.config
        freq_lo, freq_hi = cfg.freq_range

        # Log-uniform frequency distribution
        log_freqs = np.linspace(
            np.log(freq_lo), np.log(freq_hi), cfg.n_init
        )
        # Add small jitter for diversity
        log_freqs += self.rng.standard_normal(cfg.n_init) * 0.05
        frequencies = np.exp(log_freqs)
        frequencies = np.clip(frequencies, freq_lo, freq_hi)

        # Sort by frequency for ring-lattice construction
        sorted_indices = np.argsort(frequencies)

        # Create nodes
        node_ids: List[str] = []
        for i in sorted_indices:
            nid = self._new_node_id()
            node_ids.append(nid)
            is_seed = len(node_ids) <= cfg.n_seed_nodes
            self.nodes[nid] = OscillatoryNode(
                node_id=nid,
                frequency=float(frequencies[i]),
                phase=float(self.rng.uniform(0, 2 * math.pi)),
                is_seed_node=is_seed,
                _config=cfg,
            )

        # Ring-lattice: connect each node to k nearest frequency neighbors
        n = len(node_ids)
        k = cfg.k_neighbors
        for i, nid in enumerate(node_ids):
            node = self.nodes[nid]
            for offset in range(1, k // 2 + 1):
                # Connect to neighbors on both sides in frequency-sorted ring
                for j in [i - offset, i + offset]:
                    j = j % n
                    other_id = node_ids[j]
                    if other_id != nid:
                        w = float(self.rng.uniform(-0.01, 0.01))
                        node.form_connection(other_id, initial_weight=w)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, external_input: np.ndarray) -> np.ndarray:
        """Execute one timestep of all five rules.

        Args:
            external_input: Raw environment observation.

        Returns:
            Action vector from actuator interface.
        """
        cfg = self.config

        # 1. Sensor: external → per-node drives
        node_freqs = {nid: n.frequency for nid, n in self.nodes.items()}
        drives = self.sensor(external_input, node_freqs)

        # 2. Gather neighborhood activations
        neighborhoods: Dict[str, Dict[str, float]] = {}
        for nid, node in self.nodes.items():
            neighborhoods[nid] = {
                other_id: self.nodes[other_id].activation
                for other_id in node.coupling_weights
                if other_id in self.nodes
            }

        # 3. Each node: step() — Rules 1 + 3
        for nid, node in self.nodes.items():
            drive = drives.get(nid, 0.0)
            node.step(neighborhoods[nid], drive, self.rng)

        # 4. Each node: adjust_couplings() — Rule 2a
        all_pruned: List[str] = []
        for node in self.nodes.values():
            pruned = node.adjust_couplings()
            all_pruned.extend(pruned)

        # 5. Each node: propose_introductions() — Rule 2b
        all_proposals: List[Tuple[str, str]] = []
        for node in self.nodes.values():
            proposals = node.propose_introductions()
            all_proposals.extend(proposals)

        # Filter and form new connections
        self._process_introductions(all_proposals)

        # 6. Frequency entrainment
        for nid, node in self.nodes.items():
            neighbor_freqs = {
                other_id: self.nodes[other_id].frequency
                for other_id in node.coupling_weights
                if other_id in self.nodes
            }
            node.update_frequency(neighbor_freqs)

        # 7. Rules 4/5 — recruit/release (periodic, not every step)
        self._update_mi_buffers(external_input)
        check_interval = max(1, cfg.recruit_window // 10)
        if self._step_count > 0 and self._step_count % check_interval == 0:
            self._check_growth()
            self._check_pruning()

        # 8. Actuator: node activations → output
        node_acts = {nid: n.activation for nid, n in self.nodes.items()}
        output = self.actuator(node_acts)

        self._step_count += 1
        return np.asarray(output)

    # ------------------------------------------------------------------
    # Rule 2b: process introductions
    # ------------------------------------------------------------------

    def _process_introductions(
        self, proposals: List[Tuple[str, str]]
    ) -> int:
        """Form new connections from introduction proposals.

        Filters out already-connected pairs and self-connections.
        Returns number of new connections formed.
        """
        formed = 0
        seen = set()
        for a, b in proposals:
            if a == b or a not in self.nodes or b not in self.nodes:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)

            # Only form if not already connected
            node_a = self.nodes[a]
            node_b = self.nodes[b]
            if b not in node_a.coupling_weights:
                w = float(self.rng.uniform(-0.001, 0.001))
                node_a.form_connection(b, initial_weight=w)
                node_b.form_connection(a, initial_weight=w)
                formed += 1

        return formed

    # ------------------------------------------------------------------
    # MI estimation for Rules 4/5
    # ------------------------------------------------------------------

    def _update_mi_buffers(self, external_input: np.ndarray) -> None:
        """Buffer external input and internal activations for MI estimation."""
        ext = np.asarray(external_input, dtype=np.float64).ravel()
        # Internal: vector of all node activations (sorted by id for consistency)
        int_vec = np.array([
            self.nodes[nid].activation
            for nid in sorted(self.nodes.keys())
        ], dtype=np.float64)

        self._mi_buffer_ext.append(ext)
        self._mi_buffer_int.append(int_vec)

    def _estimate_external_mi(self) -> float:
        """Estimate MI between external inputs and internal activations.

        Uses the mean across external dimensions for a scalar MI estimate.
        """
        if len(self._mi_buffer_ext) < 20:
            return float("inf")  # not enough data yet

        ext_mat = np.array(list(self._mi_buffer_ext))
        int_mat = np.array(list(self._mi_buffer_int))

        # Average MI across external dimensions
        n_ext = ext_mat.shape[1] if ext_mat.ndim > 1 else 1
        n_int = int_mat.shape[1] if int_mat.ndim > 1 else 1

        total_mi = 0.0
        count = 0
        for i in range(min(n_ext, 5)):  # cap at 5 dims for speed
            ext_col = ext_mat[:, i] if ext_mat.ndim > 1 else ext_mat
            for j in range(min(n_int, 10)):  # cap at 10 internal dims
                int_col = int_mat[:, j] if int_mat.ndim > 1 else int_mat
                total_mi += fast_mi(ext_col, int_col, bins=8)
                count += 1

        return total_mi / max(count, 1)

    def _estimate_node_mi(self, node_id: str) -> float:
        """Estimate MI between a node's activations and the external input.

        This measures whether the node carries information about the
        environment, not just whether it correlates with neighbors
        (which all coupled nodes do). A node that doesn't track external
        signal is a candidate for release.
        """
        node = self.nodes[node_id]
        hist = np.array(list(node.activation_history), dtype=np.float64)

        if len(hist) < 20 or len(self._mi_buffer_ext) < 20:
            return float("inf")  # not enough data

        ext_mat = np.array(list(self._mi_buffer_ext))
        buf_len = min(len(hist), len(ext_mat))
        if buf_len < 10:
            return float("inf")

        total_mi = 0.0
        count = 0
        n_ext = ext_mat.shape[1] if ext_mat.ndim > 1 else 1
        for i in range(min(n_ext, 5)):
            ext_col = ext_mat[-buf_len:, i] if ext_mat.ndim > 1 else ext_mat[-buf_len:]
            total_mi += fast_mi(hist[-buf_len:], ext_col)
            count += 1

        return total_mi / max(count, 1)

    # ------------------------------------------------------------------
    # Rule 4: recruit
    # ------------------------------------------------------------------

    def _check_growth(self) -> bool:
        """Rule 4: recruit when the network is supercritical.

        Sustained σ > 1 means too much propagation — the network is
        overloaded and needs more nodes to distribute activity.
        Only recruits when there IS activity (prevents growth during silence).
        """
        cfg = self.config
        if len(self.nodes) >= cfg.max_nodes:
            return False

        sigmas = [n.branching_ratio for n in self.nodes.values()]
        mean_sigma = float(np.mean(sigmas))
        n_active = sum(
            1 for n in self.nodes.values()
            if abs(n.activation) > cfg.activation_threshold
        )

        # Need meaningful activity AND supercritical regime
        if n_active < max(3, len(self.nodes) // 5):
            return False
        if mean_sigma <= 1.2:  # small buffer above 1 to avoid jitter-driven growth
            return False

        self.recruit_node()
        return True

    def recruit_node(
        self, near_frequency: Optional[float] = None
    ) -> Optional[OscillatoryNode]:
        """Add a new node to the network (Rule 4 action).

        Args:
            near_frequency: Target frequency. If None, picks an
                underrepresented frequency range.

        Returns:
            The new node, or None if at upper bound.
        """
        cfg = self.config
        if len(self.nodes) >= cfg.max_nodes:
            return None

        if near_frequency is None:
            near_frequency = self._find_underrepresented_frequency()

        nid = self._new_node_id()
        node = OscillatoryNode(
            node_id=nid,
            frequency=near_frequency,
            phase=float(self.rng.uniform(0, 2 * math.pi)),
            is_seed_node=False,
            _config=cfg,
        )

        self.nodes[nid] = node
        self._mi_buffer_int.clear()  # node count changed

        # Connect to k nearest by frequency
        freq_distances = [
            (other_id, abs(other.frequency - near_frequency))
            for other_id, other in self.nodes.items()
            if other_id != nid
        ]
        freq_distances.sort(key=lambda x: x[1])
        k_freq = min(cfg.k_neighbors, len(freq_distances))

        for other_id, _ in freq_distances[:k_freq]:
            w = float(self.rng.uniform(-0.001, 0.001))
            node.form_connection(other_id, initial_weight=w)
            self.nodes[other_id].form_connection(nid, initial_weight=w)

        # Also connect to m most active nodes (m ~ k_neighbors // 2)
        m = max(1, cfg.k_neighbors // 2)
        activity_ranked = sorted(
            ((oid, abs(o.activation)) for oid, o in self.nodes.items() if oid != nid),
            key=lambda x: x[1],
            reverse=True,
        )
        for other_id, _ in activity_ranked[:m]:
            if other_id not in node.coupling_weights:
                w = float(self.rng.uniform(-0.001, 0.001))
                node.form_connection(other_id, initial_weight=w)
                self.nodes[other_id].form_connection(nid, initial_weight=w)

        return node

    def _find_underrepresented_frequency(self) -> float:
        """Find frequency range with fewest nodes."""
        cfg = self.config
        freq_lo, freq_hi = cfg.freq_range
        freqs = np.array([n.frequency for n in self.nodes.values()])

        # Divide frequency range into bins (log-space)
        n_bins = max(5, len(self.nodes) // 10)
        bin_edges = np.exp(np.linspace(np.log(freq_lo), np.log(freq_hi), n_bins + 1))
        counts, _ = np.histogram(freqs, bins=bin_edges)

        # Pick bin with fewest nodes
        min_bin = int(np.argmin(counts))
        # Random frequency within that bin
        return float(self.rng.uniform(bin_edges[min_bin], bin_edges[min_bin + 1]))

    # ------------------------------------------------------------------
    # Rule 5: release
    # ------------------------------------------------------------------

    def _check_pruning(self) -> List[str]:
        """Rule 5: release nodes that are persistently inactive while others are active.

        A node that hasn't fired recently while the network is active is
        not contributing to the collective representation.
        """
        cfg = self.config

        # Don't release during silence — nothing to judge contribution against
        n_active = sum(
            1 for n in self.nodes.values()
            if abs(n.activation) > cfg.activation_threshold
        )
        if n_active < max(3, len(self.nodes) // 5):
            return []

        released = []
        for nid in list(self.nodes.keys()):
            node = self.nodes[nid]
            if node.is_seed_node:
                continue
            if len(node.activation_history) < cfg.release_window:
                continue

            # Check recent activity: fraction of time active in release window
            recent = list(node.activation_history)[-cfg.release_window:]
            n_active_steps = sum(
                1 for a in recent if abs(a) > cfg.activation_threshold
            )
            active_frac = n_active_steps / len(recent)

            # Release if consistently silent (< 1% active)
            if active_frac < 0.01:
                self.release_node(nid)
                released.append(nid)

        return released

    def release_node(self, node_id: str) -> bool:
        """Remove a node from the network (Rule 5 action).

        Args:
            node_id: Node to release.

        Returns:
            True if released, False if protected or not found.
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        if node.is_seed_node:
            return False

        # Count non-seed nodes to check lower bound
        non_seed = sum(1 for n in self.nodes.values() if not n.is_seed_node)
        if non_seed <= 1:
            return False

        # Remove from all neighbors' coupling weights
        for other_id in list(node.coupling_weights.keys()):
            if other_id in self.nodes:
                self.nodes[other_id].remove_connection(node_id)

        del self.nodes[node_id]
        self._mi_buffer_int.clear()  # node count changed
        return True

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def frequency_distribution(self) -> Dict[str, float]:
        """Current {node_id: frequency} for all nodes."""
        return {nid: n.frequency for nid, n in self.nodes.items()}

    def get_activations(self) -> Dict[str, float]:
        """Current {node_id: activation} for all nodes."""
        return {nid: n.activation for nid, n in self.nodes.items()}

    def get_branching_ratios(self) -> Dict[str, float]:
        """Current {node_id: branching_ratio} for all nodes."""
        return {nid: n.branching_ratio for nid, n in self.nodes.items()}

    # ------------------------------------------------------------------
    # as_observer — bridge to RO Framework library
    # ------------------------------------------------------------------

    def as_observer(
        self, node_subset: Optional[set] = None
    ) -> "Observer":
        """Create an Observer wrapping this network.

        The Observer's internal DoFs are one PolarDoF per node (bounded [-1,1]).
        The world_model wraps network.step().

        Args:
            node_subset: Optional set of node_ids. If provided, only those
                nodes contribute internal DoFs.

        Returns:
            An Observer compatible with ConsciousnessEvaluator and
            KnowledgeAssessment.
        """
        from ro_framework.core.dof import PolarDoF, PolarDoFType
        from ro_framework.observer.observer import Observer

        nodes_to_use = (
            {nid: self.nodes[nid] for nid in node_subset if nid in self.nodes}
            if node_subset
            else self.nodes
        )

        internal_dofs = [
            PolarDoF(
                name=f"act_{nid}",
                pole_negative=-1.0,
                pole_positive=1.0,
                polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
            )
            for nid in sorted(nodes_to_use.keys())
        ]

        # External DoFs: determined by sensor interface dimension
        # Use a single PolarDoF as placeholder — the user can override
        external_dofs = [
            PolarDoF(
                name="ext_0",
                pole_negative=-np.inf,
                pole_positive=np.inf,
                polar_type=PolarDoFType.CONTINUOUS_REAL,
            )
        ]

        mapping = _SeedMapping(self, sorted(nodes_to_use.keys()))

        return Observer(
            name="seed_observer",
            internal_dofs=internal_dofs,
            external_dofs=external_dofs,
            world_model=mapping,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network state. Sensor/actuator are NOT included."""
        return {
            "config": self.config.to_dict(),
            "step_count": self._step_count,
            "next_node_id": self._next_node_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        sensor: SensorInterface,
        actuator: ActuatorInterface,
        seed: int = 0,
    ) -> "SeedNetwork":
        """Reconstruct network from serialized dict + interfaces."""
        config = SeedConfig.from_dict(d["config"])

        # Create instance without building initial network
        net = cls.__new__(cls)
        net.config = config
        net.sensor = sensor
        net.actuator = actuator
        net.rng = np.random.default_rng(seed)
        net._step_count = d.get("step_count", 0)
        net._next_node_id = d.get("next_node_id", 0)
        net._mi_buffer_ext = deque(maxlen=config.recruit_window)
        net._mi_buffer_int = deque(maxlen=config.recruit_window)

        # Reconstruct nodes
        net.nodes = {}
        for nid, node_d in d.get("nodes", {}).items():
            net.nodes[nid] = OscillatoryNode.from_dict(node_d, config)

        return net


# ---------------------------------------------------------------------------
# Internal mapping for as_observer()
# ---------------------------------------------------------------------------

class _SeedMapping:
    """Wraps SeedNetwork.step() as a callable for Observer.world_model."""

    def __init__(self, network: SeedNetwork, node_ids: List[str]) -> None:
        self.network = network
        self.node_ids = node_ids

    def __call__(self, state: "State") -> "State":
        from ro_framework.core.state import State

        # Extract external input from state
        ext_vec = state.to_vector(state.dofs)
        output = self.network.step(ext_vec)

        # Build internal state from node activations
        from ro_framework.core.dof import PolarDoF, PolarDoFType
        from ro_framework.core.value import Value

        values = {}
        for nid in self.node_ids:
            if nid in self.network.nodes:
                dof = PolarDoF(
                    name=f"act_{nid}",
                    pole_negative=-1.0,
                    pole_positive=1.0,
                    polar_type=PolarDoFType.CONTINUOUS_BOUNDED,
                )
                values[dof] = Value(dof=dof, raw_value=self.network.nodes[nid].activation)

        return State(values=values)
