"""
OscillatoryNode — the unit observer primitive of the Seed architecture.

Each node is simultaneously:
- A DoF contributor: one activation value to the collective state space
- A unit observer: O_i = (B_i, M_i, R_i, Mem_i)

Minimal implementation: activation dynamics + Rule 2a (Hebbian weight
adjustment governed by branching ratio). Other rules (2b, 4, 5) can be
layered on once the core self-regulation is validated.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SeedConfig:
    """All tunable parameters for the Seed architecture.

    Defaults target a SPARSE activation regime: threshold is high enough
    that nodes need coupling input to fire, not just their own drive.
    """

    # Network size
    n_init: int = 64
    k_neighbors: int = 6
    n_seed_nodes: int = 8
    max_nodes: int = 16384

    # Node dynamics — sparse regime
    # drive_amplitude (0.2) + noise_floor (0.05) << threshold (0.5)
    # so nodes need coupling input to cross threshold
    noise_floor: float = 0.05
    drive_amplitude: float = 0.2
    activation_threshold: float = 0.5
    freq_range: Tuple[float, float] = (0.01, 1.0)
    n_cycles_memory: int = 80
    dt: float = 1.0

    # Rule 2a — weight adjustment
    learning_rate: float = 0.01
    w_max: float = 2.0
    prune_weight_threshold: float = 0.005
    prune_weight_window: int = 200

    # Rule 2b — connection formation (disabled by default for minimal mode)
    connect_threshold: float = 0.3
    connect_window: int = 50

    # Rule 4 — recruit
    recruit_mi_threshold: float = 0.1
    recruit_window: int = 200

    # Rule 5 — release
    release_mi_threshold: float = 0.05
    release_window: int = 200

    # Frequency entrainment
    freq_learning_rate: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["freq_range"] = list(d["freq_range"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeedConfig":
        d = dict(d)
        if "freq_range" in d:
            d["freq_range"] = tuple(d["freq_range"])
        return cls(**d)


# ---------------------------------------------------------------------------
# OscillatoryNode
# ---------------------------------------------------------------------------

@dataclass
class OscillatoryNode:
    """The irreducible unit observer of the Seed.

    Activation dynamics (per timestep):
        input      = Σ_j w_ij * activation_j  +  external_drive
        drive      = A * sin(phase)
        activation = tanh(input + drive + noise_floor * N(0,1))
        phase     += 2π * frequency * dt

    In the sparse regime (threshold >> drive + noise), nodes are mostly
    silent. Activation above threshold requires coupling input from
    neighbors, making the branching ratio a meaningful causal measure.

    Branching ratio σ_i (EMA):
        σ = mean(number of neighbors activating at t+1 | this node active at t)
        Target: σ = 1 (criticality)
    """

    node_id: str
    frequency: float
    phase: float = 0.0
    activation: float = 0.0
    is_seed_node: bool = False

    # Coupling weights — keyed by neighbor node_id
    coupling_weights: Dict[str, float] = field(default_factory=dict)

    # Config-derived (set in __post_init__ or from_dict)
    noise_floor: float = 0.05
    drive_amplitude: float = 0.2

    # Running statistics
    branching_ratio: float = 0.0
    _step_count: int = field(default=0, repr=False)

    # Internals
    activation_history: deque = field(default=None, repr=False)
    neighbor_coactivation: Dict[Tuple[str, str], int] = field(
        default_factory=dict, repr=False
    )
    _last_neighborhood: Dict[str, float] = field(
        default_factory=dict, repr=False
    )
    _low_weight_counts: Dict[str, int] = field(
        default_factory=dict, repr=False
    )
    _config: Optional[SeedConfig] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._config is not None:
            self.noise_floor = self._config.noise_floor
            self.drive_amplitude = self._config.drive_amplitude
            self.frequency = float(np.clip(
                self.frequency,
                self._config.freq_range[0],
                self._config.freq_range[1],
            ))
        if self.activation_history is None:
            self.activation_history = deque(maxlen=self.memory_window)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def memory_window(self) -> int:
        """Effective memory window in timesteps (cycle-proportional)."""
        cfg = self._config
        n_cycles = cfg.n_cycles_memory if cfg else 80
        freq_min = cfg.freq_range[0] if cfg else 0.01
        freq = max(self.frequency, freq_min)
        period = 1.0 / freq
        return max(10, int(round(n_cycles * period)))

    # ------------------------------------------------------------------
    # Step: activation dynamics + branching ratio tracking
    # ------------------------------------------------------------------

    def step(
        self,
        neighborhood_activations: Dict[str, float],
        external_drive: float,
        rng: np.random.Generator,
    ) -> float:
        """Compute next activation and update branching ratio.

        Args:
            neighborhood_activations: {neighbor_id: activation} for
                connected neighbors.
            external_drive: sensor input for this node.
            rng: numpy random generator.

        Returns:
            New activation value in [-1, 1].
        """
        cfg = self._config
        threshold = cfg.activation_threshold if cfg else 0.5
        dt = cfg.dt if cfg else 1.0

        # Weighted sum of neighbor activations
        inp = sum(
            self.coupling_weights.get(nid, 0.0) * act
            for nid, act in neighborhood_activations.items()
        )
        inp += external_drive

        # Intrinsic oscillatory drive
        drive = self.drive_amplitude * math.sin(self.phase)

        # Irreducible noise
        noise = self.noise_floor * rng.standard_normal()

        # Was this node active before this step?
        was_active = abs(self.activation) > threshold

        # New activation
        self.activation = float(np.tanh(inp + drive + noise))

        # Advance phase
        self.phase = (self.phase + 2.0 * math.pi * self.frequency * dt) % (
            2.0 * math.pi
        )

        # Record activation
        self.activation_history.append(self.activation)

        # --- Update branching ratio (EMA) ---
        # Standard definition: count neighbors active at t+1 when
        # this node was active at t. In a sparse regime, this
        # approximates causal propagation.
        window = min(self.memory_window, 200)
        alpha = 2.0 / (window + 1)

        if was_active and neighborhood_activations:
            n_active_neighbors = sum(
                1 for act in neighborhood_activations.values()
                if abs(act) > threshold
            )
            self.branching_ratio = (
                alpha * n_active_neighbors
                + (1.0 - alpha) * self.branching_ratio
            )
        else:
            # Not active: decay σ toward 0 (no propagation happening)
            self.branching_ratio *= (1.0 - alpha)

        # --- Track neighbor co-activation (for Rule 2b when enabled) ---
        active_neighbors = [
            nid for nid, act in neighborhood_activations.items()
            if abs(act) > threshold
        ]
        for i in range(len(active_neighbors)):
            for j in range(i + 1, len(active_neighbors)):
                a, b = active_neighbors[i], active_neighbors[j]
                key = (min(a, b), max(a, b))
                self.neighbor_coactivation[key] = (
                    self.neighbor_coactivation.get(key, 0) + 1
                )

        # Cache for adjust_couplings()
        self._last_neighborhood = dict(neighborhood_activations)
        self._step_count += 1

        return self.activation

    # ------------------------------------------------------------------
    # Rule 2a: adjust weights
    # ------------------------------------------------------------------

    def adjust_couplings(self) -> List[str]:
        """Hebbian adjustment governed by branching ratio error.

        Δw_ij = lr * act_i * act_j * (1 - σ_i)

        When σ < 1 (subcritical): co-active pairs strengthen
        When σ > 1 (supercritical): co-active pairs weaken
        When σ ≈ 1: near-zero adjustment (at criticality)

        This is the ONLY weight update rule. In a sparse regime where
        activation requires coupling input, this single rule is
        sufficient for self-regulation.

        Returns:
            List of pruned neighbor node_ids.
        """
        cfg = self._config
        lr = cfg.learning_rate if cfg else 0.01
        w_max = cfg.w_max if cfg else 2.0
        prune_thresh = cfg.prune_weight_threshold if cfg else 0.005
        prune_window = cfg.prune_weight_window if cfg else 200

        pruned: List[str] = []
        sigma_error = 1.0 - self.branching_ratio

        for nid in list(self.coupling_weights.keys()):
            neighbor_act = self._last_neighborhood.get(nid, 0.0)
            hebbian = self.activation * neighbor_act
            delta = lr * hebbian * sigma_error

            self.coupling_weights[nid] = float(np.clip(
                self.coupling_weights[nid] + delta, -w_max, w_max
            ))

            # Track persistently weak connections for pruning
            if abs(self.coupling_weights[nid]) < prune_thresh:
                self._low_weight_counts[nid] = (
                    self._low_weight_counts.get(nid, 0) + 1
                )
                if self._low_weight_counts[nid] >= prune_window:
                    pruned.append(nid)
            else:
                self._low_weight_counts[nid] = 0

        for nid in pruned:
            self._remove_connection_internal(nid)

        return pruned

    # ------------------------------------------------------------------
    # Rule 2b: propose introductions
    # ------------------------------------------------------------------

    def propose_introductions(self) -> List[Tuple[str, str]]:
        """Identify neighbor pairs with persistent co-activation.

        Returns:
            List of (node_id_a, node_id_b) pairs to introduce.
        """
        cfg = self._config
        threshold = cfg.connect_threshold if cfg else 0.3
        min_steps = cfg.connect_window if cfg else 50

        if self._step_count < min_steps:
            return []

        proposals: List[Tuple[str, str]] = []
        for (a, b), count in self.neighbor_coactivation.items():
            rate = count / self._step_count
            if rate > threshold:
                proposals.append((a, b))

        return proposals

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def form_connection(
        self, other_id: str, initial_weight: float = 0.001
    ) -> None:
        """Form a new connection at near-zero weight."""
        if other_id not in self.coupling_weights:
            self.coupling_weights[other_id] = initial_weight
            self._low_weight_counts[other_id] = 0

    def remove_connection(self, other_id: str) -> None:
        """Remove a connection and clean up tracking state."""
        self._remove_connection_internal(other_id)

    def _remove_connection_internal(self, other_id: str) -> None:
        self.coupling_weights.pop(other_id, None)
        self._low_weight_counts.pop(other_id, None)
        to_remove = [
            key for key in self.neighbor_coactivation
            if other_id in key
        ]
        for key in to_remove:
            del self.neighbor_coactivation[key]

    # ------------------------------------------------------------------
    # Frequency entrainment
    # ------------------------------------------------------------------

    def update_frequency(
        self, neighbor_frequencies: Dict[str, float]
    ) -> None:
        """Drift frequency toward coupling-weighted neighborhood mean."""
        if not neighbor_frequencies:
            return

        cfg = self._config
        eta = cfg.freq_learning_rate if cfg else 0.001
        freq_min = cfg.freq_range[0] if cfg else 0.01
        freq_max = cfg.freq_range[1] if cfg else 1.0

        total_weight = sum(
            abs(self.coupling_weights.get(nid, 0.0))
            for nid in neighbor_frequencies
        )
        if total_weight < 1e-10:
            return

        weighted_freq = sum(
            abs(self.coupling_weights.get(nid, 0.0)) * f
            for nid, f in neighbor_frequencies.items()
        ) / total_weight

        self.frequency += eta * (weighted_freq - self.frequency)
        self.frequency = float(np.clip(self.frequency, freq_min, freq_max))

        # Resize history to match new memory window
        new_maxlen = self.memory_window
        if self.activation_history.maxlen != new_maxlen:
            old_data = list(self.activation_history)
            self.activation_history = deque(old_data, maxlen=new_maxlen)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state. Config is NOT included."""
        return {
            "node_id": self.node_id,
            "frequency": self.frequency,
            "phase": self.phase,
            "activation": self.activation,
            "is_seed_node": self.is_seed_node,
            "coupling_weights": dict(self.coupling_weights),
            "noise_floor": self.noise_floor,
            "drive_amplitude": self.drive_amplitude,
            "branching_ratio": self.branching_ratio,
            "_step_count": self._step_count,
            "activation_history": list(self.activation_history),
            "neighbor_coactivation": {
                f"{a},{b}": count
                for (a, b), count in self.neighbor_coactivation.items()
            },
            "_low_weight_counts": dict(self._low_weight_counts),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], config: SeedConfig
    ) -> "OscillatoryNode":
        """Reconstruct node from serialized dict + config."""
        coact = {}
        for key_str, count in d.get("neighbor_coactivation", {}).items():
            parts = key_str.split(",", 1)
            coact[(parts[0], parts[1])] = count

        history_data = d.get("activation_history", [])

        node = cls(
            node_id=d["node_id"],
            frequency=d["frequency"],
            phase=d.get("phase", 0.0),
            activation=d.get("activation", 0.0),
            is_seed_node=d.get("is_seed_node", False),
            coupling_weights=dict(d.get("coupling_weights", {})),
            noise_floor=d.get("noise_floor", config.noise_floor),
            drive_amplitude=d.get("drive_amplitude", config.drive_amplitude),
            branching_ratio=d.get("branching_ratio", 0.0),
            _step_count=d.get("_step_count", 0),
            activation_history=None,
            neighbor_coactivation=coact,
            _low_weight_counts=dict(d.get("_low_weight_counts", {})),
            _config=config,
        )
        node.activation_history = deque(history_data, maxlen=node.memory_window)
        return node
