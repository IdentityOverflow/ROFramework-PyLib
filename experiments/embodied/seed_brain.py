"""
experiments/embodied/seed_brain.py — Self-Organizing Seed Brain

Uses a SeedNetwork with hemispheric topology, motor nodes inside the network,
and reward-modulated Hebbian learning (Rule 2a).  No external readout matrix.

Architecture
------------
Sensor:    HemisphericSensor — cross-lateralized vision, cross-wired tactile,
           meters to midline callosal nodes, motor nodes get NO direct input
Reservoir: SeedNetwork — self-organizing criticality, frequency entrainment,
           adaptive growth (Rules 1-5 from seed_architecture.md)
Motor:     MotorNodeActuator — push/pull population coding from motor nodes
           embedded inside the network
Learning:  RPE modulates Rule 2a Hebbian learning rate globally.
           No separate readout training — the network IS the controller.

Key differences from ESN (brain.py) / WIM (wim_brain.py)
---------------------------------------------------------
  No W_out matrix:   motor nodes sit inside the network
  No eligibility traces: RPE modulates Hebbian learning (three-factor rule)
  Hemispheric topology: cross-lateralized vision produces tracking reflexes
  Self-organizing: sigma -> 1 emerges, frequency bands form, network grows/shrinks

Usage
-----
    python seed_brain.py                        # connect to game.py --connect
    python seed_brain.py --headless 3600        # offline headless run
    python seed_brain.py --config brains/configs/seed-64.json
    python seed_brain.py --no-learn --load brains/Seed-64.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from typing import Dict, List, Optional, Set

import numpy as np

# -- Resolve library path -----------------------------------------------------
_here     = os.path.dirname(os.path.abspath(__file__))
_repo     = os.path.normpath(os.path.join(_here, "..", ".."))
_src_path = os.path.join(_repo, "src")
for _p in (_src_path, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ro_framework.core.state import State                      # noqa: E402
from ro_framework.observer.observer import Observer, ObservationPair  # noqa: E402
from ro_framework.knowledge.tracker import KnowledgeTracker   # noqa: E402
from ro_framework.seed.network import SeedNetwork              # noqa: E402
from ro_framework.seed.node import SeedConfig                  # noqa: E402

import dofs                                                    # noqa: E402

# Reuse generic infrastructure from brain.py
import brain as _brain_mod                                     # noqa: E402
from brain import (                                            # noqa: E402
    run_connected, run_headless,
    _log_step as _log_step_base,
    _log as _log_base, _log_knowledge, _compute_extra_k,
    _open_log as _open_log_base, _format_k,
    load_config, save_config, _select_device,
    _config_path, _resolve_paths,
    OBS_DIM, _LIFE_IDX,
    LOG_EVERY, SAVE_EVERY, LOG_CAPACITY, ASSESS_EVERY,
    EAT_THRESHOLD, CRITIC_LR,
    _IdentityMapping, _EmptyReservoir,
)

# -- Motor node spec (matches test_seed_hemispheric.py) ------------------------

MOTOR_SPEC = {
    "fwd_push_L": 1, "fwd_pull_L": 1,
    "fwd_push_R": 1, "fwd_pull_R": 1,
    "turn_L": 1, "turn_R": 1,
    "eat_L": 1, "eat_R": 1,
}

# -- Seed-specific defaults ----------------------------------------------------

SEED_DEFAULT_CONFIG: dict = {
    # Identity / paths
    "name":             "",
    "brain_path":       "",
    "log_path":         "",
    "world_config":     "",
    # Seed network
    "n_init":           64,
    "k_neighbors":      6,
    "n_seed_nodes":     8,
    "max_nodes":        16384,
    "noise_floor":      0.05,
    "drive_amplitude":  0.2,
    "activation_threshold": 0.5,
    "freq_range":       [0.01, 1.0],
    "learning_rate":    0.01,
    "w_max":            2.0,
    "freq_learning_rate": 0.001,
    # Hemispheric topology
    "hemisphere_enabled": True,
    "n_callosal_bridges": 4,
    "callosal_bands":   3,
    # Motor nodes
    "motor_nodes_enabled": True,
    "motor_node_spec":  dict(MOTOR_SPEC),
    "motor_freq_range": [0.01, 0.05],
    # Reward modulation
    "reward_modulation_enabled": True,
    "rpe_scale":        5.0,
    "rpe_baseline_lr_frac": 0.2,
    # Sensor
    "sensor_bandwidth": 0.8,
    "input_scale":      1.0,
    # Learning
    "eat_threshold":    EAT_THRESHOLD,
    "critic_lr":        CRITIC_LR,
    # RO Framework
    "assess_every":     ASSESS_EVERY,
    "log_capacity":     LOG_CAPACITY,
    # Arch
    "seed":             42,
    "action_feedback":  False,
    "decision_interval": 1,
    "device":           "cpu",
}

# -- Observation layout constants ----------------------------------------------
# obs[0:242]   = 121 rays x [type_norm, proximity]
# obs[242:258] = 16 body tactile
# obs[258:260] = 2 prong tactile
# obs[260:263] = [life, satiation_norm, valence_norm]

_VIS_START  = 0
_VIS_END    = 242    # 121 rays * 2 channels
_N_RAYS     = 121
_RAY_MID    = 60     # center ray index
_TAC_START  = 242
_TAC_BODY   = 258    # 16 body tactile
_TAC_END    = 260    # + 2 prong
_METER_START = 260
_METER_END  = 263


# -- HemisphericSensor --------------------------------------------------------

class HemisphericSensor:
    """Cross-lateralized sensor mapping for hemispheric SeedNetwork.

    Vision (121 rays):
      - Rays 0-59 (left visual field)   -> RIGHT hemisphere high-freq nodes
      - Ray 60 (center)                 -> split between hemispheres
      - Rays 61-120 (right visual field) -> LEFT hemisphere high-freq nodes

    Tactile (16 body + 2 prong):
      - Body receptors 0-7 (left body)  -> RIGHT hemisphere mid-freq nodes
      - Body receptors 8-15 (right body) -> LEFT hemisphere mid-freq nodes
      - Prong sensors (obs[258:260])     -> split between hemispheres

    Meters (obs[260:263]):
      - Life, satiation, valence -> midline/callosal nodes (shared)

    Motor nodes get NO direct sensory input — they receive drive only via
    coupling from other nodes.

    The assignment is built lazily on first call and rebuilt if node membership
    changes.
    """

    def __init__(
        self,
        network: SeedNetwork,
        input_scale: float = 1.0,
        action_feedback: bool = False,
    ):
        self._network = network
        self.input_scale = input_scale
        self.action_feedback = action_feedback
        self._prev_action: np.ndarray = np.zeros(3, dtype=np.float32)

        # Cached assignment: node_id -> list of obs indices
        self._assignment: Optional[Dict[str, List[int]]] = None
        self._assignment_key: Optional[int] = None  # hash of node set for rebuild

    def _sorted_hemi_nodes(
        self, hemi_set: Set[str], node_frequencies: Dict[str, float],
    ) -> List[str]:
        """Return non-motor nodes in *hemi_set* sorted by frequency (high first)."""
        motor_nids = set(self._network._motor_nodes.keys())
        return sorted(
            [nid for nid in hemi_set if nid not in motor_nids],
            key=lambda nid: node_frequencies.get(nid, 0.0), reverse=True,
        )

    @staticmethod
    def _vision_dims() -> tuple:
        """Return (left_vis_dims, right_vis_dims, center_dims)."""
        left = []
        for r in range(0, _RAY_MID):
            left.extend([r * 2, r * 2 + 1])
        right = []
        for r in range(_RAY_MID + 1, _N_RAYS):
            right.extend([r * 2, r * 2 + 1])
        center = [_RAY_MID * 2, _RAY_MID * 2 + 1]
        return left, right, center

    @staticmethod
    def _assign_chunks(
        nodes: List[str], dims: List[int], assignment: Dict[str, List[int]],
        extra: Optional[List[int]] = None,
    ) -> None:
        """Distribute *dims* evenly across *nodes*, appending *extra* to each."""
        if not nodes or not dims:
            return
        chunks = np.array_split(dims, len(nodes))
        for i, nid in enumerate(nodes):
            combined = list(chunks[i]) + (extra or [])
            assignment.setdefault(nid, []).extend(combined)

    def _build_assignment(self, node_frequencies: Dict[str, float]) -> None:
        """Build cross-lateralized observation-to-node mapping."""
        net = self._network
        l_nodes = self._sorted_hemi_nodes(net._hemisphere_L, node_frequencies)
        r_nodes = self._sorted_hemi_nodes(net._hemisphere_R, node_frequencies)
        m_nodes = self._sorted_hemi_nodes(net._hemisphere_M, node_frequencies)

        assignment: Dict[str, List[int]] = {}
        left_vis, right_vis, center = self._vision_dims()

        # Vision: top 60% — cross-lateralized
        n_vis_r = max(1, int(len(r_nodes) * 0.60))
        n_vis_l = max(1, int(len(l_nodes) * 0.60))
        self._assign_chunks(r_nodes[:n_vis_r], left_vis, assignment, center)
        self._assign_chunks(l_nodes[:n_vis_l], right_vis, assignment, center)

        # Tactile: next 25% — cross-wired
        left_tac = list(range(_TAC_START, _TAC_START + 8))
        right_tac = list(range(_TAC_START + 8, _TAC_BODY))
        prong = list(range(_TAC_BODY, _TAC_END))

        tac_end_r = min(n_vis_r + max(1, int(len(r_nodes) * 0.25)), len(r_nodes))
        tac_end_l = min(n_vis_l + max(1, int(len(l_nodes) * 0.25)), len(l_nodes))
        self._assign_chunks(r_nodes[n_vis_r:tac_end_r], left_tac, assignment, prong)
        self._assign_chunks(l_nodes[n_vis_l:tac_end_l], right_tac, assignment, prong)

        # Meters -> midline (or remaining lowest-freq)
        meter_dims = list(range(_METER_START, _METER_END))
        meter_targets = m_nodes or (r_nodes[tac_end_r:] + l_nodes[tac_end_l:])
        for nid in meter_targets:
            assignment.setdefault(nid, []).extend(meter_dims)

        # Unassigned non-motor nodes receive global signal only
        all_assigned = set(assignment.keys())
        for nid in l_nodes + r_nodes + m_nodes:
            if nid not in all_assigned:
                assignment[nid] = []

        self._assignment = assignment
        self._assignment_key = len(node_frequencies)

    def __call__(
        self,
        external_input: np.ndarray,
        node_frequencies: Dict[str, float],
    ) -> Dict[str, float]:
        obs = external_input
        if len(obs) < OBS_DIM:
            return dict.fromkeys(node_frequencies, 0.0)

        # Rebuild if node count changed
        if self._assignment is None or len(node_frequencies) != self._assignment_key:
            self._build_assignment(node_frequencies)

        # Global signals (weak shared time reference)
        vis = obs[:_VIS_END].reshape(_N_RAYS, 2)
        global_vis = float(np.mean(vis[:, 1])) * 0.05
        global_tac = float(np.mean(obs[_TAC_START:_TAC_END])) * 0.05

        drives = {}
        for nid in node_frequencies:
            # Motor nodes get NO direct input
            if nid in self._network._motor_nodes:
                drives[nid] = 0.0
                continue

            indices = self._assignment.get(nid)
            if indices and len(indices) > 0:
                local_drive = float(np.mean(obs[indices]))
            else:
                local_drive = 0.0

            drives[nid] = float(
                (local_drive + global_vis + global_tac) * self.input_scale
            )

        return drives

    def set_prev_action(self, action: np.ndarray) -> None:
        """Store previous action for action feedback."""
        self._prev_action = np.asarray(action, dtype=np.float32)


# -- MotorNodeActuator --------------------------------------------------------

class MotorNodeActuator:
    """Reads motor node activations from the network and computes actions.

    Push/pull population coding:
      forward = tanh(mean(fwd_push_L, fwd_push_R) - mean(fwd_pull_L, fwd_pull_R))
      turn    = tanh(turn_R - turn_L)    (contralateral: right hemi = left turn)
      eat     = mean(eat_L, eat_R) > threshold

    This replaces the external W_out readout entirely.
    """

    def __init__(self, network: SeedNetwork, eat_threshold: float = 0.0):
        self._network = network
        self._eat_threshold = eat_threshold
        self._last_action = np.zeros(3, dtype=np.float32)

    def __call__(self, node_activations: Dict[str, float]) -> np.ndarray:
        acts = self._network.get_motor_activations()
        if not acts:
            return np.zeros(3, dtype=np.float32)

        fwd_push = np.mean([
            acts.get("fwd_push_L", 0.0), acts.get("fwd_push_R", 0.0)
        ])
        fwd_pull = np.mean([
            acts.get("fwd_pull_L", 0.0), acts.get("fwd_pull_R", 0.0)
        ])
        forward = float(np.tanh(fwd_push - fwd_pull))

        # Contralateral: turn_R activation = turn left, turn_L = turn right
        turn = float(np.tanh(acts.get("turn_R", 0.0) - acts.get("turn_L", 0.0)))

        eat_signal = np.mean([acts.get("eat_L", 0.0), acts.get("eat_R", 0.0)])
        eat = 1.0 if eat_signal > self._eat_threshold else 0.0

        self._last_action = np.array([forward, turn, eat], dtype=np.float32)
        return self._last_action


# -- SeedBrain ----------------------------------------------------------------

class SeedBrain:
    """Self-organizing Seed brain with motor nodes inside the network.

    No external readout matrix (W_out). Actions come from motor node activations.
    Reward modulates the network's own Hebbian learning (Rule 2a).

    External interface: forward(), learn(), reset_state(), save(), load().
    """

    _state_sl = slice(_LIFE_IDX, _LIFE_IDX + 3)

    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = "cpu",
        action_feedback: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        cfg = dict(SEED_DEFAULT_CONFIG)
        if config:
            cfg.update(config)

        if seed is None:
            seed = int(cfg.get("seed", 42))

        self._dev = device  # kept for interface compat, but no torch needed
        self._action_feedback = action_feedback

        # -- Build SeedConfig with hemispheric + motor + reward modulation -----
        freq_range = cfg["freq_range"]
        if isinstance(freq_range, list):
            freq_range = tuple(freq_range)
        motor_freq_range = cfg["motor_freq_range"]
        if isinstance(motor_freq_range, list):
            motor_freq_range = tuple(motor_freq_range)

        seed_config = SeedConfig(
            n_init=int(cfg["n_init"]),
            k_neighbors=int(cfg["k_neighbors"]),
            n_seed_nodes=int(cfg["n_seed_nodes"]),
            max_nodes=int(cfg["max_nodes"]),
            noise_floor=float(cfg["noise_floor"]),
            drive_amplitude=float(cfg["drive_amplitude"]),
            activation_threshold=float(cfg["activation_threshold"]),
            freq_range=freq_range,
            learning_rate=float(cfg["learning_rate"]),
            w_max=float(cfg["w_max"]),
            freq_learning_rate=float(cfg["freq_learning_rate"]),
            # Hemispheric
            hemisphere_enabled=bool(cfg["hemisphere_enabled"]),
            n_callosal_bridges=int(cfg["n_callosal_bridges"]),
            callosal_bands=int(cfg["callosal_bands"]),
            # Motor
            motor_nodes_enabled=bool(cfg["motor_nodes_enabled"]),
            motor_node_spec=dict(cfg["motor_node_spec"]),
            motor_freq_range=motor_freq_range,
            # Reward modulation
            reward_modulation_enabled=bool(cfg["reward_modulation_enabled"]),
            rpe_scale=float(cfg["rpe_scale"]),
            rpe_baseline_lr_frac=float(cfg["rpe_baseline_lr_frac"]),
        )

        # Sensor and actuator need the network reference, so we build in order:
        # 1) network with temporary sensor/actuator
        # 2) real sensor/actuator with network ref
        # 3) swap them in
        _tmp_sensor = lambda ext, nf: dict.fromkeys(nf, 0.0)
        _tmp_actuator = lambda na: np.zeros(3, dtype=np.float32)
        self._seed_net = SeedNetwork(
            seed_config, _tmp_sensor, _tmp_actuator, seed=seed,
        )

        self._sensor = HemisphericSensor(
            network=self._seed_net,
            input_scale=float(cfg.get("input_scale", 1.0)),
            action_feedback=action_feedback,
        )
        self._actuator = MotorNodeActuator(
            network=self._seed_net,
            eat_threshold=float(cfg.get("eat_threshold", 0.0)),
        )
        self._seed_net.sensor = self._sensor
        self._seed_net.actuator = self._actuator

        # -- Learning state (minimal: just the scalar critic) ------------------
        self._valence_pred = 0.0
        self._critic_lr = float(cfg["critic_lr"])
        self.learning_enabled = True
        self._last_obs: Optional[np.ndarray] = None
        self._last_action = np.zeros(3, dtype=np.float32)
        self._executed_action = np.zeros(3, dtype=np.float32)
        self._has_executed_action = False

        # -- RO Framework ------------------------------------------------------
        self._observer = Observer(
            name="seed_brain",
            internal_dofs=dofs.INTERNAL_DOFS,
            external_dofs=dofs.EXTERNAL_DOFS,
            world_model=_IdentityMapping(),
            log_capacity=int(cfg["log_capacity"]),
        )
        self._tracker = KnowledgeTracker(
            observer=self._observer,
            external_dofs=dofs.EXTERNAL_DOFS,
            assess_interval=1,
            min_samples=50,
        )

    # -- Forward pass ----------------------------------------------------------

    def forward(self, obs: np.ndarray) -> tuple:
        """Advance one step. Returns (fwd, turn, eat)."""
        self._seed_net.step(obs)

        # Read action from actuator (which reads motor node activations)
        fwd = float(self._actuator._last_action[0])
        turn = float(self._actuator._last_action[1])
        eat = float(self._actuator._last_action[2])

        self._last_action = np.array([fwd, turn, eat], dtype=np.float32)
        self._last_obs = obs

        # Feed action back to sensor for next step
        if self._action_feedback:
            self._sensor.set_prev_action(self._last_action)

        # Append to observer log
        self._observer.observation_log.append(ObservationPair(
            external_state=dofs.obs_to_state(obs),
            internal_state=dofs.action_to_state(fwd, turn, eat),
            timestamp=float(len(self._observer.observation_log)),
        ))

        return fwd, turn, eat

    # -- Online learning -------------------------------------------------------

    def learn(self, reward: float) -> None:
        """Compute RPE and pass to network for reward-modulated Hebbian learning."""
        if not self.learning_enabled:
            return
        rpe = reward - self._valence_pred
        self._valence_pred += self._critic_lr * rpe
        self._seed_net.set_reward_modulator(rpe)

    def set_executed_action(self, action, **_kwargs) -> None:
        """Record the action the world actually executed (interface compat)."""
        if hasattr(action, 'numpy'):
            action = action.numpy()
        self._executed_action = np.asarray(action, dtype=np.float32)
        self._has_executed_action = True

    # -- Episode reset ---------------------------------------------------------

    def reset_state(self) -> None:
        """Zero reservoir activations. Coupling weights persist across episodes."""
        for node in self._seed_net.nodes.values():
            node.activation = 0.0
            node.branching_ratio = 0.0
            node.phase = 0.0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._executed_action = np.zeros(3, dtype=np.float32)
        self._has_executed_action = False
        self._seed_net.set_reward_modulator(0.0)

    # -- Knowledge tracker -----------------------------------------------------

    def step_knowledge(self, epoch: int) -> dict:
        return self._tracker.step(epoch)

    # -- Persistence -----------------------------------------------------------

    def save(self, path: str) -> None:
        """Save network state + critic."""
        net_dict = self._seed_net.to_dict()
        np.savez(
            path,
            valence_pred=np.array(self._valence_pred),
            seed_net_json=np.array(json.dumps(net_dict)),
        )

    def load(self, path: str) -> None:
        """Load network state + critic."""
        data = np.load(path, allow_pickle=True)

        if "seed_net_json" in data:
            net_dict = json.loads(str(data["seed_net_json"]))
            self._seed_net = SeedNetwork.from_dict(
                net_dict, self._sensor, self._actuator, seed=42,
            )
            # Reconnect sensor/actuator to new network instance
            self._sensor._network = self._seed_net
            self._actuator._network = self._seed_net

        if "valence_pred" in data:
            self._valence_pred = float(data["valence_pred"])

    # -- Properties ------------------------------------------------------------

    @property
    def w_out_norm(self) -> float:
        """Backward-compat stub: returns mean motor activation magnitude."""
        acts = self._seed_net.get_motor_activations()
        if not acts:
            return 0.0
        return float(np.mean([abs(v) for v in acts.values()]))

    @property
    def mean_sigma(self) -> float:
        """Mean branching ratio across all nodes."""
        sigmas = [n.branching_ratio for n in self._seed_net.nodes.values()]
        return float(np.mean(sigmas)) if sigmas else 0.0

    @property
    def node_count(self) -> int:
        return len(self._seed_net.nodes)

    @property
    def active_fraction(self) -> float:
        """Fraction of nodes currently above activation threshold."""
        thresh = self._seed_net.config.activation_threshold
        n_active = sum(
            1 for n in self._seed_net.nodes.values()
            if abs(n.activation) > thresh
        )
        return n_active / max(len(self._seed_net.nodes), 1)

    # brain_viz compatibility stubs
    @property
    def v1_res(self) -> _EmptyReservoir:
        return _EmptyReservoir()

    @property
    def tac_res(self) -> _EmptyReservoir:
        return _EmptyReservoir()

    @property
    def val_res(self) -> _EmptyReservoir:
        return _EmptyReservoir()

    @property
    def central_res(self) -> _EmptyReservoir:
        return _EmptyReservoir()

    @property
    def motor_res(self) -> _EmptyReservoir:
        return _EmptyReservoir()

    @property
    def _last_h_motor(self) -> np.ndarray:
        """Backward-compat: motor node activations as array."""
        acts = self._seed_net.get_motor_activations()
        return np.array(list(acts.values()), dtype=np.float32) if acts else np.zeros(1)


# -- Seed-aware logging --------------------------------------------------------

_SEED_CSV_FIELDS = [
    "seed_nodes", "seed_sigma", "seed_active_frac",
    "hemi_L", "hemi_R", "hemi_M",
]


def _seed_open_log(path: str) -> "csv.DictWriter":
    """Like brain._open_log but adds Seed-specific columns."""
    import csv as _csv
    from brain import _K_DOF_NAMES, _EXTRA_K_FIELDS
    k_fields = [f"K_{n}" for n in _K_DOF_NAMES]
    fields = (["timestamp", "step", "mean_reward", "valence_pred", "w_norm",
               "eat_count", "episodes", "mean_fwd", "mean_turn"]
              + k_fields + _EXTRA_K_FIELDS + _SEED_CSV_FIELDS)
    is_new = not os.path.exists(path)
    fh = open(path, "a", newline="", buffering=1)
    writer = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return writer


def _seed_log_step(brain, step, reward_sum, log_every,
                   eat_count, episodes, fwd_sum, turn_sum,
                   csv_writer=None) -> None:
    """Seed-specific step logger — replaces _log_base print with correct labels."""
    from datetime import datetime

    mean_r    = reward_sum / max(log_every, 1)
    mean_fwd  = fwd_sum   / max(log_every, 1)
    mean_turn = turn_sum  / max(log_every, 1)
    motor_mag = brain.w_out_norm  # mean |motor activation|

    print(f"step {step:>7}  valence {mean_r:+.3f}  v_pred {brain._valence_pred:+.3f}  "
          f"|motor| {motor_mag:.4f}  fwd:{mean_fwd:+.2f}  turn:{mean_turn:+.02f}  "
          f"eat:{eat_count}  ep:{episodes}")

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step, "mean_reward": f"{mean_r:.4f}",
        "valence_pred": f"{brain._valence_pred:.4f}", "w_norm": f"{motor_mag:.4f}",
        "eat_count": eat_count, "episodes": episodes,
        "mean_fwd": f"{mean_fwd:.4f}", "mean_turn": f"{mean_turn:.4f}",
    }

    _log_knowledge(brain, step, csv_row=row)
    row.update(_compute_extra_k(brain))

    if hasattr(brain, "node_count"):
        row["seed_nodes"] = brain.node_count
        row["seed_sigma"] = f"{brain.mean_sigma:.4f}"
        row["seed_active_frac"] = f"{brain.active_fraction:.4f}"
        n_l = len(brain._seed_net._hemisphere_L)
        n_r = len(brain._seed_net._hemisphere_R)
        n_m = len(brain._seed_net._hemisphere_M)
        row["hemi_L"] = n_l
        row["hemi_R"] = n_r
        row["hemi_M"] = n_m
        print(f"  Seed: nodes={brain.node_count}  "
              f"sigma={brain.mean_sigma:.3f}  "
              f"active={brain.active_fraction:.1%}  "
              f"hemi=L{n_l}/R{n_r}/M{n_m}")

    if csv_writer is not None:
        csv_writer.writerow(row)


def _install_seed_logging() -> None:
    """Monkey-patch brain module so run_connected/run_headless use Seed logging."""
    _brain_mod._log_step = _seed_log_step
    _brain_mod._open_log = _seed_open_log


# -- CLI -----------------------------------------------------------------------

def _resolve_config(args) -> dict:
    """Load config file; auto-discover alongside --load if --config not given."""
    cfg_path = args.config
    if cfg_path is None and args.load:
        auto = _config_path(args.load)
        if os.path.exists(auto):
            cfg_path = auto
            print(f"  [config] auto-loaded from {auto}")
    if cfg_path:
        with open(cfg_path) as f:
            raw = json.load(f)
        user_cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
        cfg = {**SEED_DEFAULT_CONFIG, **user_cfg}
    else:
        cfg = dict(SEED_DEFAULT_CONFIG)
    if args.device:
        cfg["device"] = args.device
    return cfg


def _build_brain(cfg: dict) -> SeedBrain:
    return SeedBrain(
        config=cfg,
        device=cfg.get("device", "cpu"),
        action_feedback=cfg.get("action_feedback", False),
        seed=cfg.get("seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-organizing Seed brain with hemispheric topology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python seed_brain.py                                  # connect to game
  python seed_brain.py --headless 3600                  # offline 1-hour run
  python seed_brain.py --config brains/configs/seed-64.json --save brains/Seed-64.npz
  python seed_brain.py --no-learn --load brains/Seed-64.npz
""")
    parser.add_argument("--config",     metavar="PATH",
                        help="JSON config file")
    parser.add_argument("--device",     choices=["cuda", "cpu"], default=None,
                        help="Override device (seed brain defaults to CPU)")
    parser.add_argument("--no-learn",   action="store_true",
                        help="Freeze learning (inference only)")
    parser.add_argument("--save",       metavar="PATH", default=None)
    parser.add_argument("--load",       metavar="PATH", default=None)
    parser.add_argument("--log-path",   metavar="PATH", default=None)
    parser.add_argument("--headless",   type=int, default=0, metavar="N",
                        help="Run N headless steps then exit (0 = connect to game)")
    parser.add_argument("--no-reset",   action="store_true",
                        help="Headless: keep AI alive at life=0 on death")
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY, metavar="N")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N")
    args = parser.parse_args()

    cfg = _resolve_config(args)
    brain_path, log_path, load_path = _resolve_paths(args, cfg)
    args.save = brain_path
    args.log_path = log_path

    n_init = cfg["n_init"]
    hemi = "ON" if cfg["hemisphere_enabled"] else "OFF"
    motor = "ON" if cfg["motor_nodes_enabled"] else "OFF"
    reward = "ON" if cfg["reward_modulation_enabled"] else "OFF"
    print(f"Building SeedBrain (n_init={n_init}, k={cfg['k_neighbors']})...")
    print(f"  Hemisphere: {hemi}  Motor: {motor}  RewardMod: {reward}")

    brain = _build_brain(cfg)

    n_l = len(brain._seed_net._hemisphere_L)
    n_r = len(brain._seed_net._hemisphere_R)
    n_m = len(brain._seed_net._hemisphere_M)
    n_motor = len(brain._seed_net._motor_nodes)
    print(f"  Device:        {brain._dev}")
    print(f"  Nodes:         {brain.node_count} (L={n_l} R={n_r} M={n_m})")
    print(f"  Motor nodes:   {n_motor}")
    print(f"  Mean sigma:    {brain.mean_sigma:.3f}")
    di = cfg.get("decision_interval", 1)
    print(f"  decision_int:  {di}")

    if load_path:
        brain.load(load_path)
        print(f"  Loaded from {load_path}")
        print(f"    Nodes: {brain.node_count}")

    if args.no_learn:
        brain.learning_enabled = False
        print("  Learning disabled.")

    if brain_path:
        save_config(cfg, _config_path(brain_path))

    args._brain_name = cfg.get("name") or ""
    args._world_config = cfg.get("world_config") or None
    args._decision_interval = cfg.get("decision_interval", 1)

    _install_seed_logging()

    if args.headless:
        print(f"\nRunning {args.headless} headless steps...")
        run_headless(brain, args)
    else:
        print("\nWaiting for game connection...")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
