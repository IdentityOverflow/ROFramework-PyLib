"""
experiments/embodied/wim_brain.py — Wave Interference Memory Brain

Uses a 2D spring network (WIM physics) as the reservoir instead of a random
recurrent matrix.

Architecture
------------
State:     disp  (N = grid_size²)         — wave displacement = reservoir h
Dynamics:  spring physics, O(N sparse)    — replaces tanh(W_res @ h)
Input:     obs[i] injected as wave kick at node input_nodes[i]
Readout:   W_out @ tanh(disp)  →  (fwd, turn, eat)
Learning:  RPE-gated eligibility trace (identical to brain.py)

Key differences from ESN (brain.py)
------------------------------------
  O(N) dynamics vs O(N²) — 64×64 = 4096 nodes, wave step ~10× cheaper than ESN
  Linear harmonic oscillator (nonlinearity from tanh at readout only)
  Natural 2D spatial layout — input injection can be spatially structured
  Anchor nodes fix boundary displacement → richer standing wave modes

Parameters analogous to ESN
----------------------------
  tension      ≈  wave speed / coupling strength    (higher = faster propagation)
  damping      ≈  memory decay per step             (higher = shorter echo horizon)
  noise_scale  ≈  reservoir noise scale             (richness / exploration)
  input_scale  ≈  W_in norm                         (observation amplitude at injection)

Observation layout (obs[263], all values in [0, 1])
----------------------------------------------------
  obs[0:242]   Vision:  121 rays × [type_norm, proximity]
  obs[242:260] Tactile: 16 body + 2 prong receptors
  obs[260:263] Meters:  [life, satiation_norm, valence_norm]

Usage
-----
    python wim_brain.py                        # connect to game.py --connect
    python wim_brain.py --headless 3600
    python wim_brain.py --config brains/configs/wim-64.json --save brains/Wim-64.npz
    python wim_brain.py --headless 600 --device cpu --log-every 300
    python wim_brain.py --no-learn --load brains/Wim-64.npz   # inference only
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

# ── Resolve library path ───────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_repo     = os.path.normpath(os.path.join(_here, "..", ".."))
_src_path = os.path.join(_repo, "src")
for _p in (_src_path, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ro_framework.core.state import State                      # noqa: E402
from ro_framework.observer.observer import Observer, ObservationPair  # noqa: E402
from ro_framework.knowledge.tracker import KnowledgeTracker   # noqa: E402

import dofs                                                    # noqa: E402

# Reuse generic infrastructure from brain.py (run loops, logging, config helpers)
from brain import (                                            # noqa: E402
    run_connected, run_headless,
    _log_step, _open_log, _format_k,
    load_config, save_config, _select_device,
    _config_path, _resolve_paths,
    OBS_DIM, _LIFE_IDX,
    LOG_EVERY, SAVE_EVERY, LOG_CAPACITY, ASSESS_EVERY,
    EXPLORE_NOISE, EAT_THRESHOLD, LEARN_LR, CRITIC_LR, TRACE_DECAY, WEIGHT_DECAY,
    _IdentityMapping, _EmptyReservoir,
)

# ── Wave hyperparameters ───────────────────────────────────────────────────────
GRID_SIZE   = 64     # 64×64 = 4096 nodes — matches default ESN size for fair comparison
TENSION     = 0.3    # spring stiffness (controls wave propagation speed)
DAMPING     = 0.02   # velocity decay per step (shorter = longer echo)
NOISE_SCALE = 0.01   # small per-step displacement noise (wave richness)
INPUT_SCALE = 1.0    # amplitude of obs injection at input nodes

# ── Default config ─────────────────────────────────────────────────────────────
WAVE_DEFAULT_CONFIG: dict = {
    # Identity / paths
    "name":             "",
    "brain_path":       "",
    "log_path":         "",
    "world_config":     "",
    # Wave reservoir
    "grid_size":        GRID_SIZE,
    "tension":          TENSION,
    "damping":          DAMPING,
    "noise_scale":      NOISE_SCALE,
    "input_scale":      INPUT_SCALE,
    "anchor_layout":    "golden",   # "golden" | "centered" (only when brain_layout=false)
    "brain_layout":     True,       # corpus callosum + cross-lateralized sensory placement
    # Readout / learning  (same keys as brain.py for config compatibility)
    "explore_noise":    0.3,        # wave brain default — lower than ESN's 90
    "eat_threshold":    EAT_THRESHOLD,
    "learn_lr":         LEARN_LR,
    "critic_lr":        CRITIC_LR,
    "trace_decay":      TRACE_DECAY,
    "weight_decay":     WEIGHT_DECAY,
    # RO Framework
    "assess_every":     ASSESS_EVERY,
    "log_capacity":     LOG_CAPACITY,
    # Arch
    "seed":             42,
    "action_feedback":  False,
    "decision_interval": 1,
    "device":           "cuda",      # auto-selects cuda if available
}


# ── Grid helpers ───────────────────────────────────────────────────────────────

def _build_neighbors(size: int):
    """Precompute 8-connected neighbor arrays for a size×size grid.

    Returns
    -------
    nb_safe      : (N, 8) int32   — neighbor indices, boundary entries clamped to 0
    nb_valid     : (N, 8) float32 — 1 where neighbour exists, 0 at boundary
    nb_inv_count : (N,)  float32  — 1 / valid_neighbour_count per node
    """
    n = size * size
    idx = np.arange(n, dtype=np.int32).reshape(size, size)
    nb = np.full((n, 8), -1, dtype=np.int32)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for k, (dx, dy) in enumerate(offsets):
        xs = slice(max(0, -dx), size - max(0, dx))
        xd = slice(max(0,  dx), size + min(0, dx))
        ys = slice(max(0, -dy), size - max(0, dy))
        yd = slice(max(0,  dy), size + min(0, dy))
        src = idx[xs, ys].ravel()
        dst = idx[xd, yd].ravel()
        nb[src, k] = dst

    nb_safe = np.maximum(nb, 0)
    nb_valid = (nb >= 0).astype(np.float32)
    nb_inv_count = (1.0 / np.maximum(nb_valid.sum(axis=1), 1.0)).astype(np.float32)
    return nb_safe, nb_valid, nb_inv_count


def _nearest_unique_grid_pts(size: int, targets):
    """Snap float (x, y) targets to nearest unused grid index pairs."""
    xi, yi = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.column_stack((xi.ravel(), yi.ravel())).astype(np.float32)
    unused = np.ones(len(cells), dtype=bool)
    chosen = []
    for tx, ty in targets:
        d2 = (cells[:, 0] - tx) ** 2 + (cells[:, 1] - ty) ** 2
        d2[~unused] = np.inf
        p = int(np.argmin(d2))
        unused[p] = False
        chosen.append((int(cells[p, 0]), int(cells[p, 1])))
    return chosen


def _golden_anchor_grid(size: int):
    """9 anchor points in a Vogel (golden-angle) spiral — maximally even spacing."""
    c   = 0.5 * (size - 1)
    r   = max(1.0, 0.34 * (size - 1))
    phi = np.pi * (3.0 - np.sqrt(5.0))   # golden angle ≈ 137.5°
    targets = [(c, c)]
    for k in range(1, 9):
        radius = r * np.sqrt(k / 8.0)
        angle  = k * phi - 0.5 * np.pi   # rotate so k=1 points "up"
        targets.append((c + np.cos(angle) * radius, c + np.sin(angle) * radius))
    return _nearest_unique_grid_pts(size, targets)


def _centered_anchor_grid(size: int):
    """9 anchor points in a symmetric 3×3 grid."""
    mid  = (size - 1) // 2
    span = round(0.3 * (size - 1))
    idxs = [max(0, mid - span), mid, min(size - 1, mid + span)]
    targets = [(float(ix), float(iy)) for ix in idxs for iy in idxs]
    return _nearest_unique_grid_pts(size, targets)


def _corner_anchors(size: int):
    """4 corner anchor points."""
    m = max(1, size // 8)
    return [(m, m), (m, size - 1 - m), (size - 1 - m, m), (size - 1 - m, size - 1 - m)]


# ── Brain layout (corpus callosum + cross-lateralized sensory input) ──────────

def _spread_in_region(n, x0, x1, y0, y1):
    """Generate n evenly spaced (x, y) positions in a rectangle (grid coords)."""
    if n <= 0:
        return []
    w = abs(x1 - x0)
    h = abs(y1 - y0)
    aspect = max(w, 0.1) / max(h, 0.1)
    cols = max(1, int(round(np.sqrt(n * aspect))))
    rows = max(1, int(np.ceil(n / cols)))
    points = []
    for i in range(n):
        c = i % cols
        r = i // cols
        fx = (c + 0.5) / cols
        fy = (r + 0.5) / rows
        points.append((float(x0 + fx * (x1 - x0)), float(y0 + fy * (y1 - y0))))
    return points


def _snap_to_free_grid(size, targets, unavailable):
    """Snap (x, y) targets to nearest available grid cells.  Returns flat indices."""
    xi, yi = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.column_stack((xi.ravel(), yi.ravel())).astype(np.float32)
    used = np.copy(unavailable)
    indices = np.empty(len(targets), dtype=np.int64)
    for i, (tx, ty) in enumerate(targets):
        d2 = (cells[:, 0] - tx) ** 2 + (cells[:, 1] - ty) ** 2
        d2[used] = np.inf
        p = int(np.argmin(d2))
        used[p] = True
        indices[i] = p
    return indices


def _brain_input_layout(grid_size, input_dim=263, anchor_mask=None):
    """Corpus-callosum barrier + cross-lateralized sensory placement.

    Layout (dorsal view, creature facing y=0):
        y=0 (anterior / frontal — motor, decision)
        ┌──────────┬─║─┬──────────┐
        │          │ ║ │          │
        │   LEFT   │ ║ │  RIGHT   │
        │   HEMI   │ ║ │  HEMI    │
        │          │ ║ │          │
        │ R-tactile│ ║ │L-tactile │   (somatosensory, mid y)
        │          │ ║ │          │
        │ R-vision │ ║ │L-vision  │   (occipital, posterior)
        └──────────┴─║─┴──────────┘
        y=grid_size (posterior / occipital — vision)

    Cross-wiring: left visual field → right hemisphere, and vice versa.
    """
    mid = grid_size // 2
    N = grid_size * grid_size

    # ── Barrier (corpus callosum): vertical at x=mid, slits every other y ──
    barrier = np.zeros(N, dtype=bool)
    for y in range(grid_size):
        if y % 2 == 0:
            barrier[mid * grid_size + y] = True

    # ── Unavailable cells (barrier + anchors) ──
    unavailable = np.copy(barrier)
    if anchor_mask is not None:
        unavailable |= anchor_mask

    # ── Region bounds (grid index coordinates) ──
    margin = max(1, grid_size // 16)
    # Hemisphere x ranges (leave margin from edge and barrier)
    lx0, lx1 = float(margin), float(mid - margin - 1)       # left hemi
    rx0, rx1 = float(mid + margin + 1), float(grid_size - margin - 1)  # right hemi

    # Posterior (vision): y in [60%..95%] of grid
    vy0, vy1 = grid_size * 0.55, grid_size - margin - 1.0

    # Somatosensory (tactile): y in [25%..50%] of grid
    ty0, ty1 = grid_size * 0.25, grid_size * 0.50

    # Central / deep (meters): near barrier, y ~ 35-45%
    my_y = grid_size * 0.38

    # ── Generate target positions for each input channel ──
    targets = []

    # --- Vision: 121 rays × 2 channels = 242 ---
    # obs[2*i] = ray i type, obs[2*i+1] = ray i proximity
    # Rays 0-59: left visual field → RIGHT hemisphere posterior
    #   Ray 0 = most peripheral left → lateral (high x)
    #   Ray 59 = near-center → medial (low x, near barrier)
    n_field = 60
    right_vis = _spread_in_region(n_field, rx1, rx0, vy0, vy1)
    for i in range(n_field):
        tx, ty = right_vis[i]
        targets.append((tx, ty))               # type
        targets.append((tx + 0.4, ty + 0.4))   # proximity (nearby)

    # Ray 60 (center): split between hemispheres
    targets.append((rx0, vy0 - 1))   # type → right hemi
    targets.append((lx1, vy0 - 1))   # proximity → left hemi

    # Rays 61-120: right visual field → LEFT hemisphere posterior
    #   Ray 61 = near-center → medial (high x, near barrier)
    #   Ray 120 = most peripheral right → lateral (low x)
    left_vis = _spread_in_region(n_field, lx1, lx0, vy0, vy1)
    for i in range(n_field):
        tx, ty = left_vis[i]
        targets.append((tx, ty))
        targets.append((tx - 0.4, ty + 0.4))

    # --- Tactile: 18 channels (obs[242:260]) ---
    # First 8 body sensors (left body → right hemi)
    right_tac = _spread_in_region(8, rx0, rx1, ty0, ty1)
    for tx, ty in right_tac:
        targets.append((tx, ty))
    # Next 8 body sensors (right body → left hemi)
    left_tac = _spread_in_region(8, lx1, lx0, ty0, ty1)
    for tx, ty in left_tac:
        targets.append((tx, ty))
    # 2 prong sensors (front, split)
    targets.append((rx0 + 1, ty0 - 2))
    targets.append((lx1 - 1, ty0 - 2))

    # --- Meters: 3 channels (obs[260:263]) ---
    targets.append((mid - margin - 1, my_y))       # life
    targets.append((mid + margin + 1, my_y))       # satiation
    targets.append((mid - margin - 1, my_y + 2))   # valence

    assert len(targets) == input_dim, f"Expected {input_dim} targets, got {len(targets)}"

    input_indices = _snap_to_free_grid(grid_size, targets, unavailable)
    return barrier, input_indices


# ── WaveReservoir ──────────────────────────────────────────────────────────────

class WaveReservoir:
    """2D spring-network reservoir — WIM physics as a reservoir computing substrate.

    Parameters
    ----------
    grid_size    : Grid is grid_size × grid_size nodes. N = grid_size².
    input_dim    : Observation dimension (263 standard, 266 with action feedback).
    tension      : Spring stiffness coefficient. Higher = faster wave propagation.
    damping      : Velocity decay per step [0, 1). Higher = shorter memory.
    noise_scale  : Per-step Gaussian displacement noise on free nodes.
    input_scale  : Amplitude scale for observation injection.
    anchors      : Whether to fix anchor nodes at zero displacement.
    anchor_layout: "golden" (Vogel spiral, default) | "centered" (3×3 grid).
    seed         : RNG seed for input node assignment.
    device       : torch device string or None (auto-selects cuda if available).
    """

    def __init__(
        self,
        grid_size:     int   = GRID_SIZE,
        input_dim:     int   = OBS_DIM,
        tension:       float = TENSION,
        damping:       float = DAMPING,
        noise_scale:   float = NOISE_SCALE,
        input_scale:   float = INPUT_SCALE,
        anchors:       bool  = True,
        anchor_layout: str   = "golden",
        seed:          int   = 42,
        device:        Optional[str] = None,
        extra_anchors: Optional[np.ndarray] = None,
        barrier:       Optional[np.ndarray] = None,
        input_nodes:   Optional[np.ndarray] = None,
    ) -> None:
        self.grid_size    = grid_size
        self.size         = grid_size * grid_size   # N — total nodes
        self._tension     = float(tension)
        self._damping     = float(damping)
        self._noise_scale = float(noise_scale)
        self._input_scale = float(input_scale)

        # Device selection
        if device is None:
            self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._dev = torch.device(device)

        # Anchor nodes — held at zero (break translational symmetry → richer modes)
        is_anchor_np = np.zeros(self.size, dtype=bool)
        if anchors:
            pts = _golden_anchor_grid(grid_size) if anchor_layout == "golden" \
                  else _centered_anchor_grid(grid_size)
            for ax, ay in pts:
                is_anchor_np[ax * grid_size + ay] = True

        # Extra anchors (e.g. corner pins for brain layout)
        if extra_anchors is not None:
            is_anchor_np |= extra_anchors.astype(bool)

        # Barrier nodes (corpus callosum) — fixed at zero but tracked separately
        barrier_np = np.zeros(self.size, dtype=bool)
        if barrier is not None:
            barrier_np = barrier.astype(bool)
            is_anchor_np |= barrier_np
        free_mask_np = ~is_anchor_np

        # Precomputed neighbor connectivity (sparse 8-neighbour coupling)
        nb_safe_np, nb_valid_np, nb_inv_count_np = _build_neighbors(grid_size)

        # Input node assignment
        if input_nodes is not None:
            input_nodes_np = input_nodes.astype(np.int64)
            self._n_in = len(input_nodes_np)
        else:
            rng      = np.random.default_rng(seed)
            free_idx = np.where(free_mask_np)[0]
            n_in     = min(input_dim, len(free_idx))
            input_nodes_np = rng.choice(free_idx, size=n_in, replace=False).astype(np.int64)
            self._n_in = n_in

        # ── Torch tensors (GPU) ──────────────────────────────────────────────
        dev = self._dev

        # State tensors
        self._disp   = torch.zeros(self.size, dtype=torch.float32, device=dev)
        self._vel    = torch.zeros(self.size, dtype=torch.float32, device=dev)
        self._energy = torch.zeros(self.size, dtype=torch.float32, device=dev)

        # Topology tensors
        self._nb_safe      = torch.from_numpy(nb_safe_np.astype(np.int64)).to(dev)        # (N, 8) long
        self._nb_valid     = torch.from_numpy(nb_valid_np).to(dev)                         # (N, 8) float32
        self._nb_inv_count = torch.from_numpy(nb_inv_count_np).to(dev)                     # (N,)   float32

        # Mask tensors
        self._is_anchor  = torch.from_numpy(is_anchor_np).to(dev)   # (N,) bool
        self._free_mask  = torch.from_numpy(free_mask_np).to(dev)   # (N,) bool
        self._is_barrier = torch.from_numpy(barrier_np).to(dev)     # (N,) bool

        # Input nodes
        self._input_nodes = torch.from_numpy(input_nodes_np).to(dev)  # (n_in,) long

        # brain_viz compat: no dense W_res; return (0,0) — graph view shows no edges
        self.recurrent_weights = np.zeros((0, 0), dtype=np.float32)

    # ── Physics step (shared by step() and step_wave()) ─────────────────────

    def _physics_step(self) -> None:
        """One step of discrete wave equation with energy tracking."""
        # Gather neighbor displacements
        nb_d = self._disp[self._nb_safe] * self._nb_valid   # (N, 8)
        mean_nb = nb_d.sum(dim=1) * self._nb_inv_count

        # Force and velocity
        force = self._tension * (mean_nb - self._disp)
        self._vel.mul_(1.0 - self._damping).add_(force)
        self._vel[self._is_anchor] = 0.0
        self._disp.add_(self._vel)

        # Noise on free nodes
        if self._noise_scale > 0.0:
            noise = torch.randn(self.size, device=self._dev) * self._noise_scale
            noise[self._is_anchor] = 0.0
            self._disp.add_(noise)

        self._disp[self._is_anchor] = 0.0

        # Energy tracking (for viz): free nodes = |vel|+|disp|, anchors absorb from neighbors
        nb_e = self._energy[self._nb_safe] * self._nb_valid
        sum_nb_e = nb_e.sum(dim=1)
        self._energy[self._free_mask] = (
            self._vel[self._free_mask].abs() + self._disp[self._free_mask].abs()
        ).clamp_(0, 1)
        self._energy[self._is_anchor] = (
            sum_nb_e[self._is_anchor] * self._nb_inv_count[self._is_anchor] * 0.5
        ).clamp_(0, 1)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Advance one step: inject input, run spring physics, return state.

        Parameters
        ----------
        x : (input_dim,) torch.float32 tensor on device

        Returns
        -------
        (N,) torch.float32 tensor — tanh(displacement), the new hidden state (on device)
        """
        # Inject observation as wave kicks at designated input nodes
        self._disp[self._input_nodes] += self._input_scale * x[:self._n_in]

        self._physics_step()

        return torch.tanh(self._disp)

    def step_wave(self) -> None:
        """Advance one physics step with no input injection (standalone wave mode)."""
        self._physics_step()

    @property
    def state(self) -> np.ndarray:
        """tanh-compressed displacement as numpy array — for brain_viz compatibility."""
        return torch.tanh(self._disp).cpu().numpy()

    def reset(self) -> None:
        """Zero displacement, velocity, and energy (episode reset)."""
        self._disp.zero_()
        self._vel.zero_()
        self._energy.zero_()


# ── WimBrain ───────────────────────────────────────────────────────────────────

class WimBrain:
    """
    Wave Interference Memory Brain — drop-in replacement for EmbodiedBrain.

    Identical external interface: forward(), learn(), reset_state(), save(), load().
    Uses WaveReservoir instead of SingleReservoir.  All run loops and logging
    infrastructure are imported from brain.py and work unchanged via duck typing.

    Parameters
    ----------
    config : dict or None
        Override any WAVE_DEFAULT_CONFIG key.
    device : str
        "cpu" (default) or "cuda" (only the readout/learning tensors live on GPU;
        wave physics is always numpy/CPU — overhead is negligible).
    action_feedback : bool
        Feed previous (fwd, turn, eat) back as additional input (input_dim += 3).
    seed : int
        RNG seed for reservoir initialisation.
    """

    _state_sl = slice(_LIFE_IDX, _LIFE_IDX + 3)   # obs[260:263] for episode detect

    def __init__(
        self,
        config:          Optional[dict] = None,
        device:          str            = "cpu",
        action_feedback: bool           = False,
        seed:            int            = 42,
    ) -> None:
        cfg = dict(WAVE_DEFAULT_CONFIG)
        if config:
            cfg.update(config)

        self._dev             = _select_device(device)
        self._action_feedback = action_feedback
        grid_size             = int(cfg["grid_size"])
        input_dim             = OBS_DIM + (3 if action_feedback else 0)
        res_size              = grid_size * grid_size

        # ── Wave reservoir ──────────────────────────────────────────────────
        brain_layout = cfg.get("brain_layout", True)

        if brain_layout:
            # Corner anchors only (barrier provides boundary conditions)
            anchor_pts = _corner_anchors(grid_size)
            anchor_mask = np.zeros(grid_size * grid_size, dtype=bool)
            for ax, ay in anchor_pts:
                anchor_mask[ax * grid_size + ay] = True
            barrier_mask, input_node_indices = _brain_input_layout(
                grid_size, input_dim, anchor_mask,
            )
            self._reservoir = WaveReservoir(
                grid_size     = grid_size,
                input_dim     = input_dim,
                tension       = cfg["tension"],
                damping       = cfg["damping"],
                noise_scale   = cfg["noise_scale"],
                input_scale   = cfg["input_scale"],
                anchors       = False,
                seed          = seed,
                device        = str(self._dev),
                extra_anchors = anchor_mask,
                barrier       = barrier_mask,
                input_nodes   = input_node_indices,
            )
        else:
            self._reservoir = WaveReservoir(
                grid_size     = grid_size,
                input_dim     = input_dim,
                tension       = cfg["tension"],
                damping       = cfg["damping"],
                noise_scale   = cfg["noise_scale"],
                input_scale   = cfg["input_scale"],
                anchors       = True,
                anchor_layout = cfg["anchor_layout"],
                seed          = seed,
                device        = str(self._dev),
            )

        # ── Readout (trained) ───────────────────────────────────────────────
        rng = np.random.default_rng(seed + 99)
        self.W_out = torch.from_numpy(
            rng.standard_normal((3, res_size)).astype(np.float32) * 0.01
        ).to(self._dev)
        self.b_out = torch.zeros(3, dtype=torch.float32, device=self._dev)

        # ── Online learning state ───────────────────────────────────────────
        self._trace        = torch.zeros(3, res_size, dtype=torch.float32, device=self._dev)
        self._valence_pred = 0.0
        self._last_action  = torch.zeros(3,        dtype=torch.float32, device=self._dev)
        self._last_h       = torch.zeros(res_size, dtype=torch.float32, device=self._dev)

        self._explore_noise   = float(cfg["explore_noise"])
        self._eat_threshold   = float(cfg["eat_threshold"])
        self._learn_lr        = float(cfg["learn_lr"])
        self._critic_lr       = float(cfg["critic_lr"])
        self._trace_decay     = float(cfg["trace_decay"])
        self._weight_decay    = float(cfg["weight_decay"])
        self.learning_enabled = True
        self._last_obs: Optional[np.ndarray] = None

        # ── RO Framework — Observer + KnowledgeTracker ─────────────────────
        self._observer = Observer(
            name          = "wim_brain",
            internal_dofs = dofs.INTERNAL_DOFS,
            external_dofs = dofs.EXTERNAL_DOFS,
            world_model   = _IdentityMapping(),
            log_capacity  = int(cfg["log_capacity"]),
        )
        self._tracker = KnowledgeTracker(
            observer        = self._observer,
            external_dofs   = dofs.EXTERNAL_DOFS,
            assess_interval = 1,
            min_samples     = 50,
        )

    # ── Forward pass ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, obs: np.ndarray) -> tuple[float, float, float]:
        """Advance one step.  obs: (263,) float32.  Returns (fwd, turn, eat)."""
        x = torch.from_numpy(obs.astype(np.float32)).to(self._dev)
        if self._action_feedback:
            x = torch.cat([x, self._last_action])

        h = self._reservoir.step(x)                        # torch tensor on device
        self._last_h.copy_(h)

        raw   = self.W_out @ h + self.b_out
        noise = torch.randn(2, dtype=torch.float32, device=self._dev) * self._explore_noise

        fwd  = float(torch.tanh(raw[0] + noise[0]))
        turn = float(torch.tanh(raw[1] + noise[1]))
        eat  = 1.0 if float(raw[2]) > self._eat_threshold else 0.0

        self._last_action[0] = fwd
        self._last_action[1] = turn
        self._last_action[2] = eat
        self._last_obs = obs

        self._observer.observation_log.append(ObservationPair(
            external_state = dofs.obs_to_state(obs),
            internal_state = dofs.action_to_state(fwd, turn, eat),
            timestamp      = float(len(self._observer.observation_log)),
        ))

        return fwd, turn, eat

    # ── Online learning ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def learn(self, reward: float) -> None:
        """RPE-gated eligibility trace update.  Call once per step after forward()."""
        if not self.learning_enabled:
            return
        rpe = reward - self._valence_pred
        self._valence_pred += self._critic_lr * rpe
        self._trace.mul_(self._trace_decay).add_(
            torch.outer(self._last_action, self._last_h)
        )
        self.W_out.add_(self._trace, alpha=self._learn_lr * rpe)
        self.W_out.mul_(1.0 - self._weight_decay)

    # ── Episode reset ────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Zero wave state, eligibility trace, and last action.  Keeps valence_pred."""
        self._reservoir.reset()
        self._trace.zero_()
        self._last_action.zero_()
        self._last_h.zero_()

    # ── Knowledge tracker ────────────────────────────────────────────────────────

    def step_knowledge(self, epoch: int) -> dict:
        """Advance KnowledgeTracker.  Returns assessment dict or {}."""
        return self._tracker.step(epoch)

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path,
                 W_out        = self.W_out.cpu().numpy(),
                 b_out        = self.b_out.cpu().numpy(),
                 valence_pred = np.array(self._valence_pred))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.W_out.copy_(torch.from_numpy(data["W_out"].astype(np.float32)))
        self.b_out.copy_(torch.from_numpy(data["b_out"].astype(np.float32)))
        self._valence_pred = float(data["valence_pred"])

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def w_out_norm(self) -> float:
        return float(torch.linalg.norm(self.W_out))

    # brain_viz compatibility — v1_res and motor_res are the single active reservoir
    @property
    def v1_res(self) -> WaveReservoir:
        return self._reservoir

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
    def motor_res(self) -> WaveReservoir:
        return self._reservoir

    @property
    def _last_h_motor(self) -> torch.Tensor:
        return self._last_h


# ── CLI helpers ──────────────────────────────────────────────────────────────────

def _resolve_config(args) -> dict:
    """Load config file; auto-discover alongside --load if --config not given."""
    cfg_path = args.config
    if cfg_path is None and args.load:
        auto = _config_path(args.load)
        if os.path.exists(auto):
            cfg_path = auto
            print(f"  [config] auto-loaded from {auto}")
    if cfg_path:
        raw = load_config(cfg_path)
        # Merge: WAVE_DEFAULT_CONFIG provides defaults for wave-specific keys
        cfg = {**WAVE_DEFAULT_CONFIG, **raw}
    else:
        cfg = dict(WAVE_DEFAULT_CONFIG)
    if args.device:
        cfg["device"] = args.device
    return cfg


def _build_brain(cfg: dict) -> WimBrain:
    return WimBrain(
        config = {k: cfg[k] for k in WAVE_DEFAULT_CONFIG if k in cfg},
        device          = cfg["device"],
        action_feedback = cfg.get("action_feedback", False),
        seed            = cfg.get("seed", 42),
    )


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave Interference Memory Brain (2D spring-network reservoir).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hyperparameters are set via a JSON config file (--config).
Same format as brain.py configs; wave-specific keys: grid_size, tension, damping,
noise_scale, input_scale, anchor_layout.

examples:
  python wim_brain.py --config brains/configs/wim-64.json --save brains/Wim-64.npz
  python wim_brain.py --load brains/Wim-64.npz                   # auto-loads .json
  python wim_brain.py --headless 3600 --log-every 300
  python wim_brain.py --headless 600 --device cpu --no-learn
""")
    parser.add_argument("--config",     metavar="PATH",
                        help="JSON config file")
    parser.add_argument("--device",    choices=["cuda", "cpu"],
                        help="Override device from config (readout/learning only)")
    parser.add_argument("--no-learn",  action="store_true",
                        help="Freeze W_out (inference only)")
    parser.add_argument("--save",      metavar="PATH", default=None,
                        help="Override brain_path for this run")
    parser.add_argument("--load",      metavar="PATH", default=None,
                        help="Force-load weights from PATH")
    parser.add_argument("--log-path",  metavar="PATH", default=None,
                        help="Override log_path for this run")
    parser.add_argument("--headless",  type=int, default=0, metavar="N",
                        help="Run N headless steps then exit (0 = connect to game)")
    parser.add_argument("--no-reset",  action="store_true",
                        help="Headless: keep AI alive at life=0 on death")
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY,  metavar="N")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N")
    args = parser.parse_args()

    cfg                             = _resolve_config(args)
    brain_path, log_path, load_path = _resolve_paths(args, cfg)
    args.save     = brain_path
    args.log_path = log_path

    grid_size = int(cfg["grid_size"])
    res_size  = grid_size * grid_size
    print(f"Building WimBrain  (grid={grid_size}×{grid_size}={res_size} nodes, "
          f"action_feedback={cfg['action_feedback']})…")

    brain = _build_brain(cfg)

    n_input = brain._reservoir._n_in
    print(f"  Device:         {brain._dev}")
    print(f"  input_dim:      {OBS_DIM + (3 if cfg['action_feedback'] else 0)}  "
          f"({n_input} nodes receive direct obs injection)")
    print(f"  grid:           {grid_size}×{grid_size} = {res_size} nodes")
    print(f"  tension:        {cfg['tension']}   damping: {cfg['damping']}")
    print(f"  noise_scale:    {cfg['noise_scale']}   input_scale: {cfg['input_scale']}")
    print(f"  anchor_layout:  {cfg['anchor_layout']}")
    print(f"  explore_noise:  {cfg['explore_noise']}")
    print(f"  W_out:          {tuple(brain.W_out.shape)}  |norm|={brain.w_out_norm:.5f}")

    if load_path:
        brain.load(load_path)
        print(f"  Loaded weights from {load_path}  |W_out|={brain.w_out_norm:.5f}")
    elif brain_path:
        print(f"  No checkpoint found at {brain_path} — starting fresh.")

    if args.no_learn:
        brain.learning_enabled = False
        print("  Learning disabled.")

    if brain_path:
        save_config(cfg, _config_path(brain_path))

    args._brain_name        = cfg.get("name") or ""
    args._world_config      = cfg.get("world_config") or None
    args._decision_interval = cfg.get("decision_interval", 1)

    if args.headless:
        print(f"\nRunning {args.headless} headless steps…")
        run_headless(brain, args)
    else:
        print("\nWaiting for game connection…")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
