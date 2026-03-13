"""
experiments/embodied/dofs.py — DoF schema for the embodied AI experiment.

All DoF instances are module-level singletons. The RO Framework uses identity-based
hashing for DoFs, so they must never be recreated between calls.

External DoFs (what the brain perceives):
  food_max_proximity   — ScalarDoF [0,1]: highest food proximity in forward vision cone
  danger_max_proximity — ScalarDoF [0,1]: highest danger proximity in forward vision cone
  life                 — ScalarDoF [0,1]: entity health (0=dead, 1=full)
  satiation_norm       — PolarDoF  [0,1]: 0=starving · 0.5=neutral · 1=full (hunger↔satiation)
  valence_norm         — PolarDoF  [0,1]: 0=max pain · 0.5=neutral · 1=max pleasure

Internal DoFs — Phase 1 (action outputs):
  fwd_output  — PolarDoF [-1,1]: forward/reverse motor command (+1=full fwd, -1=full rev)
  turn_output — PolarDoF [-1,1]: turn motor command (+1=right, -1=left)
  eat_output  — ScalarDoF [0,1]: eat action (0=no, 1=eat)

Internal DoFs — Phase 2 stubs (frequency band amplitudes, not yet wired):
  vis_band_power — mean Goertzel amplitude of visual-band reservoir neurons
  tac_band_power — mean Goertzel amplitude of tactile-band reservoir neurons
  val_band_power — mean Goertzel amplitude of internal-state-band reservoir neurons

Observation vector layout (obs[263], all values in [0,1]):
  obs[0:242]   Vision: 121 rays × [type_norm, proximity]
                 type_norm: 0=none · 0.25=wall · 0.5=food · 0.75=danger · 1.0=other
                 proximity: 1 − dist/range when hit, else 0
  obs[242:258] Body tactile: 16 receptors (signal: wall=0.7, food=0.3, entity=0.5, danger=1.0)
  obs[258:260] Prong tactile: 2 receptors
  obs[260:263] Meters: [life, satiation_norm, valence_norm]
"""

from __future__ import annotations

import sys
import os

import numpy as np

# ── Resolve library path ───────────────────────────────────────────────────────
# Allow running this file directly from experiments/embodied/ without installing
_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_src_path  = os.path.join(_repo_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from ro_framework import PolarDoF, ScalarDoF, State

# ── Observation constants (must match env.py) ─────────────────────────────────
N_RAYS         = 121
N_TOUCH_BODY   = 16
N_TOUCH_PRONGS = 2
STATE_SIZE     = 3

_VIS_END   = N_RAYS * 2           # 242
_TAC_END   = _VIS_END + N_TOUCH_BODY + N_TOUCH_PRONGS  # 260
_LIFE_IDX  = _TAC_END             # 260
_SAT_IDX   = _TAC_END + 1         # 261
_VAL_IDX   = _TAC_END + 2         # 262
OBS_SIZE   = _TAC_END + STATE_SIZE # 263

# type_norm values from env.py: hit_type / (HIT_TYPES-1) = hit_type / 4
_FOOD_TYPE_NORM   = 2 / 4   # 0.5
_DANGER_TYPE_NORM = 3 / 4   # 0.75
_TYPE_TOL         = 0.01    # tolerance for float comparison

# ── External DoFs ─────────────────────────────────────────────────────────────

food_max_proximity = ScalarDoF(
    "food_max_proximity",
    description="Max proximity of food items in the forward vision cone. 0=absent/far, 1=contact.",
    min_value=0.0,
    max_value=1.0,
)

danger_max_proximity = ScalarDoF(
    "danger_max_proximity",
    description="Max proximity of danger items in the forward vision cone. 0=absent/far, 1=contact.",
    min_value=0.0,
    max_value=1.0,
)

life = ScalarDoF(
    "life",
    description="Entity health. 1.0=full health, 0.0=dead (triggers episode reset).",
    min_value=0.0,
    max_value=1.0,
)

satiation_norm = PolarDoF(
    "satiation_norm",
    description=(
        "Hunger↔satiation axis, normalised to [0,1]. "
        "0.0=starving · 0.5=neutral · 1.0=full. "
        "Raw satiation ∈ [-1,1]; satiation_norm = (satiation+1)/2."
    ),
    pole_negative=0.0,
    pole_positive=1.0,
)

valence_norm = PolarDoF(
    "valence_norm",
    description=(
        "Pain↔pleasure axis, normalised to [0,1]. "
        "0.0=max pain · 0.5=neutral · 1.0=max pleasure. "
        "Raw valence ∈ [-1,1]; valence_norm = (valence+1)/2."
    ),
    pole_negative=0.0,
    pole_positive=1.0,
)

EXTERNAL_DOFS = [
    food_max_proximity,
    danger_max_proximity,
    life,
    satiation_norm,
    valence_norm,
]

# ── Internal DoFs — Phase 1: action outputs ───────────────────────────────────

fwd_output = PolarDoF(
    "fwd_output",
    description="Forward/reverse motor output after tanh. +1=full forward, -1=full reverse.",
    pole_negative=-1.0,
    pole_positive=1.0,
)

turn_output = PolarDoF(
    "turn_output",
    description="Turn motor output after tanh. +1=right, -1=left.",
    pole_negative=-1.0,
    pole_positive=1.0,
)

eat_output = ScalarDoF(
    "eat_output",
    description="Eat action. 1.0=eat triggered, 0.0=not triggered.",
    min_value=0.0,
    max_value=1.0,
)

INTERNAL_DOFS = [fwd_output, turn_output, eat_output]

# ── Internal DoFs — Phase 2 stubs (frequency band amplitudes) ─────────────────
# Defined here so dofs.py is the single source of truth for all DoF identities.
# Not wired until FrequencyTracker is added in Phase 2.

vis_band_power = ScalarDoF(
    "vis_band_power",
    description="[Phase 2] Mean Goertzel amplitude at visual carrier frequency across vis-band neurons.",
    min_value=0.0,
    max_value=1.0,
)

tac_band_power = ScalarDoF(
    "tac_band_power",
    description="[Phase 2] Mean Goertzel amplitude at tactile carrier frequency across tac-band neurons.",
    min_value=0.0,
    max_value=1.0,
)

val_band_power = ScalarDoF(
    "val_band_power",
    description="[Phase 2] Mean Goertzel amplitude at internal-state carrier frequency across val-band neurons.",
    min_value=0.0,
    max_value=1.0,
)

# ── State converters ──────────────────────────────────────────────────────────

def obs_to_state(obs: np.ndarray) -> State:
    """
    Convert a raw obs[263] vector into a named State with the 5 external DoFs.

    Derived features:
      food_max_proximity   — max proximity across all rays where type_norm ≈ 0.5 (food)
      danger_max_proximity — max proximity across all rays where type_norm ≈ 0.75 (danger)
      life, satiation_norm, valence_norm — read directly from obs[260:263]

    Parameters
    ----------
    obs : np.ndarray, shape (263,), dtype float32
        Raw observation vector from World.get_ai_observation().

    Returns
    -------
    State with keys: food_max_proximity, danger_max_proximity, life, satiation_norm, valence_norm
    """
    vis = obs[:_VIS_END].reshape(N_RAYS, 2)  # (121, 2): col0=type_norm, col1=proximity

    food_mask   = np.abs(vis[:, 0] - _FOOD_TYPE_NORM)   < _TYPE_TOL
    danger_mask = np.abs(vis[:, 0] - _DANGER_TYPE_NORM) < _TYPE_TOL

    food_prox   = float(vis[food_mask,   1].max()) if food_mask.any()   else 0.0
    danger_prox = float(vis[danger_mask, 1].max()) if danger_mask.any() else 0.0

    return State({
        food_max_proximity:   food_prox,
        danger_max_proximity: danger_prox,
        life:                 float(obs[_LIFE_IDX]),
        satiation_norm:       float(obs[_SAT_IDX]),
        valence_norm:         float(obs[_VAL_IDX]),
    })


def action_to_state(fwd: float, turn: float, eat: float) -> State:
    """
    Pack (fwd, turn, eat) action values into a named internal State.

    Parameters
    ----------
    fwd  : float ∈ [-1, 1]
    turn : float ∈ [-1, 1]
    eat  : float ∈ {0.0, 1.0}

    Returns
    -------
    State with keys: fwd_output, turn_output, eat_output
    """
    return State({
        fwd_output:  float(np.clip(fwd,  -1.0, 1.0)),
        turn_output: float(np.clip(turn, -1.0, 1.0)),
        eat_output:  float(np.clip(eat,   0.0, 1.0)),
    })
