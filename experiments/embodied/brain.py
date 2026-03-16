"""
experiments/embodied/brain.py — Single-reservoir ESN brain with RO Framework integration.

Architecture
------------
A single fixed-weight reservoir (SingleReservoir from reservoir.py) maps the full
263-dim observation vector to a 4096-dim hidden state.  Only the readout layer
W_out (3 × 4096) is trained via online RPE-gated eligibility traces.

RO Framework integration
------------------------
An Observer wraps the brain: EXTERNAL_DOFS = world percepts (food, danger, life,
satiation, valence); INTERNAL_DOFS = action outputs (fwd, turn, eat).
Observation pairs are appended manually to observer.observation_log after each
forward pass, bypassing observe() (which would call the world_model and validate
the result).  KnowledgeTracker computes K(d_ext) = (ρ, ε, σ, C) every ASSESS_EVERY
steps — e.g. K(food_max_proximity → fwd_output) tells us whether the brain has
learned to approach food.

Reservoir size presets
----------------------
    RES_TINY   =   512  →   1 MB    fast unit tests / CPU
    RES_SMALL  =  1024  →   4 MB    quick experiments
    RES_MEDIUM =  2187  →  18 MB    3^7, matches old sub-reservoirs
    RES_LARGE  =  4096  →  64 MB    default for RTX 4090
    RES_XL     =  8192  → 256 MB    4090 comfortable
    RES_XXL    = 16384  →   1 GB    4090 has 24 GB, fine for experiments

Usage
-----
    python brain.py                        connect to game.py --connect
    python brain.py --device cpu           force CPU (testing)
    python brain.py --headless 3600        offline headless run
    python brain.py --headless 600 --device cpu --log-every 300
    python brain.py --action-feedback      feed previous action back to reservoir
    python brain.py --res-size 8192        larger reservoir
    python brain.py --load brain.npz       resume checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch

# ── Resolve library path (running directly from experiments/embodied/) ─────────
_here     = os.path.dirname(os.path.abspath(__file__))
_repo     = os.path.normpath(os.path.join(_here, "..", ".."))
_src_path = os.path.join(_repo, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
if _here not in sys.path:
    sys.path.insert(0, _here)

from ro_framework import PolarDoF, ScalarDoF  # noqa: E402 (after sys.path fix)
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer, ObservationPair
from ro_framework.knowledge.tracker import KnowledgeTracker

import dofs  # noqa: E402
from reservoir import SingleReservoir  # noqa: E402

# ── Observation layout (must match env.py / dofs.py) ──────────────────────────
OBS_DIM    = 263
_LIFE_IDX  = 260   # obs[260:263] = [life, satiation_norm, valence_norm]

# ── Hyperparameters ────────────────────────────────────────────────────────────
RESERVOIR_SIZE  = 4096
SPECTRAL_RADIUS = 0.99
NOISE_SCALE     = 90.0
ALPHA           = 1.0
BIAS_SCALE      = 0.0   # must stay 0 — any bias → spinning attractor via W_out

EXPLORE_NOISE   = 90.0
EAT_THRESHOLD   = 0.0

LEARN_LR     = 1e-4
CRITIC_LR    = 1e-3
TRACE_DECAY  = 0.99
WEIGHT_DECAY = 1e-5

LOG_EVERY    = 300
SAVE_EVERY   = 3600
LOG_CAPACITY = 5000   # ~5 min at 60 fps; enough for stable K estimates
ASSESS_EVERY = 300    # matches LOG_EVERY — K printed alongside step log


# ── Device helper ──────────────────────────────────────────────────────────────

def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        vram  = props.total_memory / 1024 ** 3
        print(f"GPU: {props.name}  ({vram:.1f} GB VRAM)")
        return dev
    return torch.device(requested)


# ── RO Framework stubs ─────────────────────────────────────────────────────────

class _IdentityMapping:
    """Satisfies MappingFunction protocol.  Never actually called — log is
    populated manually in forward() to keep the reservoir side-effect-free."""
    def __call__(self, external_state: State) -> State:
        return external_state


class _EmptyReservoir:
    """Stub representing unused sub-reservoir slots.
    Returned by tac_res / val_res / central_res properties for brain_viz
    compatibility; guards against len(state)==0 are in brain_viz._draw_heatmaps."""
    state             = np.zeros(0, dtype=np.float32)
    recurrent_weights = np.zeros((0, 0), dtype=np.float32)
    size              = 0


# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    # Identity / paths
    "name":            "",       # display name in HUD + monitor; empty = use slot #N
    "brain_path":      "",       # save + auto-load path (.npz); empty = no persistence
    "log_path":        "",       # CSV training log path; empty = no log
    "world_config":    "",       # ruleset JSON this brain was trained on (informational;
                                 # used in headless mode to configure the World)
    # Reservoir
    "res_size":        RESERVOIR_SIZE,
    "spectral_radius": SPECTRAL_RADIUS,
    "noise_scale":     NOISE_SCALE,
    "alpha":           ALPHA,
    # Readout / learning
    "explore_noise":   EXPLORE_NOISE,
    "eat_threshold":   EAT_THRESHOLD,
    "learn_lr":        LEARN_LR,
    "critic_lr":       CRITIC_LR,
    "trace_decay":     TRACE_DECAY,
    "weight_decay":    WEIGHT_DECAY,
    # RO Framework
    "assess_every":    ASSESS_EVERY,
    "log_capacity":    LOG_CAPACITY,
    # Arch
    "seed":            42,
    "action_feedback": False,
    "decision_interval": 1,  # hold each action for N game frames (1 = every frame)
    "device":          "cuda",
}


def _config_path(npz_path: str) -> str:
    """Derive config path from .npz path: brain.npz → brain.json"""
    return os.path.splitext(npz_path)[0] + ".json"


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Strip documentation keys (_doc, _comment_*)
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        print(f"  [config] unknown keys ignored: {sorted(unknown)}")
    return {**DEFAULT_CONFIG, **cfg}


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


# ── K-metric formatter ─────────────────────────────────────────────────────────

def _format_k(assessment) -> str:
    if assessment is None:
        return "n/a  (warming up)"
    a = assessment
    return (f"ρ={a.correlation:.3f}  [{a.knowledge_type}]  "
            f"ε={a.systematic_error:.3f}  C={a.calibration:.3f}")


# ── Brain ──────────────────────────────────────────────────────────────────────

class EmbodiedBrain:
    """
    Single-reservoir ESN brain integrated with the RO Framework.

    Parameters
    ----------
    config : dict or None
        Override any of the module-level hyperparameters by key name.
    device : str
        "cuda" (default) or "cpu".
    action_feedback : bool
        If True, the previous (fwd, turn, eat) triple is concatenated to the
        obs vector before each reservoir step, making input_dim = 266.
    carrier : bool
        [Phase 2 stub] Accepted but ignored in Phase 1.  Will activate
        FrequencyTracker-based carrier wave injection when implemented.
    seed : int
        Reservoir initialisation seed.
    """

    # Used by headless run loop to detect episode resets via life index
    _state_sl = slice(_LIFE_IDX, _LIFE_IDX + 3)  # obs[260:263]

    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = "cuda",
        action_feedback: bool = False,
        carrier: bool = False,
        seed: int = 42,
    ) -> None:
        cfg = dict(
            RESERVOIR_SIZE=RESERVOIR_SIZE, SPECTRAL_RADIUS=SPECTRAL_RADIUS,
            NOISE_SCALE=NOISE_SCALE, ALPHA=ALPHA, BIAS_SCALE=BIAS_SCALE,
            EXPLORE_NOISE=EXPLORE_NOISE, EAT_THRESHOLD=EAT_THRESHOLD,
            LEARN_LR=LEARN_LR, CRITIC_LR=CRITIC_LR,
            TRACE_DECAY=TRACE_DECAY, WEIGHT_DECAY=WEIGHT_DECAY,
            LOG_CAPACITY=LOG_CAPACITY, ASSESS_EVERY=ASSESS_EVERY,
        )
        if config:
            cfg.update(config)

        self._dev             = _select_device(device)
        self._action_feedback = action_feedback
        res_size              = cfg["RESERVOIR_SIZE"]
        input_dim             = OBS_DIM + (3 if action_feedback else 0)

        # ── Reservoir ──────────────────────────────────────────────────────────
        self._reservoir = SingleReservoir(
            input_dim       = input_dim,
            size            = res_size,
            spectral_radius = cfg["SPECTRAL_RADIUS"],
            noise_scale     = cfg["NOISE_SCALE"],
            alpha           = cfg["ALPHA"],
            bias_scale      = cfg["BIAS_SCALE"],
            seed            = seed,
            device          = self._dev,
        )

        # ── Readout (trained) ──────────────────────────────────────────────────
        rng = np.random.default_rng(seed + 99)
        self.W_out = torch.from_numpy(
            rng.standard_normal((3, res_size)).astype(np.float32) * 0.01
        ).to(self._dev)
        self.b_out = torch.zeros(3, dtype=torch.float32, device=self._dev)

        # ── Online learning state ──────────────────────────────────────────────
        self._trace        = torch.zeros(3, res_size, dtype=torch.float32, device=self._dev)
        self._valence_pred = 0.0
        self._last_action  = torch.zeros(3,        dtype=torch.float32, device=self._dev)
        self._last_h       = torch.zeros(res_size, dtype=torch.float32, device=self._dev)

        self._explore_noise = cfg["EXPLORE_NOISE"]
        self._eat_threshold = cfg["EAT_THRESHOLD"]
        self._learn_lr      = cfg["LEARN_LR"]
        self._critic_lr     = cfg["CRITIC_LR"]
        self._trace_decay   = cfg["TRACE_DECAY"]
        self._weight_decay  = cfg["WEIGHT_DECAY"]
        self.learning_enabled = True
        self._last_obs: Optional[np.ndarray] = None

        # ── RO Framework — Observer + KnowledgeTracker ─────────────────────────
        self._observer = Observer(
            name          = "embodied_brain",
            internal_dofs = dofs.INTERNAL_DOFS,
            external_dofs = dofs.EXTERNAL_DOFS,
            world_model   = _IdentityMapping(),
            log_capacity  = cfg["LOG_CAPACITY"],
        )
        self._tracker = KnowledgeTracker(
            observer        = self._observer,
            external_dofs   = dofs.EXTERNAL_DOFS,
            assess_interval = 1,   # always fire when called; call frequency = log_every
            min_samples     = 50,
        )

        # ── Phase 2 stub ───────────────────────────────────────────────────────
        if carrier:
            print("[brain] --carrier flag set: Phase 2 FrequencyTracker not yet "
                  "implemented — running with plain reservoir noise.")

    # ── Forward pass ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, obs: np.ndarray) -> tuple[float, float, float]:
        """
        Advance one step.

        Parameters
        ----------
        obs : np.ndarray, shape (263,), float32
            Raw observation vector from World.get_ai_observation().

        Returns
        -------
        (fwd, turn, eat) : three floats
        """
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self._dev)
        x     = torch.cat([obs_t, self._last_action]) if self._action_feedback else obs_t

        h = self._reservoir.step(x)
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

        # ── Append to observer log (manual — bypasses world_model) ─────────────
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

    # ── Episode reset ───────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Zero reservoir hidden state, eligibility trace, and last action.
        Does NOT reset valence_pred (critic baseline survives resets)."""
        self._reservoir.reset()
        self._trace.zero_()
        self._last_action.zero_()
        self._last_h.zero_()

    # ── Knowledge tracker ───────────────────────────────────────────────────────

    def step_knowledge(self, epoch: int) -> dict:
        """Advance KnowledgeTracker one step.  Returns assessment dict or {}."""
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

    # brain_viz compatibility: single reservoir maps to both v1 and motor slots
    @property
    def v1_res(self) -> SingleReservoir:
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
    def motor_res(self) -> SingleReservoir:
        return self._reservoir

    @property
    def _last_h_motor(self) -> torch.Tensor:
        """brain_viz compatibility alias for _last_h."""
        return self._last_h


# ── Logging helpers ─────────────────────────────────────────────────────────────

_K_DOF_NAMES  = [d.name for d in dofs.EXTERNAL_DOFS]   # fixed column order for CSV
_EXTRA_K_FIELDS = [
    "K_sat_eat",        # K(satiation_norm → eat_output)        — hunger drives eating?
    "K_food_eat_cond",  # K(food_prox → eat_output | food visible) — react when food present?
    "K_food_eat_lag2",  # K(food_prox[t] → eat_output[t+2])    — delayed eat response?
    "K_fwd_autocorr",   # ρ(fwd[t], fwd[t+1])                  — motor persistence / attractor
    "K_turn_autocorr",  # ρ(turn[t], turn[t+1])                 — spiral vs chaotic?
]

def _open_log(path: str) -> csv.DictWriter:
    k_fields = [f"K_{n}" for n in _K_DOF_NAMES]
    fields   = ["timestamp", "step", "mean_reward", "valence_pred", "w_norm",
                "eat_count", "episodes", "mean_fwd", "mean_turn"] + k_fields + _EXTRA_K_FIELDS
    is_new = not os.path.exists(path)
    fh     = open(path, "a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return writer


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson ρ; returns 0 if either array is near-constant."""
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _compute_extra_k(brain: "EmbodiedBrain") -> dict:
    """Compute extra K variants directly from the rolling observation log.

    All four measures use the same LOG_CAPACITY window as the standard K tracker.
    Returns a dict with keys matching _EXTRA_K_FIELDS; empty dict if <50 pairs.
    """
    pairs = list(brain._observer.observation_log._pairs)
    if len(pairs) < 50:
        return {}

    food = np.array([p.external_state.get_value(dofs.food_max_proximity) for p in pairs], dtype=np.float32)
    sat  = np.array([p.external_state.get_value(dofs.satiation_norm)      for p in pairs], dtype=np.float32)
    eat  = np.array([p.internal_state.get_value(dofs.eat_output)          for p in pairs], dtype=np.float32)
    fwd  = np.array([p.internal_state.get_value(dofs.fwd_output)          for p in pairs], dtype=np.float32)
    turn = np.array([p.internal_state.get_value(dofs.turn_output)         for p in pairs], dtype=np.float32)

    # Conditional K — only timesteps where food is actually in FOV
    vis = food > 0.05
    k_food_eat_cond = _rho(food[vis], eat[vis]) if vis.sum() >= 30 else ""

    lag = 2
    return {
        "K_sat_eat":       f"{_rho(sat, eat):.4f}",
        "K_food_eat_cond": f"{k_food_eat_cond:.4f}" if k_food_eat_cond != "" else "",
        "K_food_eat_lag2": f"{_rho(food[:-lag], eat[lag:]):.4f}",
        "K_fwd_autocorr":  f"{_rho(fwd[:-1],  fwd[1:]):.4f}",
        "K_turn_autocorr": f"{_rho(turn[:-1], turn[1:]):.4f}",
    }


def _log(step, reward_sum, log_every, valence_pred, w_norm,
         eat_count, episodes=0, fwd_sum=0.0, turn_sum=0.0) -> dict:
    """Print the step summary and return the CSV row dict (K fields added later)."""
    mean_r    = reward_sum / max(log_every, 1)
    mean_fwd  = fwd_sum   / max(log_every, 1)
    mean_turn = turn_sum  / max(log_every, 1)
    print(f"step {step:>7}  reward {mean_r:+.3f}  valence_pred {valence_pred:+.3f}  "
          f"|W_out| {w_norm:.4f}  fwd:{mean_fwd:+.2f}  turn:{mean_turn:+.2f}  "
          f"eat:{eat_count}  ep:{episodes}")
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step, "mean_reward": f"{mean_r:.4f}",
        "valence_pred": f"{valence_pred:.4f}", "w_norm": f"{w_norm:.4f}",
        "eat_count": eat_count, "episodes": episodes,
        "mean_fwd": f"{mean_fwd:.4f}", "mean_turn": f"{mean_turn:.4f}",
    }


def _log_step(brain: EmbodiedBrain, step: int, reward_sum: float, log_every: int,
              eat_count: int, episodes: int, fwd_sum: float, turn_sum: float,
              csv_writer=None) -> None:
    """Print step summary + K metrics and write one CSV row."""
    row = _log(step, reward_sum, log_every, brain._valence_pred,
               brain.w_out_norm, eat_count, episodes, fwd_sum, turn_sum)
    _log_knowledge(brain, step, csv_row=row)
    row.update(_compute_extra_k(brain))
    if csv_writer is not None:
        csv_writer.writerow(row)


def _log_knowledge(brain: EmbodiedBrain, step: int,
                   csv_row: dict | None = None) -> None:
    """Print K(d_ext) metrics and optionally write them into csv_row.

    csv_row is the dict that _log() will write; K values are added in-place
    so the main log row and K values land on the same CSV line.
    """
    results = brain.step_knowledge(epoch=step)
    if not results:
        return
    for ext_dof, assessment in results.items():
        k_str = _format_k(assessment)
        print(f"  K({ext_dof.name}): {k_str}")
        if csv_row is not None:
            rho = f"{assessment.correlation:.4f}" if assessment is not None else ""
            csv_row[f"K_{ext_dof.name}"] = rho


def _recv_with_retry(client, zmq_again_cls):
    while True:
        try:
            return client.recv_obs()
        except Exception as exc:
            if zmq_again_cls and isinstance(exc, zmq_again_cls):
                print("[brain] recv timeout — retrying")
            else:
                raise


# ── Run loops ───────────────────────────────────────────────────────────────────

def run_connected(brain: EmbodiedBrain, args: argparse.Namespace) -> None:
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from connector import AgentConnector  # noqa: PLC0415

    try:
        import zmq
        _zmq_again = zmq.Again
    except ImportError:
        _zmq_again = None

    if args.save:
        def _handler(*_):
            brain.save(args.save)
            print(f"\n[SIGUSR1] Saved → {args.save}  (PID {os.getpid()})")
        signal.signal(signal.SIGUSR1, _handler)
        print(f"  Save on demand: kill -USR1 {os.getpid()}")

    client     = AgentConnector()
    client.connect(name=getattr(args, "_brain_name", ""))
    csv_writer = _open_log(args.log_path) if args.log_path else None

    decision_interval = getattr(args, "_decision_interval", 1)
    step       = 0
    eat_count  = 0
    episodes   = 0
    reward_sum = 0.0
    fwd_sum    = 0.0
    turn_sum   = 0.0
    fwd = turn = eat = 0.0

    _frame_dt = 1.0 / 60.0   # target one action per game frame
    try:
        obs, reward, done, _ = client.recv_obs()
        _t = time.monotonic()
        while True:
            if step % decision_interval == 0:
                fwd, turn, eat = brain.forward(obs)
            brain.learn(reward)
            if done:
                brain.reset_state()
                episodes += 1
            client.send_action((fwd, turn, eat))

            reward_sum += reward
            eat_count  += int(eat > 0.5)
            fwd_sum    += fwd
            turn_sum   += turn
            step       += 1

            if step % args.log_every == 0:
                _log_step(brain, step, reward_sum, args.log_every,
                          eat_count, episodes, fwd_sum, turn_sum, csv_writer)
                reward_sum = eat_count = episodes = 0
                fwd_sum    = turn_sum  = 0.0

            if args.save and step % args.save_every == 0:
                brain.save(args.save)
                print(f"  [saved → {args.save}]")

            # Pace to ~60fps so actions don't pile up in the game's receive buffer
            _now = time.monotonic()
            _remaining = _frame_dt - (_now - _t)
            if _remaining > 0:
                time.sleep(_remaining)
            _t = time.monotonic()

            obs, reward, done, _ = _recv_with_retry(client, _zmq_again)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.save:
            brain.save(args.save)
            print(f"Saved → {args.save}")
        client.close()


def _load_world_config(path: str) -> dict:
    """Load a world ruleset JSON, stripping _comment_* keys."""
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def run_headless(brain: EmbodiedBrain, args: argparse.Namespace) -> None:
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from env import World  # noqa: PLC0415

    world_cfg  = _load_world_config(args._world_config) if getattr(args, "_world_config", None) else {}
    world      = World(seed=getattr(args, "seed", 42), no_reset_on_death=args.no_reset, cfg=world_cfg)
    world.add_agent()   # headless mode bypasses registration; spawn slot 0 directly
    csv_writer = _open_log(args.log_path) if args.log_path else None
    decision_interval = getattr(args, "_decision_interval", 1)
    n_steps    = args.headless
    step       = 0
    eat_count  = 0
    episodes   = 0
    reward_sum = 0.0
    fwd_sum    = 0.0
    turn_sum   = 0.0
    prev_life  = 1.0
    fwd = turn = eat = 0.0

    try:
        while step < n_steps:
            obs    = world.get_ai_observation()
            reward = float(world.ai.meters.valence)
            life   = float(obs[_LIFE_IDX])
            # Detect episode boundary: life jumps from near-0 to near-1 after reset.
            # With --no-reset life stays at 0, so there are no episode boundaries.
            done   = (not args.no_reset) and bool(life > 0.9 and prev_life < 0.1)
            prev_life = life

            if step % decision_interval == 0:
                fwd, turn, eat = brain.forward(obs)
            brain.learn(reward)
            if done:
                brain.reset_state()
                episodes += 1
            world.step(ai_action=(fwd, turn, eat))

            reward_sum += reward
            eat_count  += int(eat > 0.5)
            fwd_sum    += fwd
            turn_sum   += turn
            step       += 1

            if step % args.log_every == 0:
                _log_step(brain, step, reward_sum, args.log_every,
                          eat_count, episodes, fwd_sum, turn_sum, csv_writer)
                reward_sum = eat_count = episodes = 0
                fwd_sum    = turn_sum  = 0.0

            if args.save and step % args.save_every == 0:
                brain.save(args.save)
                print(f"  [saved → {args.save}]")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.save:
            brain.save(args.save)
            print(f"Saved → {args.save}")


# ── CLI helpers ─────────────────────────────────────────────────────────────────

def _resolve_config(args) -> dict:
    """Load config file; auto-discover alongside --load if --config not given."""
    cfg_path = args.config
    if cfg_path is None and args.load:
        auto = _config_path(args.load)
        if os.path.exists(auto):
            cfg_path = auto
            print(f"  [config] auto-loaded from {auto}")
    cfg = load_config(cfg_path) if cfg_path else dict(DEFAULT_CONFIG)
    if args.device:
        cfg["device"] = args.device
    return cfg


def _resolve_paths(args, cfg) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (brain_path, log_path, load_path).

    CLI flags override config; auto-load fires when brain_path exists on disk
    and --load was not explicitly set.
    """
    brain_path = args.save     or cfg["brain_path"] or None
    log_path   = args.log_path or cfg["log_path"]   or None
    load_path  = args.load
    if load_path is None and brain_path and os.path.exists(brain_path):
        load_path = brain_path
    return brain_path, log_path, load_path


def _build_brain(cfg: dict, carrier: bool) -> "EmbodiedBrain":
    return EmbodiedBrain(
        config = {
            "RESERVOIR_SIZE":  cfg["res_size"],
            "SPECTRAL_RADIUS": cfg["spectral_radius"],
            "NOISE_SCALE":     cfg["noise_scale"],
            "ALPHA":           cfg["alpha"],
            "EXPLORE_NOISE":   cfg["explore_noise"],
            "EAT_THRESHOLD":   cfg["eat_threshold"],
            "LEARN_LR":        cfg["learn_lr"],
            "CRITIC_LR":       cfg["critic_lr"],
            "TRACE_DECAY":     cfg["trace_decay"],
            "WEIGHT_DECAY":    cfg["weight_decay"],
            "ASSESS_EVERY":    cfg["assess_every"],
            "LOG_CAPACITY":    cfg["log_capacity"],
        },
        device          = cfg["device"],
        action_feedback = cfg["action_feedback"],
        carrier         = carrier,
        seed            = cfg["seed"],
    )


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-reservoir ESN brain with RO Framework integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hyperparameters are set via a JSON config file (--config).
If --load is given and a matching .json exists alongside the .npz, it is
loaded automatically — no need to repeat --config on resume.

examples:
  python brain.py --config brains/configs/bob-16k.json --save brains/Bob-16k.npz
  python brain.py --load brains/Bob-16k.npz --save brains/Bob-16k.npz   # auto-loads .json
  python brain.py --config brains/configs/bob-16k.json --device cpu --headless 3600
  python brain.py --no-learn --load brains/Bob-16k.npz                  # inference only
""")
    parser.add_argument("--config",     metavar="PATH",
                        help="JSON config file (see brains/configs/config-template.json)")
    parser.add_argument("--device",    choices=["cuda", "cpu"],
                        help="Override device from config")
    parser.add_argument("--no-learn",  action="store_true",
                        help="Freeze W_out (inference only)")
    parser.add_argument("--carrier",   action="store_true",
                        help="[Phase 2 stub] Carrier wave scaffold — no-op in Phase 1")
    parser.add_argument("--save",      metavar="PATH", default=None,
                        help="Override brain_path from config for this run")
    parser.add_argument("--load",      metavar="PATH", default=None,
                        help="Force-load weights from PATH (skips auto-load from config)")
    parser.add_argument("--log-path",  metavar="PATH", default=None,
                        help="Override log_path from config for this run")
    parser.add_argument("--headless",  type=int, default=0, metavar="N",
                        help="Run N headless steps then exit (0 = connect to game)")
    parser.add_argument("--no-reset",  action="store_true",
                        help="Headless: keep AI alive at life=0/valence=-1 on death")
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY,  metavar="N")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N")
    args = parser.parse_args()

    cfg                        = _resolve_config(args)
    brain_path, log_path, load_path = _resolve_paths(args, cfg)

    # Keep args compatible with run_connected / run_headless
    args.save     = brain_path
    args.log_path = log_path

    print(f"Building EmbodiedBrain  (res_size={cfg['res_size']}, "
          f"action_feedback={cfg['action_feedback']})…")

    brain = _build_brain(cfg, args.carrier)

    print(f"  Device:        {brain._dev}")
    print(f"  input_dim:     {OBS_DIM + (3 if cfg['action_feedback'] else 0)}")
    print(f"  res_size:      {cfg['res_size']}")
    print(f"  noise_scale:   {cfg['noise_scale']}")
    print(f"  explore_noise: {cfg['explore_noise']}")
    print(f"  alpha:         {cfg['alpha']}")
    print(f"  W_out:         {tuple(brain.W_out.shape)}  |norm|={brain.w_out_norm:.5f}")
    di = cfg.get('decision_interval', 1)
    print(f"  decision_int:  {di}  {'(new action every frame)' if di <= 1 else f'(hold each action for {di} frames)'}")
    print(f"  K assess:      every {cfg['assess_every']} steps  "
          f"(log capacity {cfg['log_capacity']}, min_samples=50)")

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

    args._brain_name         = cfg.get("name") or ""
    args._world_config       = cfg.get("world_config") or None
    args._decision_interval  = cfg.get("decision_interval", 1)

    if args.headless:
        print(f"\nRunning {args.headless} headless steps…")
        run_headless(brain, args)
    else:
        print("\nWaiting for game connection…")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
