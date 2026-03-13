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
import os
import signal
import sys
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
NOISE_SCALE     = 0.9
ALPHA           = 1.0
BIAS_SCALE      = 0.0   # must stay 0 — any bias → spinning attractor via W_out

EXPLORE_NOISE   = 0.9
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

_K_DOF_NAMES = [d.name for d in dofs.EXTERNAL_DOFS]   # fixed column order for CSV

def _open_log(path: str) -> csv.DictWriter:
    k_fields = [f"K_{n}" for n in _K_DOF_NAMES]
    fields   = ["step", "mean_reward", "valence_pred", "w_norm",
                "eat_count", "episodes", "mean_fwd", "mean_turn"] + k_fields
    is_new = not os.path.exists(path)
    fh     = open(path, "a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return writer


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
    client.connect()
    csv_writer = _open_log(args.log_path) if args.log_path else None

    step       = 0
    eat_count  = 0
    episodes   = 0
    reward_sum = 0.0
    fwd_sum    = 0.0
    turn_sum   = 0.0

    try:
        obs, reward, done, _ = client.recv_obs()
        while True:
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

            obs, reward, done, _ = _recv_with_retry(client, _zmq_again)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.save:
            brain.save(args.save)
            print(f"Saved → {args.save}")
        client.close()


def run_headless(brain: EmbodiedBrain, args: argparse.Namespace) -> None:
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from env import World  # noqa: PLC0415

    world      = World(seed=args.seed)
    csv_writer = _open_log(args.log_path) if args.log_path else None
    n_steps    = args.headless
    step       = 0
    eat_count  = 0
    episodes   = 0
    reward_sum = 0.0
    fwd_sum    = 0.0
    turn_sum   = 0.0
    prev_life  = 1.0

    try:
        while step < n_steps:
            obs    = world.get_ai_observation()
            reward = float(world.ai.meters.valence)
            life   = float(obs[_LIFE_IDX])
            done   = bool(life > 0.9 and prev_life < 0.1)
            prev_life = life

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


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-reservoir ESN brain with RO Framework integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python brain.py                             connect to game.py --connect
  python brain.py --device cpu               force CPU (testing)
  python brain.py --headless 3600            offline headless run
  python brain.py --headless 600 --device cpu --log-every 300
  python brain.py --action-feedback          input_dim = 266
  python brain.py --res-size 8192            larger reservoir
  python brain.py --load brain.npz           resume checkpoint
  python brain.py --headless 3600 --save brain.npz
""")
    parser.add_argument("--device",          default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--action-feedback", action="store_true",
                        help="Feed previous action back into reservoir (input_dim += 3)")
    parser.add_argument("--carrier",         action="store_true",
                        help="[Phase 2 stub] Carrier wave scaffold — no-op in Phase 1")
    parser.add_argument("--res-size",        type=int, default=RESERVOIR_SIZE, metavar="N",
                        help=f"Reservoir size (default: {RESERVOIR_SIZE})")
    parser.add_argument("--assess-every",    type=int, default=ASSESS_EVERY, metavar="N",
                        help=f"K assessment interval in steps (default: {ASSESS_EVERY})")
    parser.add_argument("--no-learn",        action="store_true",
                        help="Freeze W_out (inference only)")
    parser.add_argument("--save",            metavar="PATH")
    parser.add_argument("--load",            metavar="PATH")
    parser.add_argument("--headless",        type=int, default=0, metavar="N",
                        help="Run N headless steps then exit (0 = connect to game)")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--log-every",       type=int, default=LOG_EVERY,  metavar="N")
    parser.add_argument("--save-every",      type=int, default=SAVE_EVERY, metavar="N")
    parser.add_argument("--log-path",        metavar="PATH")
    args = parser.parse_args()

    print(f"Building EmbodiedBrain  (res_size={args.res_size}, "
          f"action_feedback={args.action_feedback})…")

    brain = EmbodiedBrain(
        config          = {"RESERVOIR_SIZE": args.res_size,
                           "ASSESS_EVERY":   args.assess_every},
        device          = args.device,
        action_feedback = args.action_feedback,
        carrier         = args.carrier,
        seed            = args.seed,
    )

    print(f"  Device:     {brain._dev}")
    print(f"  input_dim:  {OBS_DIM + (3 if args.action_feedback else 0)}")
    print(f"  res_size:   {args.res_size}")
    print(f"  W_out:      {tuple(brain.W_out.shape)}  |norm|={brain.w_out_norm:.5f}")
    print(f"  K assess:   every {args.assess_every} steps  "
          f"(log capacity {LOG_CAPACITY}, min_samples=50)")

    if args.load:
        brain.load(args.load)
        print(f"  Loaded weights from {args.load}  |W_out|={brain.w_out_norm:.5f}")

    if args.no_learn:
        brain.learning_enabled = False
        print("  Learning disabled.")

    if args.headless:
        print(f"\nRunning {args.headless} headless steps…")
        run_headless(brain, args)
    else:
        print("\nWaiting for game connection…")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
