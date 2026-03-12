"""
experiments/embodied/brain_gpu.py — GPU-accelerated ESN Brain (PyTorch)

Same architecture and learning rule as brain.py, but:
  • All reservoir tensors live on CUDA (float32 for 2× throughput vs float64)
  • 4× larger default reservoir sizes — at this scale GPU matmuls dominate CPU
  • `--device cpu` falls back gracefully for testing without a GPU

Architecture (same topology, larger reservoirs):

    vision(242)  →  [V1_res,     1024]  ─┐
    tactile(18)  →  [tac_res,     512]  ─┤ concat(1792) → [central_res, 2048]
    state(3)     →  [val_res,     256]  ─┘               → [motor_res,  1024]
                                                          → W_out (3×1024)
                                                          → (fwd, turn, eat)

GPU win vs brain.py:
  • Matrix-vector sizes now 1024–2048: GPU kernel launch overhead becomes negligible
  • float32 arithmetic: 2× throughput over float64 on any CUDA device
  • Spring layout in brain_viz.py scales quadratically — at n=1024 CPU takes ~10×
    longer; GPU completes in a few seconds

Weights are NOT compatible with brain.py saves (different MOTOR_SIZE).
Use --save / --load with .npz files created by this script only.

Usage:
    python brain_gpu.py                          connect to game.py --connect
    python brain_gpu.py --device cpu             run on CPU (for testing)
    python brain_gpu.py --headless 3600          offline headless run
    python brain_gpu.py --load brain_gpu.npz     resume checkpoint
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys

import numpy as np
import torch

# ── Device selection ───────────────────────────────────────────────────────────

def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        vram  = props.total_memory / 1024 ** 3
        print(f"GPU: {props.name}  ({vram:.1f} GB VRAM)")
        return dev
    return torch.device(requested)


# ── Sensor counts (same as brain.py — must match the environment) ─────────────
N_RAYS         = 121
N_TOUCH_BODY   = 16
N_TOUCH_PRONGS = 2
STATE_SIZE     = 3

_VIS_SIZE    = N_RAYS * 2
_TAC_SIZE_IN = N_TOUCH_BODY + N_TOUCH_PRONGS
VIS_END      = _VIS_SIZE
TAC_END      = VIS_END + _TAC_SIZE_IN
STATE_END    = TAC_END + STATE_SIZE
OBS_SIZE     = STATE_END   # 263

# ── Reservoir sizes — 4× brain.py defaults, sized for GPU throughput ──────────
V1_SIZE      = 729    # was 256
TAC_SIZE     = 729     # was 128
VAL_SIZE     = 729     # was 64
CENTRAL_SIZE = 2187    # was 512
MOTOR_SIZE   = 729    # was 256

# ── Dynamics ───────────────────────────────────────────────────────────────────
SPECTRAL_RADIUS     = 0.99
VAL_SPECTRAL_RADIUS = 0.85

V1_NOISE      = 0.9
TAC_NOISE     = 0.9
VAL_NOISE     = 0.005
CENTRAL_NOISE = 0.9
MOTOR_NOISE   = 0.9

V1_INPUT_SCALING      = 1.0
TAC_INPUT_SCALING     = 1.0
VAL_INPUT_SCALING     = 1.0
CENTRAL_INPUT_SCALING = 1.0
MOTOR_INPUT_SCALING   = 1.0

V1_BIAS_SCALE      = 0.0
TAC_BIAS_SCALE     = 0.0
VAL_BIAS_SCALE     = 0.0
CENTRAL_BIAS_SCALE = 0.0
MOTOR_BIAS_SCALE   = 0.0

V1_ALPHA      = 1.0
TAC_ALPHA     = 1.0
VAL_ALPHA     = 1.0
CENTRAL_ALPHA = 1.0
MOTOR_ALPHA   = 1.0

EXPLORE_NOISE = 0.2
EAT_THRESHOLD = 0.0

# ── Learning ───────────────────────────────────────────────────────────────────
LEARN_LR     = 1e-4
CRITIC_LR    = 1e-3
TRACE_DECAY  = 0.99
WEIGHT_DECAY = 1e-5

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_EVERY  = 300
SAVE_EVERY = 3600


# ── Reservoir ─────────────────────────────────────────────────────────────────

class Reservoir:
    """
    Single fixed-weight ESN reservoir backed by PyTorch tensors.

    Weights are initialised with numpy (spectral radius needs np.linalg.eigvals),
    then moved to the target device as float32 and never touched again.

    h_t = (1−α)·h_{t-1} + α·tanh(W_in @ x_t + W_res @ h_{t-1} + bias + noise)
    """

    def __init__(
        self,
        input_dim: int,
        size: int,
        spectral_radius: float = SPECTRAL_RADIUS,
        input_scaling: float = 1.0,
        noise_scale: float = 0.01,
        alpha: float = 1.0,
        bias_scale: float = 0.0,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self._size        = size
        self._noise_scale = noise_scale
        self._alpha       = alpha
        self._dev         = device or torch.device("cpu")

        # Initialise weights deterministically with numpy
        init_rng = np.random.default_rng(seed)
        w_in_np  = (init_rng.standard_normal((size, input_dim)) * input_scaling).astype(np.float32)
        bias_np  = (init_rng.standard_normal(size) * bias_scale).astype(np.float32)

        w_res_np = init_rng.standard_normal((size, size)).astype(np.float32)
        eigvals  = np.linalg.eigvals(w_res_np.astype(np.float64))   # eigvals needs float64
        cur_sr   = float(np.max(np.abs(eigvals)))
        if cur_sr > 0:
            w_res_np *= float(spectral_radius) / cur_sr
        self._w_res_np = w_res_np.copy()   # kept as numpy for brain_viz spring layout

        # Move to device
        self._W_in  = torch.from_numpy(w_in_np).to(self._dev)
        self._W_res = torch.from_numpy(w_res_np).to(self._dev)
        self._bias  = torch.from_numpy(bias_np).to(self._dev)
        self._h     = torch.zeros(size, dtype=torch.float32, device=self._dev)

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Advance one step. x must already be a float32 tensor on self._dev."""
        noise = torch.randn(self._size, dtype=torch.float32, device=self._dev) * self._noise_scale
        pre   = torch.tanh(self._W_in @ x + self._W_res @ self._h + self._bias + noise)
        self._h = (1.0 - self._alpha) * self._h + self._alpha * pre
        return self._h

    def reset(self) -> None:
        self._h.zero_()

    @property
    def state(self) -> np.ndarray:
        """Current hidden state as numpy float32, for visualisation."""
        return self._h.cpu().numpy()

    @property
    def recurrent_weights(self) -> np.ndarray:
        """Fixed W_res as numpy (for spring layout in brain_viz)."""
        return self._w_res_np


# ── Brain ──────────────────────────────────────────────────────────────────────

class EmbodiedBrainGPU:
    """
    GPU-accelerated multi-reservoir ESN brain.

    Identical learning rule and external API to EmbodiedBrain (brain.py).
    Only W_out / b_out / trace are trained; all reservoir weights are fixed.
    """

    def __init__(self, config: dict | None = None, device: str = "cuda") -> None:
        cfg  = _build_cfg(config)
        seed = cfg.get("seed", 42)
        self._dev = _select_device(device)

        vis_sz = cfg["N_RAYS"] * 2
        tac_sz = cfg["N_TOUCH_BODY"] + cfg["N_TOUCH_PRONGS"]
        st_sz  = cfg["STATE_SIZE"]
        self._vis_sl   = slice(0,              vis_sz)
        self._tac_sl   = slice(vis_sz,         vis_sz + tac_sz)
        self._state_sl = slice(vis_sz + tac_sz, vis_sz + tac_sz + st_sz)

        v1_sz  = cfg["V1_SIZE"]
        tac_rs = cfg["TAC_SIZE"]
        val_sz = cfg["VAL_SIZE"]
        cen_sz = cfg["CENTRAL_SIZE"]
        mot_sz = cfg["MOTOR_SIZE"]
        kw     = {"device": self._dev}

        self.v1_res = Reservoir(
            vis_sz, v1_sz, spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["V1_INPUT_SCALING"], noise_scale=cfg["V1_NOISE"],
            alpha=cfg["V1_ALPHA"], bias_scale=cfg["V1_BIAS_SCALE"], seed=seed, **kw)
        self.tac_res = Reservoir(
            tac_sz, tac_rs, spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["TAC_INPUT_SCALING"], noise_scale=cfg["TAC_NOISE"],
            alpha=cfg["TAC_ALPHA"], bias_scale=cfg["TAC_BIAS_SCALE"], seed=seed + 1, **kw)
        self.val_res = Reservoir(
            st_sz, val_sz, spectral_radius=cfg["VAL_SPECTRAL_RADIUS"],
            input_scaling=cfg["VAL_INPUT_SCALING"], noise_scale=cfg["VAL_NOISE"],
            alpha=cfg["VAL_ALPHA"], bias_scale=cfg["VAL_BIAS_SCALE"], seed=seed + 2, **kw)
        self.central_res = Reservoir(
            v1_sz + tac_rs + val_sz, cen_sz, spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["CENTRAL_INPUT_SCALING"], noise_scale=cfg["CENTRAL_NOISE"],
            alpha=cfg["CENTRAL_ALPHA"], bias_scale=cfg["CENTRAL_BIAS_SCALE"], seed=seed + 3, **kw)
        self.motor_res = Reservoir(
            cen_sz, mot_sz, spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["MOTOR_INPUT_SCALING"], noise_scale=cfg["MOTOR_NOISE"],
            alpha=cfg["MOTOR_ALPHA"], bias_scale=cfg["MOTOR_BIAS_SCALE"], seed=seed + 4, **kw)

        # Readout — trained on device
        rng = np.random.default_rng(seed + 99)
        self.W_out = torch.from_numpy(
            rng.standard_normal((3, mot_sz)).astype(np.float32) * 0.01
        ).to(self._dev)
        self.b_out = torch.zeros(3, dtype=torch.float32, device=self._dev)

        # Online learning state (on device for fast outer product)
        self._trace        = torch.zeros(3, mot_sz, dtype=torch.float32, device=self._dev)
        self._valence_pred = 0.0
        self._last_action  = torch.zeros(3,      dtype=torch.float32, device=self._dev)
        self._last_h_motor = torch.zeros(mot_sz, dtype=torch.float32, device=self._dev)

        # Scalar config
        self._explore_noise   = cfg["EXPLORE_NOISE"]
        self._eat_threshold   = cfg["EAT_THRESHOLD"]
        self._learn_lr        = cfg["LEARN_LR"]
        self._critic_lr       = cfg["CRITIC_LR"]
        self._trace_decay     = cfg["TRACE_DECAY"]
        self._weight_decay    = cfg["WEIGHT_DECAY"]
        self.learning_enabled = True

    # ── Forward pass ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, obs: np.ndarray) -> tuple[float, float, float]:
        # Upload obs once per step — 263 floats is negligible transfer
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self._dev)

        h_v1  = self.v1_res.step(obs_t[self._vis_sl])
        h_tac = self.tac_res.step(obs_t[self._tac_sl])
        h_val = self.val_res.step(obs_t[self._state_sl])

        h_central = self.central_res.step(torch.cat([h_v1, h_tac, h_val]))
        h_motor   = self.motor_res.step(h_central)

        raw = self.W_out @ h_motor + self.b_out
        self._last_h_motor.copy_(h_motor)

        # Exploration noise + nonlinearities
        noise = torch.randn(2, dtype=torch.float32, device=self._dev) * self._explore_noise
        fwd   = float(torch.tanh(raw[0] + noise[0]))
        turn  = float(torch.tanh(raw[1] + noise[1]))
        eat   = 1.0 if float(raw[2]) > self._eat_threshold else 0.0

        self._last_action[0] = fwd
        self._last_action[1] = turn
        self._last_action[2] = eat
        return fwd, turn, eat

    # ── Online learning ────────────────────────────────────────────────────────

    @torch.no_grad()
    def learn(self, reward: float) -> None:
        if not self.learning_enabled:
            return
        rpe = reward - self._valence_pred
        self._valence_pred += self._critic_lr * rpe
        self._trace.mul_(self._trace_decay).add_(
            torch.outer(self._last_action, self._last_h_motor)
        )
        self.W_out.add_(self._trace, alpha=self._learn_lr * rpe)
        self.W_out.mul_(1.0 - self._weight_decay)

    # ── Episode reset ──────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        for res in (self.v1_res, self.tac_res, self.val_res,
                    self.central_res, self.motor_res):
            res.reset()
        self._trace.zero_()
        self._last_action.zero_()
        self._last_h_motor.zero_()

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path,
                 W_out=self.W_out.cpu().numpy(),
                 b_out=self.b_out.cpu().numpy(),
                 valence_pred=np.array(self._valence_pred))

    def load(self, path: str) -> None:
        data = np.load(path)
        self.W_out.copy_(torch.from_numpy(data["W_out"].astype(np.float32)))
        self.b_out.copy_(torch.from_numpy(data["b_out"].astype(np.float32)))
        self._valence_pred = float(data["valence_pred"])

    # ── Helpers for logging ────────────────────────────────────────────────────

    @property
    def w_out_norm(self) -> float:
        return float(torch.linalg.norm(self.W_out))


# ── Config helper (mirrors brain.py) ──────────────────────────────────────────

def _build_cfg(user: dict | None) -> dict:
    defaults = dict(
        N_RAYS=N_RAYS, N_TOUCH_BODY=N_TOUCH_BODY,
        N_TOUCH_PRONGS=N_TOUCH_PRONGS, STATE_SIZE=STATE_SIZE,
        V1_SIZE=V1_SIZE, TAC_SIZE=TAC_SIZE, VAL_SIZE=VAL_SIZE,
        CENTRAL_SIZE=CENTRAL_SIZE, MOTOR_SIZE=MOTOR_SIZE,
        SPECTRAL_RADIUS=SPECTRAL_RADIUS, VAL_SPECTRAL_RADIUS=VAL_SPECTRAL_RADIUS,
        V1_NOISE=V1_NOISE, TAC_NOISE=TAC_NOISE, VAL_NOISE=VAL_NOISE,
        CENTRAL_NOISE=CENTRAL_NOISE, MOTOR_NOISE=MOTOR_NOISE,
        V1_INPUT_SCALING=V1_INPUT_SCALING, TAC_INPUT_SCALING=TAC_INPUT_SCALING,
        VAL_INPUT_SCALING=VAL_INPUT_SCALING, CENTRAL_INPUT_SCALING=CENTRAL_INPUT_SCALING,
        MOTOR_INPUT_SCALING=MOTOR_INPUT_SCALING,
        V1_ALPHA=V1_ALPHA, TAC_ALPHA=TAC_ALPHA, VAL_ALPHA=VAL_ALPHA,
        CENTRAL_ALPHA=CENTRAL_ALPHA, MOTOR_ALPHA=MOTOR_ALPHA,
        V1_BIAS_SCALE=V1_BIAS_SCALE, TAC_BIAS_SCALE=TAC_BIAS_SCALE,
        VAL_BIAS_SCALE=VAL_BIAS_SCALE, CENTRAL_BIAS_SCALE=CENTRAL_BIAS_SCALE,
        MOTOR_BIAS_SCALE=MOTOR_BIAS_SCALE,
        EXPLORE_NOISE=EXPLORE_NOISE, EAT_THRESHOLD=EAT_THRESHOLD,
        LEARN_LR=LEARN_LR, CRITIC_LR=CRITIC_LR,
        TRACE_DECAY=TRACE_DECAY, WEIGHT_DECAY=WEIGHT_DECAY,
    )
    if user:
        defaults.update(user)
    return defaults


# ── Logging (shared format with brain.py) ─────────────────────────────────────

def _open_log(path: str) -> csv.DictWriter:
    fields = ["step", "mean_reward", "valence_pred", "w_norm", "eat_count", "episodes",
              "mean_fwd", "mean_turn"]
    is_new = not os.path.exists(path)
    fh = open(path, "a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fields)
    if is_new:
        writer.writeheader()
    return writer


def _log(step, reward_sum, log_every, valence_pred, w_norm,
         eat_count, episodes=0, csv_writer=None,
         fwd_sum=0.0, turn_sum=0.0) -> None:
    mean_r    = reward_sum / max(log_every, 1)
    mean_fwd  = fwd_sum   / max(log_every, 1)
    mean_turn = turn_sum  / max(log_every, 1)
    print(f"step {step:>7}  reward {mean_r:+.3f}  valence_pred {valence_pred:+.3f}  "
          f"|W_out| {w_norm:.4f}  fwd:{mean_fwd:+.2f}  turn:{mean_turn:+.2f}  "
          f"eat:{eat_count}  ep:{episodes}")
    if csv_writer is not None:
        csv_writer.writerow({
            "step": step, "mean_reward": f"{mean_r:.4f}",
            "valence_pred": f"{valence_pred:.4f}", "w_norm": f"{w_norm:.4f}",
            "eat_count": eat_count, "episodes": episodes,
            "mean_fwd": f"{mean_fwd:.4f}", "mean_turn": f"{mean_turn:.4f}",
        })


def _recv_with_retry(client, zmq_again_cls):
    while True:
        try:
            return client.recv_obs()
        except Exception as exc:
            if zmq_again_cls and isinstance(exc, zmq_again_cls):
                print("[brain_gpu] recv timeout — retrying")
            else:
                raise


# ── Run loops ──────────────────────────────────────────────────────────────────

def run_connected(brain: EmbodiedBrainGPU, args: argparse.Namespace) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
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
                _log(step, reward_sum, args.log_every, brain._valence_pred,
                     brain.w_out_norm, eat_count, episodes, csv_writer, fwd_sum, turn_sum)
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


def run_headless(brain: EmbodiedBrainGPU, args: argparse.Namespace) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
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
    life_idx   = brain._state_sl.start

    try:
        while step < n_steps:
            obs    = world.get_ai_observation()
            reward = float(world.ai.meters.valence)
            life   = float(obs[life_idx])
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
                _log(step, reward_sum, args.log_every, brain._valence_pred,
                     brain.w_out_norm, eat_count, episodes, csv_writer, fwd_sum, turn_sum)
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated ESN brain (PyTorch, 4× larger reservoirs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python brain_gpu.py                          connect + learn on GPU
  python brain_gpu.py --device cpu             force CPU (for testing)
  python brain_gpu.py --headless 18000         offline headless run
  python brain_gpu.py --load brain_gpu.npz     resume checkpoint
  python brain_gpu.py --seed 7                 different reservoir init
""")
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--no-learn",   action="store_true",
                        help="Freeze W_out (inference only)")
    parser.add_argument("--save",       metavar="PATH")
    parser.add_argument("--load",       metavar="PATH")
    parser.add_argument("--headless",   type=int, default=0, metavar="N")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY,  metavar="N")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N")
    parser.add_argument("--log-path",   metavar="PATH")
    args = parser.parse_args()

    print(f"Building GPU brain  (V1={V1_SIZE}, TAC={TAC_SIZE}, VAL={VAL_SIZE}, "
          f"CENTRAL={CENTRAL_SIZE}, MOTOR={MOTOR_SIZE})…")
    brain = EmbodiedBrainGPU(config={"seed": args.seed}, device=args.device)

    if args.load:
        brain.load(args.load)
        print(f"Loaded weights from {args.load}")

    if args.no_learn:
        brain.learning_enabled = False
        print("Learning disabled.")

    print(f"  Device: {brain._dev}  |  W_out: {brain.w_out_norm:.5f}  |  "
          f"W_out shape: {tuple(brain.W_out.shape)}")

    if args.headless:
        print(f"Running {args.headless} headless steps…")
        run_headless(brain, args)
    else:
        print("Waiting for game connection…")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
