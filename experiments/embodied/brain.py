"""
experiments/embodied/brain.py — Multi-Reservoir ESN Brain

Architecture (all inter-reservoir weights are fixed random; only W_out trains):

    vision(242)  →  [V1_res,      256]  ─┐
    tactile(18)  →  [tac_res,     128]  ─┤ concat(448) → [central_res, 512]
    state(3)     →  [val_res,      64]  ─┘              → [motor_res,   256]
                                                         → W_out (3×256)
                                                         → (fwd, turn, eat)

W_out is updated online via RPE-gated eligibility traces — no pretraining.
All reservoir weights are fixed random (spectral-radius scaled) and never change.

Usage:
    python brain.py                          connect to game.py --connect, learn
    python brain.py --no-learn               connect, inference only
    python brain.py --load brain.npz         resume from saved weights
    python brain.py --headless 18000 --save brain.npz   offline run
    python brain.py --seed 7                 different reservoir init
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

import numpy as np

# ── Sensor counts — edit these to adapt to a different env ────────────────────
N_RAYS         = 121   # vision rays
N_TOUCH_BODY   = 16    # body tactile receptors
N_TOUCH_PRONGS = 2     # prong receptors
STATE_SIZE     = 3     # internal meters: life, satiation_norm, valence_norm

# Obs slices — derived from sensor counts, never hardcoded elsewhere
_VIS_SIZE    = N_RAYS * 2                       # 242
_TAC_SIZE_IN = N_TOUCH_BODY + N_TOUCH_PRONGS    # 18
VIS_START    = 0
VIS_END      = _VIS_SIZE                        # 242
TAC_START    = VIS_END                          # 242
TAC_END      = TAC_START + _TAC_SIZE_IN         # 260
STATE_START  = TAC_END                          # 260
STATE_END    = STATE_START + STATE_SIZE         # 263
OBS_SIZE     = STATE_END                        # 263
VALENCE_IDX  = STATE_END - 1                    # 262

# ── Reservoir sizes ────────────────────────────────────────────────────────────
V1_SIZE      = 256
TAC_SIZE     = 128
VAL_SIZE     = 64
CENTRAL_SIZE = 512
MOTOR_SIZE   = 256

# ── Reservoir dynamics (per-reservoir) ────────────────────────────────────────
SPECTRAL_RADIUS     = 0.99
VAL_SPECTRAL_RADIUS = 0.95   # value reservoir: faster, more reactive

V1_NOISE      = 0.01
TAC_NOISE     = 0.01
VAL_NOISE     = 0.005   # more precise — fewer spurious value fluctuations
CENTRAL_NOISE = 0.01
MOTOR_NOISE   = 0.01

V1_INPUT_SCALING      = 1.0
TAC_INPUT_SCALING     = 1.0
VAL_INPUT_SCALING     = 1.0
CENTRAL_INPUT_SCALING = 1.0
MOTOR_INPUT_SCALING   = 1.0

EXPLORE_NOISE = 0.2    # Gaussian noise std on fwd/turn raw outputs
EAT_THRESHOLD = 0.0    # eat when raw[2] > 0  (equiv. sigmoid > 0.5)

# ── Online learning ────────────────────────────────────────────────────────────
LEARN_LR     = 1e-4    # W_out update learning rate
CRITIC_LR    = 1e-3    # valence prediction EMA rate (slow baseline)
TRACE_DECAY  = 0.9     # eligibility trace decay per step
WEIGHT_DECAY = 1e-5    # L2 regularisation on W_out per step

# ── Logging / saving ───────────────────────────────────────────────────────────
LOG_EVERY  = 300       # print stats every N steps
SAVE_EVERY = 3600      # save weights every N steps (~1 min at 60fps)


# ── Reservoir ─────────────────────────────────────────────────────────────────

class Reservoir:
    """
    Single fixed-weight Echo State Network reservoir.

    All weights (W_in, W_res, bias) are randomly initialised at construction
    and never updated.  Hidden state h is advanced via step().

    Update equation:
        h_t = tanh(W_in @ x_t  +  W_res @ h_{t-1}  +  bias  +  noise_t)
    """

    def __init__(
        self,
        input_dim: int,
        size: int,
        spectral_radius: float = SPECTRAL_RADIUS,
        input_scaling: float = 1.0,
        noise_scale: float = 0.01,
        seed: int = 0,
    ) -> None:
        self._size        = size
        self._noise_scale = noise_scale
        self._rng         = np.random.default_rng(seed)

        init_rng = np.random.default_rng(seed)   # deterministic weight init

        self._W_in  = init_rng.standard_normal((size, input_dim)) * input_scaling
        self._bias  = init_rng.standard_normal(size) * 0.1

        W_res = init_rng.standard_normal((size, size))
        eigvals    = np.linalg.eigvals(W_res)
        current_sr = float(np.max(np.abs(eigvals)))
        if current_sr > 0:
            W_res = W_res * (spectral_radius / current_sr)
        self._W_res = W_res

        self._h = np.zeros(size, dtype=np.float64)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Advance one time step; return new hidden state (shape: size,)."""
        noise   = self._noise_scale * self._rng.standard_normal(self._size)
        self._h = np.tanh(self._W_in @ x + self._W_res @ self._h + self._bias + noise)
        return self._h

    def reset(self) -> None:
        """Zero the hidden state (call on episode end / done=True)."""
        self._h[:] = 0.0

    @property
    def state(self) -> np.ndarray:
        """Current hidden state, shape (size,)."""
        return self._h


# ── Brain ──────────────────────────────────────────────────────────────────────

class EmbodiedBrain:
    """
    Multi-reservoir ESN brain for the embodied AI agent.

    All five reservoirs have fixed random weights.  Only the linear readout
    W_out is trained, via online RPE-gated eligibility traces (no pretraining).

    Parameters
    ----------
    config : dict | None
        Optional overrides for any module-level constant by name.
        Sensor-count overrides (N_RAYS, N_TOUCH_BODY, N_TOUCH_PRONGS) cause obs
        slices to be recomputed automatically.
        Example: {"V1_NOISE": 0.02, "LEARN_LR": 5e-5, "N_RAYS": 60}
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = _build_cfg(config)
        seed = cfg.get("seed", 42)

        # Obs slices — derived from (potentially overridden) sensor counts
        n_rays   = cfg["N_RAYS"]
        n_body   = cfg["N_TOUCH_BODY"]
        n_prongs = cfg["N_TOUCH_PRONGS"]
        st_sz    = cfg["STATE_SIZE"]
        vis_sz   = n_rays * 2
        tac_sz   = n_body + n_prongs
        self._vis_sl   = slice(0,                vis_sz)
        self._tac_sl   = slice(vis_sz,           vis_sz + tac_sz)
        self._state_sl = slice(vis_sz + tac_sz,  vis_sz + tac_sz + st_sz)

        # Build five reservoirs
        v1_sz  = cfg["V1_SIZE"]
        tac_rs = cfg["TAC_SIZE"]
        val_sz = cfg["VAL_SIZE"]
        cen_sz = cfg["CENTRAL_SIZE"]
        mot_sz = cfg["MOTOR_SIZE"]

        self.v1_res = Reservoir(
            vis_sz, v1_sz,
            spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["V1_INPUT_SCALING"],
            noise_scale=cfg["V1_NOISE"], seed=seed,
        )
        self.tac_res = Reservoir(
            tac_sz, tac_rs,
            spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["TAC_INPUT_SCALING"],
            noise_scale=cfg["TAC_NOISE"], seed=seed + 1,
        )
        self.val_res = Reservoir(
            st_sz, val_sz,
            spectral_radius=cfg["VAL_SPECTRAL_RADIUS"],
            input_scaling=cfg["VAL_INPUT_SCALING"],
            noise_scale=cfg["VAL_NOISE"], seed=seed + 2,
        )
        self.central_res = Reservoir(
            v1_sz + tac_rs + val_sz, cen_sz,   # input = 448
            spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["CENTRAL_INPUT_SCALING"],
            noise_scale=cfg["CENTRAL_NOISE"], seed=seed + 3,
        )
        self.motor_res = Reservoir(
            cen_sz, mot_sz,
            spectral_radius=cfg["SPECTRAL_RADIUS"],
            input_scaling=cfg["MOTOR_INPUT_SCALING"],
            noise_scale=cfg["MOTOR_NOISE"], seed=seed + 4,
        )

        # Readout (only trained component)
        rng = np.random.default_rng(seed + 99)
        self.W_out = rng.standard_normal((3, mot_sz)) * 0.01
        self.b_out = np.zeros(3)

        # Online learning state
        self._trace           = np.zeros((3, mot_sz))
        self._valence_pred    = 0.0
        self._last_raw_action = np.zeros(3)
        self._last_h_motor    = np.zeros(mot_sz)

        # Runtime config
        self._explore_noise    = cfg["EXPLORE_NOISE"]
        self._eat_threshold    = cfg["EAT_THRESHOLD"]
        self._learn_lr         = cfg["LEARN_LR"]
        self._critic_lr        = cfg["CRITIC_LR"]
        self._trace_decay      = cfg["TRACE_DECAY"]
        self._weight_decay     = cfg["WEIGHT_DECAY"]
        self.learning_enabled  = True
        self._rng              = np.random.default_rng(seed + 100)

    # ── Forward pass ───────────────────────────────────────────────────────────

    def forward(self, obs: np.ndarray) -> tuple[float, float, float]:
        """
        Observation → (fwd, turn, eat).

        Pipeline:
          1. Slice obs → (vision, tactile, state).
          2. Step sensory reservoirs independently.
          3. Concatenate states → step central reservoir.
          4. Step motor reservoir on central state.
          5. Compute raw = W_out @ h_motor + b_out; cache for learn().
          6. Add exploration noise to fwd/turn channels; apply nonlinearities.

        Returns
        -------
        (fwd, turn, eat) : fwd/turn ∈ [-1, 1],  eat ∈ {0.0, 1.0}
        """
        h_v1  = self.v1_res.step(obs[self._vis_sl])
        h_tac = self.tac_res.step(obs[self._tac_sl])
        h_val = self.val_res.step(obs[self._state_sl])

        h_central = self.central_res.step(np.concatenate([h_v1, h_tac, h_val]))
        h_motor   = self.motor_res.step(h_central)

        raw = self.W_out @ h_motor + self.b_out
        self._last_raw_action[:] = raw
        self._last_h_motor[:]    = h_motor

        noise = self._rng.standard_normal(2) * self._explore_noise
        fwd   = float(np.tanh(raw[0] + noise[0]))
        turn  = float(np.tanh(raw[1] + noise[1]))
        eat   = 1.0 if raw[2] > self._eat_threshold else 0.0

        return fwd, turn, eat

    # ── Online learning ────────────────────────────────────────────────────────

    def learn(self, reward: float) -> None:
        """
        Online RPE-gated eligibility trace update of W_out.

        Call after forward() and before the next forward() call.
        No-op when learning_enabled is False.

        Equations:
            rpe           = reward − valence_pred
            valence_pred += CRITIC_LR × rpe
            trace         = TRACE_DECAY × trace + outer(raw_action, h_motor)
            W_out        += LEARN_LR × rpe × trace
            W_out        *= (1 − WEIGHT_DECAY)

        Parameters
        ----------
        reward : float
            Raw valence ∈ [-1, 1] (from recv_obs() or world.ai.meters.valence).
        """
        if not self.learning_enabled:
            return
        rpe = reward - self._valence_pred
        self._valence_pred += self._critic_lr * rpe
        self._trace = (
            self._trace_decay * self._trace
            + np.outer(self._last_raw_action, self._last_h_motor)
        )
        self.W_out += self._learn_lr * rpe * self._trace
        self.W_out *= 1.0 - self._weight_decay

    # ── Episode reset ──────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """
        Reset all reservoir hidden states and eligibility trace.

        Call when done=True (world reset).  Does NOT touch W_out, b_out, or
        valence_pred — the critic baseline persists across episodes.
        """
        for res in (self.v1_res, self.tac_res, self.val_res,
                    self.central_res, self.motor_res):
            res.reset()
        self._trace[:] = 0.0
        self._last_raw_action[:] = 0.0
        self._last_h_motor[:]    = 0.0

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save learned parameters to a .npz file (W_out, b_out, valence_pred).

        Reservoir weights are not saved — they are fixed and fully reproduced
        from the seed passed at construction.
        """
        np.savez(path, W_out=self.W_out, b_out=self.b_out,
                 valence_pred=np.array(self._valence_pred))

    def load(self, path: str) -> None:
        """
        Load learned parameters from a .npz file.
        Raises FileNotFoundError if the file does not exist.
        """
        data = np.load(path)
        self.W_out[:]      = data["W_out"]
        self.b_out[:]      = data["b_out"]
        self._valence_pred = float(data["valence_pred"])


# ── Config helper ─────────────────────────────────────────────────────────────

def _build_cfg(user: dict | None) -> dict:
    defaults: dict = dict(
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
        EXPLORE_NOISE=EXPLORE_NOISE, EAT_THRESHOLD=EAT_THRESHOLD,
        LEARN_LR=LEARN_LR, CRITIC_LR=CRITIC_LR,
        TRACE_DECAY=TRACE_DECAY, WEIGHT_DECAY=WEIGHT_DECAY,
    )
    if user:
        defaults.update(user)
    return defaults


# ── Run loops ─────────────────────────────────────────────────────────────────

def _log(step: int, reward_sum: float, log_every: int,
         valence_pred: float, w_norm: float, eat_count: int) -> None:
    mean_r = reward_sum / max(log_every, 1)
    print(
        f"step {step:>7}  reward {mean_r:+.3f}  "
        f"valence_pred {valence_pred:+.3f}  "
        f"|W_out| {w_norm:.4f}  eat:{eat_count}"
    )


def _install_save_handler(brain: EmbodiedBrain, args: argparse.Namespace) -> None:
    """Install SIGUSR1 handler: kill -USR1 <pid> triggers an immediate save."""
    if not args.save:
        return

    def _handler(*_: object) -> None:
        brain.save(args.save)
        print(f"\n[SIGUSR1] Saved → {args.save}  (PID {os.getpid()})")

    signal.signal(signal.SIGUSR1, _handler)
    print(f"  Save on demand:  kill -USR1 {os.getpid()}")


def run_connected(brain: EmbodiedBrain, args: argparse.Namespace) -> None:
    """Agent loop via ZeroMQ AgentConnector (requires game.py --connect)."""
    _ensure_path()
    _install_save_handler(brain, args)
    from connector import AgentConnector  # noqa: PLC0415

    try:
        import zmq
        _ZmqAgain = zmq.Again
    except ImportError:
        _ZmqAgain = None

    client = AgentConnector()
    client.connect()

    step        = 0
    eat_count   = 0
    reward_sum  = 0.0

    try:
        obs, reward, done, _ = client.recv_obs()

        while True:
            fwd, turn, eat = brain.forward(obs)
            brain.learn(reward)
            if done:
                brain.reset_state()
            client.send_action((fwd, turn, eat))

            reward_sum += reward
            eat_count  += int(eat > 0.5)
            step       += 1

            if step % args.log_every == 0:
                _log(step, reward_sum, args.log_every,
                     brain._valence_pred,
                     float(np.linalg.norm(brain.W_out)), eat_count)
                reward_sum = 0.0
                eat_count  = 0

            if args.save and step % args.save_every == 0:
                brain.save(args.save)
                print(f"  [saved → {args.save}]")

            while True:
                try:
                    obs, reward, done, _ = client.recv_obs()
                    break
                except Exception as exc:
                    if _ZmqAgain and isinstance(exc, _ZmqAgain):
                        print("[brain] recv timeout — retrying")
                        continue
                    raise

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.save:
            brain.save(args.save)
            print(f"Saved → {args.save}")
        client.close()


def run_headless(brain: EmbodiedBrain, args: argparse.Namespace) -> None:
    """Agent loop directly against World() — no display required."""
    _ensure_path()
    from env import World  # noqa: PLC0415

    world      = World(seed=args.seed)
    n_steps    = args.headless
    step       = 0
    eat_count  = 0
    reward_sum = 0.0
    prev_life  = 1.0

    life_idx = brain._state_sl.start   # obs[260] = life

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
            world.step(ai_action=(fwd, turn, eat))

            reward_sum += reward
            eat_count  += int(eat > 0.5)
            step       += 1

            if step % args.log_every == 0:
                _log(step, reward_sum, args.log_every,
                     brain._valence_pred,
                     float(np.linalg.norm(brain.W_out)), eat_count)
                reward_sum = 0.0
                eat_count  = 0

            if args.save and step % args.save_every == 0:
                brain.save(args.save)
                print(f"  [saved → {args.save}]")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.save:
            brain.save(args.save)
            print(f"Saved → {args.save}")


def _ensure_path() -> None:
    """Add the embodied directory to sys.path so local imports work."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-reservoir ESN brain for the embodied AI agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python brain.py                             connect + learn online
  python brain.py --no-learn                  connect, inference only
  python brain.py --load brain.npz            resume from saved weights
  python brain.py --headless 18000            offline, no display
  python brain.py --headless 18000 --save w.npz --save-every 3600
  python brain.py --seed 7                    different reservoir init
        """,
    )
    parser.add_argument("--no-learn",   action="store_true",
                        help="Freeze W_out (inference only)")
    parser.add_argument("--save",       metavar="PATH",
                        help="Save weights to PATH (periodic + on exit)")
    parser.add_argument("--load",       metavar="PATH",
                        help="Load weights from PATH before starting")
    parser.add_argument("--headless",   type=int, default=0, metavar="N",
                        help="Run N steps against World() directly (no display)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed for reservoir init (default 42)")
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY, metavar="N",
                        help=f"Print stats every N steps (default {LOG_EVERY})")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N",
                        help=f"Save every N steps (default {SAVE_EVERY})")
    args = parser.parse_args()

    print("Building multi-reservoir ESN brain…")
    brain = EmbodiedBrain(config={"seed": args.seed})

    if args.load:
        brain.load(args.load)
        print(f"Loaded weights from {args.load}")

    if args.no_learn:
        brain.learning_enabled = False
        print("Learning disabled.")

    w_norm = float(np.linalg.norm(brain.W_out))
    print(
        f"Reservoirs: V1={V1_SIZE}  tac={TAC_SIZE}  val={VAL_SIZE}  "
        f"central={CENTRAL_SIZE}  motor={MOTOR_SIZE}  |W_out|={w_norm:.4f}"
    )

    if args.headless:
        print(f"Running {args.headless} headless steps…")
        run_headless(brain, args)
    else:
        print("Connecting to game… (start  python game.py --connect  first)")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
