"""
experiments/embodied/nn_brain.py — Trainable MLP Actor-Critic Brain

A straightforward feedforward network trained online with single-step actor-critic
(TD(0)). No reservoir, no eligibility traces, no fixed random weights.

Architecture
------------
Trunk:   obs[263] → Linear(512) → ReLU → Linear(256) → ReLU
Actor:   trunk → fwd  (Linear → tanh, Gaussian policy)
         trunk → turn (Linear → tanh, Gaussian policy)
         trunk → eat  (Linear → sigmoid, Bernoulli policy)
Critic:  trunk → value (Linear, scalar)

Training
--------
At each step:
  rpe  = reward − value_pred
  actor_loss = −log_prob(action | obs) · stop_grad(rpe)
  critic_loss = 0.5 · rpe²
  loss = actor_loss + critic_coef · critic_loss − entropy_coef · entropy
  loss.backward() → Adam step

Exploration decays from explore_std_init toward explore_std_min over the
first explore_decay_steps steps.

Usage
-----
    python nn_brain.py                          # connect to game.py --connect
    python nn_brain.py --headless 3600
    python nn_brain.py --config brains/configs/nn-256.json
    python nn_brain.py --no-learn --load brains/NN-256.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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

import dofs                                                    # noqa: E402

import brain as _brain_mod                                     # noqa: E402
from brain import (                                            # noqa: E402
    run_connected, run_headless,
    _log_knowledge, _compute_extra_k, _format_k,
    load_config, save_config, _select_device,
    _config_path, _resolve_paths,
    OBS_DIM, _LIFE_IDX,
    LOG_EVERY, SAVE_EVERY, LOG_CAPACITY, ASSESS_EVERY,
    _IdentityMapping, _EmptyReservoir,
)

# -- Default config -----------------------------------------------------------

NN_DEFAULT_CONFIG: dict = {
    # Identity / paths
    "name":             "",
    "brain_path":       "",
    "log_path":         "",
    "world_config":     "",
    # Network
    "hidden1":          512,
    "hidden2":          256,
    # Learning
    "actor_lr":         3e-4,
    "critic_coef":      0.5,       # weight of critic loss vs actor loss
    "entropy_coef":     0.05,      # entropy bonus — keeps policy from collapsing
    "weight_decay":     0.0,       # no L2 decay — tiny rewards can't fight it
    "max_grad_norm":    1.0,
    "gamma":            0.99,      # discount factor for TD bootstrap
    # Reward
    "reward_scale":     10.0,      # scale Δvalence into a useful gradient range
    # Exploration (std of Gaussian policy on fwd/turn)
    "explore_std_init": 0.8,
    "explore_std_min":  0.1,
    "explore_decay_steps": 200000,
    # Action
    "eat_threshold":    0.5,       # sigmoid output above which eat fires
    # RO Framework
    "assess_every":     ASSESS_EVERY,
    "log_capacity":     LOG_CAPACITY,
    # Arch
    "seed":             42,
    "action_feedback":  False,
    "decision_interval": 1,
    "device":           "cuda",
}


# -- Network ------------------------------------------------------------------

class _ActorCritic(nn.Module):
    """Shared-trunk actor-critic MLP."""

    def __init__(self, obs_dim: int, hidden1: int, hidden2: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        # Actor heads
        self.fwd_head  = nn.Linear(hidden2, 1)
        self.turn_head = nn.Linear(hidden2, 1)
        self.eat_head  = nn.Linear(hidden2, 1)
        # Critic head
        self.value_head = nn.Linear(hidden2, 1)

        # Small init for output heads — keeps initial policy near zero
        for head in (self.fwd_head, self.turn_head, self.eat_head, self.value_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        # Orthogonal init for trunk
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Returns (fwd_mean, turn_mean, eat_logit, value)."""
        h = self.trunk(obs)
        return (
            torch.tanh(self.fwd_head(h)),
            torch.tanh(self.turn_head(h)),
            self.eat_head(h),          # raw logit, sigmoid applied in log_prob
            self.value_head(h).squeeze(-1),
        )


# -- NNBrain ------------------------------------------------------------------

class NNBrain:
    """MLP actor-critic — drop-in replacement for EmbodiedBrain.

    Same external interface: forward(), learn(), reset_state(), save(), load().
    """

    _state_sl = slice(_LIFE_IDX, _LIFE_IDX + 3)

    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = "cuda",
        action_feedback: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        cfg = dict(NN_DEFAULT_CONFIG)
        if config:
            cfg.update(config)

        if seed is None:
            seed = int(cfg.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._dev = _select_device(device)
        self._action_feedback = action_feedback
        self._eat_threshold = float(cfg["eat_threshold"])

        # -- Network & optimizer ----------------------------------------------
        obs_dim = OBS_DIM + (3 if action_feedback else 0)
        self._net = _ActorCritic(
            obs_dim=obs_dim,
            hidden1=int(cfg["hidden1"]),
            hidden2=int(cfg["hidden2"]),
        ).to(self._dev)

        self._optim = torch.optim.Adam(
            self._net.parameters(),
            lr=float(cfg["actor_lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        self._critic_coef   = float(cfg["critic_coef"])
        self._entropy_coef  = float(cfg["entropy_coef"])
        self._max_grad_norm = float(cfg["max_grad_norm"])
        self._gamma         = float(cfg["gamma"])
        self._reward_scale  = float(cfg["reward_scale"])

        # -- Exploration schedule ---------------------------------------------
        self._explore_std     = float(cfg["explore_std_init"])
        self._explore_std_min = float(cfg["explore_std_min"])
        self._explore_decay   = float(cfg["explore_decay_steps"])
        self._explore_std_init = float(cfg["explore_std_init"])

        # -- TD(0) + Δvalence state ------------------------------------------
        # forward() samples actions with no_grad (fast, no graph).
        # learn()   does a fresh forward on the PREVIOUS obs with grad to get
        #           clean gradients — avoids stale-graph errors from storing
        #           tensors across optimizer steps.
        self._valence_pred   = 0.0
        self._prev_valence   = 0.0    # for Δvalence computation
        self._curr_value_f   = 0.0    # V(s_t) as scalar — bootstrap target for next step
        # stored as numpy (no graph attached) — re-forwarded in learn()
        self._prev_obs_np:   Optional[np.ndarray] = None
        self._curr_obs_np:   Optional[np.ndarray] = None
        # stored actions as plain numpy — used to recompute log_prob in learn()
        self._prev_act_np:   Optional[np.ndarray] = None  # [fwd, turn, eat]
        self._curr_act_np:   Optional[np.ndarray] = None
        self._last_action    = np.zeros(3, dtype=np.float32)
        self._prev_action    = np.zeros(3, dtype=np.float32)  # action_feedback
        self._step           = 0

        self.learning_enabled = True
        self._last_obs: Optional[np.ndarray] = None

        # -- RO Framework -----------------------------------------------------
        self._observer = Observer(
            name="nn_brain",
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

    # -- Forward pass ---------------------------------------------------------

    def forward(self, obs: np.ndarray) -> tuple:
        """Sample an action. Runs with no_grad — learn() does the gradient pass."""
        obs_t = self._obs_tensor(obs)

        with torch.no_grad():
            fwd_mean, turn_mean, eat_logit, value = self._net(obs_t)

        # Decay exploration std
        std = max(
            self._explore_std_min,
            self._explore_std_init
            - (self._explore_std_init - self._explore_std_min)
            * min(self._step / max(self._explore_decay, 1), 1.0),
        )
        self._explore_std = std
        std_t = torch.tensor(std, dtype=torch.float32, device=self._dev)

        # Sample actions
        fwd_raw  = (fwd_mean  + torch.randn_like(fwd_mean)  * std_t).clamp(-1.0, 1.0)
        turn_raw = (turn_mean + torch.randn_like(turn_mean) * std_t).clamp(-1.0, 1.0)
        eat_prob   = torch.sigmoid(eat_logit)
        eat_sample = torch.bernoulli(eat_prob)

        fwd  = float(fwd_raw)
        turn = float(turn_raw)
        eat  = float(eat_sample)

        # Shift obs/action buffers (stored as numpy — no graph)
        self._prev_obs_np  = self._curr_obs_np
        self._prev_act_np  = self._curr_act_np
        self._curr_obs_np  = obs.copy()
        self._curr_act_np  = np.array([fwd, turn, eat], dtype=np.float32)
        self._curr_value_f = float(value)   # scalar bootstrap target for next learn()
        self._valence_pred = float(value)

        self._last_action = self._curr_act_np
        self._step += 1

        if self._action_feedback:
            self._prev_action = self._last_action.copy()

        self._observer.observation_log.append(ObservationPair(
            external_state=dofs.obs_to_state(obs),
            internal_state=dofs.action_to_state(fwd, turn, eat),
            timestamp=float(len(self._observer.observation_log)),
        ))

        return fwd, turn, eat

    # -- Learning -------------------------------------------------------------

    def learn(self, reward: float) -> None:
        """TD(0) actor-critic update on the PREVIOUS transition.

        forward() samples actions cheaply with no_grad.
        learn() does a fresh forward pass on the previous obs (with grad) so
        gradients flow through current parameters — no stale-graph issues.

        Reward signal: Δvalence * reward_scale.  Raw valence hovers near 0;
        the per-step delta reflects actual events (eating, danger, hunger).

        TD target: Δr * scale + γ * V(s_t)_scalar  [V(s_t) = current value, detached]
        Loss targets V(s_{t-1}) freshly computed with current params.
        """
        if not self.learning_enabled:
            return
        # Need two consecutive forward() calls
        if self._prev_obs_np is None:
            self._prev_valence = reward
            return

        # Δvalence reward
        delta_r = (reward - self._prev_valence) * self._reward_scale
        self._prev_valence = reward

        # Fresh forward on previous obs — gradients flow through current params
        prev_obs_t = self._obs_tensor(self._prev_obs_np)
        fwd_mean, turn_mean, eat_logit, value = self._net(prev_obs_t)

        # Recompute log_prob and entropy from the action we actually took
        log_prob, entropy = self._log_prob_entropy(
            fwd_mean, turn_mean, eat_logit, self._prev_act_np,
        )

        # TD target uses current V(s_t) as scalar (no grad — it's the bootstrap)
        td_target_f = delta_r + self._gamma * self._curr_value_f
        td_target_t = torch.tensor(td_target_f, dtype=torch.float32, device=self._dev)
        rpe = td_target_t - value   # gradient flows through value (critic update)

        actor_loss   = -(log_prob * rpe.detach())
        critic_loss  = 0.5 * rpe ** 2
        entropy_loss = -self._entropy_coef * entropy

        loss = (actor_loss + self._critic_coef * critic_loss + entropy_loss).mean()

        self._optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), self._max_grad_norm)
        self._optim.step()

    def set_executed_action(self, action, **_kwargs) -> None:
        """Interface compat — NNBrain uses sampled actions directly."""
        pass

    # -- Episode reset --------------------------------------------------------

    def reset_state(self) -> None:
        """Clear per-episode state at episode boundaries."""
        self._prev_obs_np  = None   # breaks TD chain across episodes
        self._prev_act_np  = None
        self._last_action  = np.zeros(3, dtype=np.float32)
        self._prev_action  = np.zeros(3, dtype=np.float32)
        # _prev_valence intentionally NOT reset — valence is continuous across episodes

    # -- Helpers ---------------------------------------------------------------

    def _log_prob_entropy(
        self,
        fwd_mean: torch.Tensor,
        turn_mean: torch.Tensor,
        eat_logit: torch.Tensor,
        action_np: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log_prob and entropy for a stored action under current policy."""
        std   = self._explore_std
        std_t = torch.tensor(std, dtype=torch.float32, device=self._dev)
        log_std = torch.log(std_t)
        eps = 1e-8

        fwd_a  = torch.tensor(float(action_np[0]), dtype=torch.float32, device=self._dev)
        turn_a = torch.tensor(float(action_np[1]), dtype=torch.float32, device=self._dev)
        eat_a  = torch.tensor(float(action_np[2]), dtype=torch.float32, device=self._dev)

        lp_fwd  = -0.5 * ((fwd_a  - fwd_mean)  / std_t) ** 2 - log_std - 0.9189
        lp_turn = -0.5 * ((turn_a - turn_mean) / std_t) ** 2 - log_std - 0.9189

        eat_prob = torch.sigmoid(eat_logit)
        lp_eat   = eat_a * torch.log(eat_prob + eps) + \
                   (1 - eat_a) * torch.log(1 - eat_prob + eps)

        log_prob = lp_fwd + lp_turn + lp_eat

        ent_cont = 0.5 + 0.5 * math.log(2 * math.pi) + log_std  # same for fwd and turn
        ent_eat  = -(eat_prob * torch.log(eat_prob + eps)
                     + (1 - eat_prob) * torch.log(1 - eat_prob + eps))
        entropy  = 2 * ent_cont + ent_eat

        return log_prob, entropy

    # -- Knowledge tracker ----------------------------------------------------

    def step_knowledge(self, epoch: int) -> dict:
        return self._tracker.step(epoch)

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "net_state": self._net.state_dict(),
            "optim_state": self._optim.state_dict(),
            "step": self._step,
            "valence_pred": self._valence_pred,
            "explore_std": self._explore_std,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self._dev, weights_only=False)
        self._net.load_state_dict(ckpt["net_state"])
        self._optim.load_state_dict(ckpt["optim_state"])
        self._step         = int(ckpt.get("step", 0))
        self._valence_pred = float(ckpt.get("valence_pred", 0.0))
        self._explore_std  = float(ckpt.get("explore_std", self._explore_std))

    # -- Properties -----------------------------------------------------------

    @property
    def w_out_norm(self) -> float:
        """Total parameter norm (diagnostic)."""
        return float(sum(
            p.data.norm() for p in self._net.parameters()
        ))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    # brain_viz compat stubs
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
    def _last_h_motor(self) -> torch.Tensor:
        return torch.zeros(1)

    # -- Helpers --------------------------------------------------------------

    def _obs_tensor(self, obs: np.ndarray) -> torch.Tensor:
        if self._action_feedback:
            obs = np.concatenate([obs, self._prev_action])
        return torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self._dev)


# -- Logging ------------------------------------------------------------------

_NN_CSV_FIELDS = ["param_norm", "explore_std", "train_step"]


def _nn_open_log(path: str) -> csv.DictWriter:
    from brain import _K_DOF_NAMES, _EXTRA_K_FIELDS
    k_fields = [f"K_{n}" for n in _K_DOF_NAMES]
    fields = (["timestamp", "step", "mean_reward", "valence_pred", "w_norm",
               "eat_count", "episodes", "mean_fwd", "mean_turn"]
              + k_fields + _EXTRA_K_FIELDS + _NN_CSV_FIELDS)
    is_new = not os.path.exists(path)
    fh = open(path, "a", newline="", buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return writer


def _nn_log_step(brain, step, reward_sum, log_every,
                 eat_count, episodes, fwd_sum, turn_sum,
                 csv_writer=None) -> None:
    mean_r    = reward_sum / max(log_every, 1)
    mean_fwd  = fwd_sum   / max(log_every, 1)
    mean_turn = turn_sum  / max(log_every, 1)

    print(f"step {step:>7}  valence {mean_r:+.3f}  v_pred {brain._valence_pred:+.3f}  "
          f"fwd:{mean_fwd:+.2f}  turn:{mean_turn:+.02f}  "
          f"eat:{eat_count}  ep:{episodes}  "
          f"std:{brain._explore_std:.3f}")

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step, "mean_reward": f"{mean_r:.4f}",
        "valence_pred": f"{brain._valence_pred:.4f}",
        "w_norm": f"{brain.w_out_norm:.4f}",
        "eat_count": eat_count, "episodes": episodes,
        "mean_fwd": f"{mean_fwd:.4f}", "mean_turn": f"{mean_turn:.4f}",
        "param_norm": f"{brain.w_out_norm:.4f}",
        "explore_std": f"{brain._explore_std:.4f}",
        "train_step": brain._step,
    }

    _log_knowledge(brain, step, csv_row=row)
    row.update(_compute_extra_k(brain))

    if csv_writer is not None:
        csv_writer.writerow(row)


def _install_nn_logging() -> None:
    _brain_mod._log_step = _nn_log_step
    _brain_mod._open_log = _nn_open_log


# -- CLI ----------------------------------------------------------------------

def _resolve_config(args) -> dict:
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
        cfg = {**NN_DEFAULT_CONFIG, **user_cfg}
    else:
        cfg = dict(NN_DEFAULT_CONFIG)
    if args.device:
        cfg["device"] = args.device
    return cfg


def _build_brain(cfg: dict) -> NNBrain:
    return NNBrain(
        config=cfg,
        device=cfg.get("device", "cuda"),
        action_feedback=cfg.get("action_feedback", False),
        seed=cfg.get("seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLP actor-critic brain for the embodied environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python nn_brain.py                                      # connect to game
  python nn_brain.py --headless 3600                      # offline 1-hour run
  python nn_brain.py --config brains/configs/nn-256.json --save brains/NN-256.pt
  python nn_brain.py --no-learn --load brains/NN-256.pt
""")
    parser.add_argument("--config",     metavar="PATH")
    parser.add_argument("--device",     choices=["cuda", "cpu"], default=None)
    parser.add_argument("--no-learn",   action="store_true")
    parser.add_argument("--save",       metavar="PATH", default=None)
    parser.add_argument("--load",       metavar="PATH", default=None)
    parser.add_argument("--log-path",   metavar="PATH", default=None)
    parser.add_argument("--headless",   type=int, default=0, metavar="N")
    parser.add_argument("--no-reset",   action="store_true")
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY, metavar="N")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N")
    args = parser.parse_args()

    cfg = _resolve_config(args)
    brain_path, log_path, load_path = _resolve_paths(args, cfg)
    args.save = brain_path
    args.log_path = log_path

    print(f"Building NNBrain ({cfg['hidden1']} → {cfg['hidden2']})...")
    brain = _build_brain(cfg)
    print(f"  Device:        {brain._dev}")
    print(f"  Parameters:    {brain.param_count:,}")
    print(f"  Explore std:   {brain._explore_std:.3f} → {brain._explore_std_min:.3f} "
          f"over {int(cfg['explore_decay_steps']):,} steps")
    print(f"  Actor lr:      {cfg['actor_lr']}")
    print(f"  Critic coef:   {cfg['critic_coef']}")

    if load_path:
        brain.load(load_path)
        print(f"  Loaded from {load_path}  (step={brain._step})")

    if args.no_learn:
        brain.learning_enabled = False
        print("  Learning disabled.")

    if brain_path:
        save_config(cfg, _config_path(brain_path))

    args._brain_name = cfg.get("name") or ""
    args._world_config = cfg.get("world_config") or None
    args._decision_interval = cfg.get("decision_interval", 1)

    _install_nn_logging()

    if args.headless:
        print(f"\nRunning {args.headless} headless steps...")
        run_headless(brain, args)
    else:
        print("\nWaiting for game connection...")
        run_connected(brain, args)


if __name__ == "__main__":
    main()
