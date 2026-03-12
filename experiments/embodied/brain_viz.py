"""
brain_viz.py — Live ESN brain visualiser (1080p).

Layout
------
  Top-left   : Reservoir heatmaps (all 5, neuron-level activation)
  Centre     : Motor reservoir graph — nodes=neurons, edges=W_res connections,
               colour=activation.  Spring layout computed once at startup.
  Right strip: Rolling signal plots — fwd, turn, reward, RPE
  Bottom bar : Step / reward / action stats

Usage:
    python brain_viz.py                  # connected to game.py --connect
    python brain_viz.py --headless 3600  # headless with World()
    python brain_viz.py --load brain.npz # resume from checkpoint
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from typing import Deque

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(__file__))
from brain import (  # noqa: E402
    EmbodiedBrain,
    _open_log,
    _log,
    _recv_with_retry,
)

# ── Window / layout ────────────────────────────────────────────────────────────
WIN_W = 1920
WIN_H = 1080

# Columns
HEAT_X     = 8
HEAT_W     = 340

GRAPH_X    = HEAT_X + HEAT_W + 12
GRAPH_W    = 900
GRAPH_H    = 900

PLOT_X     = GRAPH_X + GRAPH_W + 12
PLOT_W     = WIN_W - PLOT_X - 8

STATS_H    = 28    # bottom stats bar
GRAPH_Y    = 8
HEAT_Y     = 8

HISTORY    = 600   # rolling plot length (steps)

# Graph edges: keep top EDGE_FRAC fraction by |weight|
EDGE_FRAC  = 0.04  # 4 % of n² edges  ≈ 2600 for n=256

# Spring layout iterations (one-time at startup)
LAYOUT_ITERS = 500

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = (10,  11,  16)
PANEL   = (18,  20,  28)
BORDER  = (45,  50,  70)
WHITE   = (215, 220, 230)
DIM     = (85,  90, 115)
RED     = (220,  50,  50)
GREEN   = ( 55, 215,  85)
BLUE    = ( 55, 130, 235)
YELLOW  = (225, 210,  55)
CYAN    = ( 55, 205, 215)
ORANGE  = (235, 140,  50)


def _act_col(v: float) -> tuple[int, int, int]:
    """Neuron activation → RGB.  negative=red, zero=dark, positive=green."""
    v = float(np.clip(v, -1.0, 1.0))
    if v >= 0.0:
        return (int(v * 35), int(v * 210), int(v * 75))
    return (int(-v * 225), int(-v * 45), int(-v * 45))


def _edge_col(w: float, alpha: float) -> tuple[int, int, int]:
    """Signed weight → RGB.  positive=blue, negative=orange."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if w >= 0.0:
        return (int(a * 40), int(a * 90), int(a * 220))
    return (int(a * 220), int(a * 110), int(a * 30))


# ── Spring layout ──────────────────────────────────────────────────────────────

def _spring_layout(w_res: np.ndarray, n_iter: int = LAYOUT_ITERS, seed: int = 0) -> np.ndarray:
    """
    Force-directed layout for reservoir recurrent weights.
    Returns (n, 2) positions normalised to [-1, 1].

    Only the top EDGE_FRAC fraction of edges by |weight| contribute
    spring forces; the rest are treated as absent.
    """
    n   = w_res.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n, 2)).astype(np.float32) * 0.5

    # Sparse spring mask — top edges by |weight|
    thresh   = np.percentile(np.abs(w_res), (1.0 - EDGE_FRAC) * 100)
    w_spring = np.where(np.abs(w_res) >= thresh, w_res.astype(np.float32), 0.0)

    print(f"  spring layout: {n} nodes, {int((w_spring != 0).sum())} spring edges, "
          f"{n_iter} iterations…", flush=True)

    for t in range(n_iter):
        lr = 0.06 * (1.0 - t / n_iter) + 0.004   # linear cooling

        # Repulsion (all pairs, vectorised)
        d   = pos[:, None, :] - pos[None, :, :]    # (n, n, 2)
        r   = np.linalg.norm(d, axis=2, keepdims=True)  # (n, n, 1)
        r   = np.maximum(r, 0.08)
        rep = (d / r ** 2).sum(axis=1)             # (n, 2)

        # Attraction (spring edges only)
        att = ((pos[None, :, :] - pos[:, None, :])
               * np.abs(w_spring[:, :, None])).sum(axis=1)   # (n, 2)

        pos += lr * (rep * 0.25 + att * 0.75)
        pos -= pos.mean(axis=0)

    mx = np.max(np.abs(pos))
    if mx > 0:
        pos /= mx
    return pos.astype(np.float64)


# ── Reservoir graph panel ──────────────────────────────────────────────────────

class _ReservoirGraph:
    """
    Renders a single reservoir as a node-edge graph.

    Node colour  = current activation (green=+1, red=-1, black=0).
    Edge colour  = weight sign (blue=excitatory, orange=inhibitory).
    Edge opacity = weight magnitude relative to max |weight|.
    Only the top EDGE_FRAC fraction of edges are drawn.
    """

    def __init__(self, reservoir, rect: pygame.Rect, seed: int = 0) -> None:
        self.res  = reservoir
        self.rect = rect
        n = len(reservoir.state)

        # Compute spring layout (one-time, printed to console)
        raw_pos = _spring_layout(reservoir.recurrent_weights, seed=seed)
        # Map [-1, 1] → pixel coords inside rect (with padding)
        pad = 18
        self.px = (((raw_pos[:, 0] + 1.0) * 0.5) * (rect.w - 2 * pad) + rect.x + pad).astype(int)
        self.py = (((raw_pos[:, 1] + 1.0) * 0.5) * (rect.h - 2 * pad) + rect.y + pad).astype(int)

        # Pre-build sorted edge list: [(i, j, w, alpha), …]  top EDGE_FRAC by |w|
        wrec     = reservoir.recurrent_weights
        flat_idx = np.argsort(np.abs(wrec).ravel())[::-1]
        k        = max(1, int(n * n * EDGE_FRAC))
        top_idx  = flat_idx[:k]
        rows, cols = np.unravel_index(top_idx, wrec.shape)
        ws       = wrec[rows, cols]
        w_max    = float(np.max(np.abs(ws))) + 1e-9
        alphas   = np.abs(ws) / w_max
        self.edges = list(zip(rows.tolist(), cols.tolist(),
                              ws.tolist(), alphas.tolist()))

        # Node radius scales with n
        self.r_node = max(2, int(10 - n / 50))

        print(f"  graph ready: {n} nodes, {k} edges drawn.", flush=True)

    def draw(self, screen: pygame.Surface, font) -> None:
        # Background panel
        pygame.draw.rect(screen, PANEL,  self.rect)
        pygame.draw.rect(screen, BORDER, self.rect, 1)

        state = self.res.state
        r     = self.r_node

        # Edges
        for i, j, w, alpha in self.edges:
            if alpha < 0.15:
                continue
            col = _edge_col(w, alpha * 0.7)
            pygame.draw.line(screen, col,
                             (self.px[i], self.py[i]),
                             (self.px[j], self.py[j]))

        # Nodes (drawn on top of edges)
        for idx in range(len(state)):
            col = _act_col(state[idx])
            pygame.draw.circle(screen, col, (self.px[idx], self.py[idx]), r)

        # Label
        screen.blit(font.render(
            f"Motor reservoir  ({len(state)} neurons)  —  "
            "node colour = activation  ·  edge colour = weight sign",
            True, DIM), (self.rect.x + 6, self.rect.y + 4))


# ── Main visualiser ────────────────────────────────────────────────────────────

class BrainViz:
    """Pygame window rendering ESN internal state in real time."""

    def __init__(self, brain: EmbodiedBrain) -> None:
        self.brain = brain

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("ESN Brain — Live Visualiser")
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.font_md = pygame.font.SysFont("monospace", 13)
        self.font_lg = pygame.font.SysFont("monospace", 17, bold=True)
        self.clock   = pygame.time.Clock()

        # Rolling histories
        self.h_fwd:    Deque[float] = collections.deque([0.0] * HISTORY, maxlen=HISTORY)
        self.h_turn:   Deque[float] = collections.deque([0.0] * HISTORY, maxlen=HISTORY)
        self.h_reward: Deque[float] = collections.deque([0.0] * HISTORY, maxlen=HISTORY)
        self.h_rpe:    Deque[float] = collections.deque([0.0] * HISTORY, maxlen=HISTORY)

        # Build graph (one-time spring layout — may take a few seconds)
        print("Building motor reservoir graph layout…")
        graph_rect = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        self.graph = _ReservoirGraph(brain.motor_res, graph_rect)

    # ── public API ─────────────────────────────────────────────────────────────

    def update(self, fwd: float, turn: float, reward: float, rpe: float) -> None:
        self.h_fwd.append(fwd)
        self.h_turn.append(turn)
        self.h_reward.append(reward)
        self.h_rpe.append(rpe)

    def render(self, step: int, fwd: float, turn: float, eat: float,
               reward: float, rpe: float) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(BG)
        self._draw_heatmaps()
        self.graph.draw(self.screen, self.font_sm)
        self._draw_plots()
        self._draw_stats_bar(step, fwd, turn, eat, reward, rpe)
        self._draw_attractor_alert()
        pygame.display.flip()
        self.clock.tick(60)
        return True

    def close(self) -> None:
        pygame.quit()

    # ── heatmaps ───────────────────────────────────────────────────────────────

    def _draw_heatmaps(self) -> None:
        res_list = [
            ("V1",      self.brain.v1_res.state),
            ("Tactile", self.brain.tac_res.state),
            ("Value",   self.brain.val_res.state),
            ("Central", self.brain.central_res.state),
            ("Motor",   self.brain.motor_res.state),
        ]

        label_h = 14
        bar_h   = 60
        gap     = 10
        total_h = len(res_list) * (label_h + bar_h + gap)
        y       = HEAT_Y + (WIN_H - STATS_H - total_h) // 2   # vertically centred

        self.screen.blit(
            self.font_sm.render(
                "Reservoir activations  (green = +1 · black = 0 · red = −1)",
                True, DIM),
            (HEAT_X, y - 18))

        for label, state in res_list:
            # Label + mean activity
            mean_abs = float(np.mean(np.abs(state)))
            self.screen.blit(
                self.font_sm.render(f"{label}  |μ|={mean_abs:.3f}", True, DIM),
                (HEAT_X, y))
            y += label_h

            surf = pygame.Surface((HEAT_W, bar_h))
            surf.fill((22, 24, 33))
            n = len(state)
            for i, v in enumerate(state):
                x0 = int(i * HEAT_W / n)
                w0 = max(1, int((i + 1) * HEAT_W / n) - x0)
                pygame.draw.rect(surf, _act_col(v), (x0, 0, w0, bar_h))

            self.screen.blit(surf, (HEAT_X, y))
            pygame.draw.rect(self.screen, BORDER, (HEAT_X, y, HEAT_W, bar_h), 1)
            y += bar_h + gap

    # ── rolling plots ──────────────────────────────────────────────────────────

    def _draw_plots(self) -> None:
        usable_h = WIN_H - STATS_H - 8
        plots = [
            ("fwd",    self.h_fwd,    GREEN,  (-1.0, 1.0)),
            ("turn",   self.h_turn,   BLUE,   (-1.0, 1.0)),
            ("reward", self.h_reward, YELLOW, (-0.6, 0.6)),
            ("rpe",    self.h_rpe,    ORANGE, (-0.5, 0.5)),
        ]
        n_plots = len(plots)
        gap     = 8
        plot_h  = (usable_h - gap * (n_plots + 1) - 14 * n_plots) // n_plots
        y       = 8

        self.screen.blit(
            self.font_sm.render(f"← {HISTORY} steps", True, DIM),
            (PLOT_X, y))
        y += 16

        for label, data, color, (lo, hi) in plots:
            arr = list(data)

            # Label + current value
            cur = arr[-1] if arr else 0.0
            self.screen.blit(
                self.font_sm.render(f"{label}  {cur:+.3f}", True, color),
                (PLOT_X, y))
            y += 14

            rect = pygame.Rect(PLOT_X, y, PLOT_W, plot_h)
            pygame.draw.rect(self.screen, PANEL,  rect)
            pygame.draw.rect(self.screen, BORDER, rect, 1)

            # Zero line
            zy = int(y + (1.0 - (0.0 - lo) / (hi - lo)) * plot_h)
            pygame.draw.line(self.screen, BORDER, (PLOT_X, zy), (PLOT_X + PLOT_W, zy))

            # Data polyline
            if len(arr) > 1:
                pts = []
                for i, v in enumerate(arr):
                    px = PLOT_X + int(i * PLOT_W / len(arr))
                    vc = float(np.clip(v, lo, hi))
                    py = int(y + (1.0 - (vc - lo) / (hi - lo)) * plot_h)
                    pts.append((px, py))
                pygame.draw.lines(self.screen, color, False, pts, 2)

            y += plot_h + gap

    # ── bottom stats bar ───────────────────────────────────────────────────────

    def _draw_stats_bar(self, step, fwd, turn, eat, reward, rpe) -> None:
        y    = WIN_H - STATS_H
        bar  = pygame.Rect(0, y, WIN_W, STATS_H)
        pygame.draw.rect(self.screen, PANEL,  bar)
        pygame.draw.line(self.screen, BORDER, (0, y), (WIN_W, y))

        w_norm = float(np.linalg.norm(self.brain.W_out))
        vp     = self.brain._valence_pred
        eat_s  = "[EAT]" if eat > 0.5 else "     "
        line   = (f"  step:{step:>8}    reward:{reward:+.4f}    rpe:{rpe:+.4f}    "
                  f"valence_pred:{vp:+.4f}    |W_out|:{w_norm:.5f}    "
                  f"fwd:{fwd:+.3f}    turn:{turn:+.3f}    {eat_s}")
        self.screen.blit(self.font_md.render(line, True, WHITE), (0, y + 6))

    # ── spinning alert ─────────────────────────────────────────────────────────

    def _draw_attractor_alert(self) -> None:
        if len(self.h_turn) < 60:
            return
        recent = list(self.h_turn)[-60:]
        mean_t = float(np.mean(recent))
        std_t  = float(np.std(recent))
        if abs(mean_t) > 0.5 and std_t < 0.25:
            direction = "RIGHT" if mean_t > 0 else "LEFT"
            msg  = f"  SPINNING {direction}  mean_turn={mean_t:+.2f}  std={std_t:.2f}  "
            surf = self.font_lg.render(msg, True, (8, 8, 12))
            bg   = pygame.Surface((surf.get_width() + 8, surf.get_height() + 6))
            bg.fill(RED)
            bg.blit(surf, (4, 3))
            self.screen.blit(bg, (WIN_W // 2 - bg.get_width() // 2, WIN_H - STATS_H - bg.get_height() - 6))


# ── Run loops ──────────────────────────────────────────────────────────────────

def _make_run_state():
    return {"step": 0, "eat_count": 0, "episodes": 0,
            "reward_sum": 0.0, "fwd_sum": 0.0, "turn_sum": 0.0}


def _tick(state, fwd, turn, eat, reward):
    state["reward_sum"] += reward
    state["eat_count"]  += int(eat > 0.5)
    state["fwd_sum"]    += fwd
    state["turn_sum"]   += turn
    state["step"]       += 1


def _maybe_log(state, brain, args, csv_writer):
    if state["step"] % args.log_every != 0:
        return
    _log(state["step"], state["reward_sum"], args.log_every,
         brain._valence_pred, float(np.linalg.norm(brain.W_out)),
         state["eat_count"], state["episodes"], csv_writer,
         state["fwd_sum"], state["turn_sum"])
    state["reward_sum"] = 0.0
    state["eat_count"]  = 0
    state["episodes"]   = 0
    state["fwd_sum"]    = 0.0
    state["turn_sum"]   = 0.0


def run_connected_viz(brain: EmbodiedBrain, viz: BrainViz,
                      args: argparse.Namespace) -> None:
    try:
        import zmq
        _zmq_again = zmq.Again
    except ImportError:
        _zmq_again = None

    from connector import AgentConnector  # noqa: PLC0415
    client     = AgentConnector()
    client.connect()
    csv_writer = _open_log(args.log_path) if args.log_path else None
    st         = _make_run_state()

    try:
        obs, reward, done, _ = client.recv_obs()
        while True:
            fwd, turn, eat = brain.forward(obs)
            rpe = reward - brain._valence_pred
            brain.learn(reward)
            if done:
                brain.reset_state()
                st["episodes"] += 1
            client.send_action((fwd, turn, eat))
            _tick(st, fwd, turn, eat, reward)
            viz.update(fwd, turn, reward, rpe)
            if not viz.render(st["step"], fwd, turn, eat, reward, rpe):
                break
            _maybe_log(st, brain, args, csv_writer)
            if args.save and st["step"] % args.save_every == 0:
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


def run_headless_viz(brain: EmbodiedBrain, viz: BrainViz,
                     args: argparse.Namespace) -> None:
    from env import World  # noqa: PLC0415
    world      = World(seed=args.seed)
    csv_writer = _open_log(args.log_path) if args.log_path else None
    st         = _make_run_state()
    prev_life  = 1.0
    life_idx   = brain._state_sl.start

    try:
        while st["step"] < args.headless:
            obs    = world.get_ai_observation()
            reward = float(world.ai.meters.valence)
            life   = float(obs[life_idx])
            done   = bool(life > 0.9 and prev_life < 0.1)
            prev_life = life

            fwd, turn, eat = brain.forward(obs)
            rpe = reward - brain._valence_pred
            brain.learn(reward)
            if done:
                brain.reset_state()
                st["episodes"] += 1
            world.step(ai_action=(fwd, turn, eat))

            _tick(st, fwd, turn, eat, reward)
            viz.update(fwd, turn, reward, rpe)
            if not viz.render(st["step"], fwd, turn, eat, reward, rpe):
                break
            _maybe_log(st, brain, args, csv_writer)
            if args.save and st["step"] % args.save_every == 0:
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
    from brain import LOG_EVERY, SAVE_EVERY  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="ESN brain with live 1080p pygame visualiser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python brain_viz.py                     connect to game.py --connect
  python brain_viz.py --headless 3600     headless 3600 steps + visualiser
  python brain_viz.py --load brain.npz    resume from checkpoint
""")
    parser.add_argument("--save",       metavar="PATH")
    parser.add_argument("--load",       metavar="PATH")
    parser.add_argument("--no-learn",   action="store_true")
    parser.add_argument("--headless",   type=int, default=0, metavar="N")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--log-every",  type=int, default=LOG_EVERY,  metavar="N")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, metavar="N")
    parser.add_argument("--log-path",   metavar="PATH")
    args = parser.parse_args()

    brain = EmbodiedBrain()
    if args.load:
        brain.load(args.load)
        print(f"Loaded weights from {args.load}")
    if args.no_learn:
        brain._learning_enabled = False

    viz = BrainViz(brain)   # spring layout computed here

    try:
        if args.headless:
            print(f"Running {args.headless} headless steps with visualiser…")
            run_headless_viz(brain, viz, args)
        else:
            print("Waiting for game connection…")
            run_connected_viz(brain, viz, args)
    finally:
        viz.close()


if __name__ == "__main__":
    main()
