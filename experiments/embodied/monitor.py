"""
AI sensory monitor — real-time terminal display of all connected AI entities.

Subscribes to the game's monitor broadcast socket (port 5555).
Does NOT spawn an agent body in the world.

Usage:
    # Terminal 1 — game with connector
    python game.py --connect

    # Terminal 2 — monitor (shows all connected brains)
    python monitor.py

    # Show a specific slot only
    python monitor.py --slot 1

Layout (one agent or --slot):
    ╭─ AI #0 ────────────────────────────────────────────────────────╮
    │  VISION  · · · ·  ██  ·  █  · · · ·  ████  · · ·             │
    │  TACTILE ·  ·  ·  ·  ·  ·  [L]  ●  [R]  ·  ·  ·             │
    │  METERS  Life ████████░  0.82  Sat +0.34  Val +0.12           │
    ╰────────────────────────────────────────────────────────────────╯
    ╭─ AI #1  (compact) ─────────────────────────────────────────────╮
    │  Life ████████░  0.71   Sat −0.10   Val −0.23                  │
    ╰────────────────────────────────────────────────────────────────╯
    step 1234   30 fps

Press Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import time
import sys
import os
import shutil
import struct

import numpy as np

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    raise SystemExit("rich is required:  pip install rich")

sys.path.insert(0, os.path.dirname(__file__))
from env import OBS_SIZE, N_RAYS, N_TOUCH_BODY, N_TOUCH_PRONGS
from connector import MonitorConnector

# ── Obs slice indices ──────────────────────────────────────────────────────────
_VIS_END      = N_RAYS * 2
_TAC_BODY_END = _VIS_END + N_TOUCH_BODY
_TAC_END      = _TAC_BODY_END + N_TOUCH_PRONGS
_LIFE_IDX     = _TAC_END
_SAT_IDX      = _TAC_END + 1
_VAL_IDX      = _TAC_END + 2

# ── Display helpers ────────────────────────────────────────────────────────────
_VIS_STYLE = {0: "grey30", 1: "grey70", 2: "yellow", 3: "red", 4: "green"}
_TYPE_CHARS = {0: "·", 1: "w", 2: "f", 3: "d", 4: "e"}
_BLOCKS = " ▁▂▃▄▅▆▇█"

def _vision_cols() -> int:
    term_w = shutil.get_terminal_size((100, 40)).columns
    return min(N_RAYS, max(40, term_w - 8))

def _tac_style(sig: float) -> str:
    if sig < 0.05:  return "grey23"
    if sig >= 0.9:  return "red"
    if sig >= 0.65: return "white"
    if sig >= 0.45: return "green"
    return "yellow"

def _vision_strip(obs: np.ndarray) -> Text:
    text      = Text()
    ray_types = obs[0:_VIS_END:2]
    ray_prox  = obs[1:_VIS_END:2]
    n_cols    = _vision_cols()
    indices   = np.round(np.linspace(0, N_RAYS - 1, n_cols)).astype(int)
    for i in indices:
        hit   = int(round(float(ray_types[i]) * 4))
        prox  = float(ray_prox[i])
        style = _VIS_STYLE.get(hit, "grey30")
        ch    = "·" if hit == 0 else _BLOCKS[max(1, min(8, int(prox * 8) + 1))]
        text.append(ch, style=style)
    return text

def _tactile_strip(obs: np.ndarray) -> Text:
    body   = obs[_VIS_END:_TAC_BODY_END]
    prongs = obs[_TAC_BODY_END:_TAC_END]
    half   = N_TOUCH_BODY // 2
    order  = list(range(half, 0, -1)) + [0] + list(range(1, half + 1))
    text   = Text()
    for idx in order:
        sig = float(body[idx])
        ch  = "·" if sig < 0.05 else _BLOCKS[max(1, min(8, int(sig * 8) + 1))]
        text.append(ch + " ", style=_tac_style(sig))
    l_sig, r_sig = float(prongs[0]), float(prongs[1])
    pch = lambda s: "·" if s < 0.05 else _BLOCKS[max(1, min(8, int(s * 8) + 1))]
    text.append("  ❬", style="grey50")
    text.append(pch(l_sig), style=_tac_style(l_sig))
    text.append("❭ ❬", style="grey50")
    text.append(pch(r_sig), style=_tac_style(r_sig))
    text.append("❭", style="grey50")
    return text

_BAR_W = 24

def _unipolar_bar(v: float, style: str) -> Text:
    n = max(0, min(_BAR_W, int(v * _BAR_W)))
    t = Text()
    t.append("█" * n, style=style)
    t.append("░" * (_BAR_W - n), style="grey23")
    return t

def _polar_bar(v: float, sneg: str, spos: str) -> Text:
    half = _BAR_W // 2
    v    = float(np.clip(v, -1, 1))
    t    = Text()
    t.append("◄", style="grey50")
    if v >= 0:
        fill = max(0, min(half, int(v * half)))
        t.append("░" * half, style="grey23")
        t.append("█" * fill, style=spos)
        t.append("░" * (half - fill), style="grey23")
    else:
        fill = max(0, min(half, int(-v * half)))
        t.append("░" * (half - fill), style="grey23")
        t.append("█" * fill, style=sneg)
        t.append("░" * half, style="grey23")
    t.append("►", style="grey50")
    return t

def _meters_table(obs: np.ndarray) -> Table:
    life = float(obs[_LIFE_IDX])
    sat  = float(obs[_SAT_IDX]) * 2 - 1
    val  = float(obs[_VAL_IDX]) * 2 - 1
    tbl  = Table.grid(padding=(0, 1))
    tbl.add_column(width=10)
    tbl.add_column(width=_BAR_W + 2)
    tbl.add_column(width=7)
    tbl.add_row(Text("Life",      style="bold grey85"), _unipolar_bar(life, "green" if life > 0.4 else "red"),       Text(f"{life:.2f}",  style="grey85"))
    tbl.add_row(Text("Satiation", style="bold grey85"), _polar_bar(sat, "dark_orange", "green"),                     Text(f"{sat:+.2f}",  style="dark_orange" if sat < 0 else "green"))
    tbl.add_row(Text("Valence",   style="bold grey85"), _polar_bar(val, "red", "medium_purple1"),                    Text(f"{val:+.2f}",  style="red" if val < 0 else "medium_purple1"))
    return tbl

def _compact_meters(obs: np.ndarray) -> Text:
    """One-line meter summary for secondary agents."""
    life = float(obs[_LIFE_IDX])
    sat  = float(obs[_SAT_IDX]) * 2 - 1
    val  = float(obs[_VAL_IDX]) * 2 - 1
    t    = Text()
    t.append("Life ", style="bold grey85")
    t.append(_unipolar_bar(life, "green" if life > 0.4 else "red"))
    t.append(f" {life:.2f}   ", style="grey85")
    t.append("Sat ", style="bold grey85")
    t.append(f"{sat:+.2f}   ", style="dark_orange" if sat < 0 else "green")
    t.append("Val ", style="bold grey85")
    t.append(f"{val:+.2f}", style="red" if val < 0 else "medium_purple1")
    return t


# ── Full layout ────────────────────────────────────────────────────────────────

def _build_display(
    step: int,
    agents: list,
    fps: float,
    focus_slot: int | None,
) -> Table:
    root = Table.grid(padding=(0, 0))
    root.add_column()

    n = len(agents)

    # Pick which slot gets full detail
    detail_slot = focus_slot if (focus_slot is not None and focus_slot < n) else 0

    for i, agent in enumerate(agents):
        obs    = agent["obs"]
        reward = agent["reward"]
        title  = f"[bold cyan]AI #{i}[/]"
        if n > 1:
            title += f"[grey50]  ({n} agents total)[/]" if i == detail_slot else ""

        if i == detail_slot:
            # Full detail panel
            inner = Table.grid(padding=(0, 0))
            inner.add_column()
            inner.add_row(Text("VISION  ", style="grey50") + _vision_strip(obs))
            inner.add_row(Text("TACTILE ", style="grey50") + _tactile_strip(obs))
            inner.add_row(_meters_table(obs))
            status = Text()
            status.append(f"step {step:>7}  reward ", style="grey60")
            status.append(f"{reward:+.3f}", style="red" if reward < 0 else "medium_purple1")
            status.append(f"  {fps:.0f} fps", style="grey60")
            inner.add_row(status)
            root.add_row(Panel(inner, title=title, title_align="left",
                               border_style="grey42", padding=(0, 1)))
        else:
            # Compact panel for other agents
            compact = _compact_meters(obs)
            compact.append(f"   step {step:>7}  reward ", style="grey60")
            compact.append(f"{reward:+.3f}", style="red" if reward < 0 else "medium_purple1")
            root.add_row(Panel(compact, title=title,
                               title_align="left", border_style="grey30",
                               padding=(0, 1)))

    if n == 0:
        root.add_row(Text("  Waiting for agents…", style="grey50"))

    return root


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent sensory monitor (read-only).")
    parser.add_argument("--slot",  type=int, default=None, metavar="N",
                        help="Show full detail for slot N (default: 0)")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=None, metavar="PORT",
                        help="Monitor port (default: MONITOR_PORT from connector.py)")
    args = parser.parse_args()

    from connector import MONITOR_PORT
    port = args.port or MONITOR_PORT

    console = Console()
    console.print(f"[grey60]Connecting to monitor socket {args.host}:{port}…[/]")

    mon = MonitorConnector(host=args.host, port=port)
    mon.connect()
    console.print("[green]Connected.[/]  Waiting for first frame…")

    step, agents = mon.recv(timeout_ms=15_000)
    step_prev = step
    t_prev    = time.monotonic()
    fps       = 0.0

    with Live(console=console, refresh_per_second=20, screen=True) as live:
        try:
            while True:
                step, agents = mon.recv(timeout_ms=500)
                now = time.monotonic()
                t_delta = max(1e-6, now - t_prev)
                if t_delta > 0.2:
                    fps       = (step - step_prev) / t_delta
                    step_prev = step
                    t_prev    = now
                live.update(_build_display(step, agents, fps, args.slot))
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")

    mon.close()
    console.print("[grey60]Monitor closed.[/]")


if __name__ == "__main__":
    main()
