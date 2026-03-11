"""
AI sensory monitor — real-time terminal display of the AI entity's full
sensory experience, received via the connector.

Usage:
    # Terminal 1 — start the game with connector enabled
    python game.py --connect

    # Terminal 2 — start the monitor
    python monitor.py

Press Ctrl-C to quit the monitor; the game keeps running.

Layout:
    ╭─ VISION ───────────────────────────────────────────────────────╮
    │  · · · ·  ██  ·  █  · · · ·  ████  · · ·  █  · · · · · · ·  │
    ╰────────────────────────────────────────────────────────────────╯
    ╭─ TACTILE ──────────────────────────────────────────────────────╮
    │  ·  ·  ·  ·  ·  ·  [L]  ●  [R]  ·  ·  ·  ·  ·  ·  ·  ·  ·  │
    ╰────────────────────────────────────────────────────────────────╯
    ╭─ METERS ───────────────────────────────────────────────────────╮
    │  Life      ████████████████████░░░░░░░  0.82                   │
    │  Satiation ◄░░░░░░░░░░░░░░░███████████► +0.34                  │
    │  Valence   ◄░░░░░░░░░░░░░░░███████████► +0.12                  │
    ╰────────────────────────────────────────────────────────────────╯
    step 1234   reward +0.12   30 fps
"""

from __future__ import annotations

import time
import sys
import os
import shutil

import numpy as np

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.columns import Columns
except ImportError:
    raise SystemExit("rich is required:  pip install rich")

sys.path.insert(0, os.path.dirname(__file__))
from env import (
    OBS_SIZE, N_RAYS, N_TOUCH_BODY, N_TOUCH_PRONGS,
    TOUCH_WALL, TOUCH_FOOD, TOUCH_ENTITY, TOUCH_DANGER,
    PRONG_ANGLE,
)
from connector import AgentConnector

# ── Obs slice indices ──────────────────────────────────────────────────────────
_VIS_END    = N_RAYS * 2            # 242
_TAC_BODY_END = _VIS_END + N_TOUCH_BODY      # 258
_TAC_END    = _TAC_BODY_END + N_TOUCH_PRONGS # 260
_LIFE_IDX   = _TAC_END              # 260
_SAT_IDX    = _TAC_END + 1          # 261
_VAL_IDX    = _TAC_END + 2          # 262

# ── Display constants ──────────────────────────────────────────────────────────
# Vision strip width: fill most of the terminal, capped at N_RAYS (no upsampling).
def _vision_cols() -> int:
    term_w = shutil.get_terminal_size((100, 40)).columns
    # Subtract panel borders (4) + padding (2) + a small margin
    return min(N_RAYS, max(40, term_w - 8))

# ANSI-compatible rich colour tags per object type
_VIS_STYLE = {
    0: "grey30",           # nothing
    1: "grey70",           # wall
    2: "yellow",           # food
    3: "red",              # danger
    4: "green",            # other entity
}

_TYPE_CHARS = {0: "·", 1: "w", 2: "f", 3: "d", 4: "e"}

# Tactile colour by signal strength (thresholds match _tactile_color in game.py)
def _tac_style(sig: float) -> str:
    if sig < 0.05:   return "grey23"
    if sig >= 0.9:   return "red"
    if sig >= 0.65:  return "white"
    if sig >= 0.45:  return "green"
    return "yellow"


# ── Vision strip ──────────────────────────────────────────────────────────────

def _vision_strip(obs: np.ndarray) -> Text:
    """
    One character per display column, colour = object type, brightness = proximity.
    Uses block characters ▁▂▃▄▅▆▇█ to encode proximity height.
    """
    blocks = " ▁▂▃▄▅▆▇█"
    text = Text()
    ray_types = obs[0:_VIS_END:2]
    ray_prox  = obs[1:_VIS_END:2]

    # Downsample N_RAYS → terminal width
    n_cols  = _vision_cols()
    indices = np.round(np.linspace(0, N_RAYS - 1, n_cols)).astype(int)

    for i in indices:
        t    = float(ray_types[i])
        prox = float(ray_prox[i])
        # Decode hit type from normalised value
        hit = int(round(t * 4))   # 0..4
        style = _VIS_STYLE.get(hit, "grey30")
        if hit == 0:
            ch = "·"
        else:
            block_idx = max(1, min(8, int(prox * 8) + 1))
            ch = blocks[block_idx]
        text.append(ch, style=style)

    return text


def _vision_numbers(obs: np.ndarray) -> tuple[Text, Text]:
    """
    Type-code row and proximity-digit row to display beneath the vision strip.

    type row : · w f d e  (one char per display column, same colour as strip)
    prox row : 0–9        (tenths of proximity, · when nothing detected)
    """
    ray_types = obs[0:_VIS_END:2]
    ray_prox  = obs[1:_VIS_END:2]
    n_cols    = _vision_cols()
    indices   = np.round(np.linspace(0, N_RAYS - 1, n_cols)).astype(int)

    type_row = Text()
    prox_row = Text()

    for i in indices:
        t    = float(ray_types[i])
        prox = float(ray_prox[i])
        hit  = int(round(t * 4))
        style = _VIS_STYLE.get(hit, "grey30")

        type_row.append(_TYPE_CHARS.get(hit, "?"), style=style)
        if hit == 0:
            prox_row.append("·", style="grey30")
        else:
            prox_row.append(str(int(round(prox * 9))), style=style)

    return type_row, prox_row


# ── Tactile strip ─────────────────────────────────────────────────────────────

def _tactile_strip(obs: np.ndarray) -> Text:
    """
    16 body receptors unfolded flat (centre = forward), plus two prong
    indicators floating in their angular position above the strip.
    """
    body   = obs[_VIS_END:_TAC_BODY_END]   # (16,)
    prongs = obs[_TAC_BODY_END:_TAC_END]   # (2,)

    blocks = " ▁▂▃▄▅▆▇█"
    text = Text()

    half = N_TOUCH_BODY // 2

    # Build ordered list: rear-left … forward … rear-right (unfolded ring)
    # receptor index 0 = forward, 1..half = CW (right side), N-1..half+1 = CCW (left side)
    # Displayed left→right: rear-left (N//2), …, 1, 0 (forward), 1, …, rear-right (N//2)
    order = list(range(half, 0, -1)) + [0] + list(range(1, half + 1))

    for display_pos, idx in enumerate(order):
        sig   = float(body[idx])
        style = _tac_style(sig)
        if sig < 0.05:
            ch = "·"
        else:
            block_idx = max(1, min(8, int(sig * 8) + 1))
            ch = blocks[block_idx]
        text.append(ch + " ", style=style)

    # Append prong indicators at the end as [L] and [R] with colour
    l_sig = float(prongs[0])
    r_sig = float(prongs[1])
    text.append("  ", style="grey23")
    text.append("❬", style="grey50")
    text.append(_prong_char(l_sig), style=_tac_style(l_sig))
    text.append("❭ ", style="grey50")
    text.append("❬", style="grey50")
    text.append(_prong_char(r_sig), style=_tac_style(r_sig))
    text.append("❭", style="grey50")

    return text


def _tactile_numbers(obs: np.ndarray) -> Text:
    """
    Numeric row beneath the tactile strip: 16 body values + 2 prong values.
    Each body value shown as X.X (one decimal), coloured to match the strip.
    Prong values shown as X.X at the end, matching the ❬❭ indicator positions.
    """
    body   = obs[_VIS_END:_TAC_BODY_END]
    prongs = obs[_TAC_BODY_END:_TAC_END]

    half  = N_TOUCH_BODY // 2
    order = list(range(half, 0, -1)) + [0] + list(range(1, half + 1))

    text = Text()
    for idx in order:
        sig = float(body[idx])
        text.append(f"{sig:.1f}", style=_tac_style(sig))
        text.append(" ", style="grey23")

    l_sig = float(prongs[0])
    r_sig = float(prongs[1])
    text.append("  ", style="grey23")
    text.append(f"{l_sig:.1f}", style=_tac_style(l_sig))
    text.append("  ", style="grey23")
    text.append(f"{r_sig:.1f}", style=_tac_style(r_sig))

    return text


def _prong_char(sig: float) -> str:
    blocks = "·▁▃▅▇█"
    if sig < 0.05:
        return "·"
    idx = max(1, min(5, int(sig * 5) + 1))
    return blocks[idx]


# ── Meter bars ────────────────────────────────────────────────────────────────

_BAR_W = 32   # characters for filled bar region

def _unipolar_bar(value: float, style: str) -> Text:
    """0→1 bar."""
    filled = max(0, min(_BAR_W, int(value * _BAR_W)))
    text = Text()
    text.append("█" * filled,          style=style)
    text.append("░" * (_BAR_W - filled), style="grey23")
    return text


def _polar_bar(value: float, style_neg: str, style_pos: str) -> Text:
    """−1→+1 bar centred at 0."""
    half = _BAR_W // 2
    v    = float(np.clip(value, -1.0, 1.0))
    text = Text()
    if v >= 0:
        fill = max(0, min(half, int(v * half)))
        text.append("◄", style="grey50")
        text.append("░" * half,        style="grey23")
        text.append("█" * fill,        style=style_pos)
        text.append("░" * (half - fill), style="grey23")
        text.append("►", style="grey50")
    else:
        fill = max(0, min(half, int(-v * half)))
        text.append("◄", style="grey50")
        text.append("░" * (half - fill), style="grey23")
        text.append("█" * fill,        style=style_neg)
        text.append("░" * half,        style="grey23")
        text.append("►", style="grey50")
    return text


_LABEL_STYLE = "bold grey85"


def _meters_table(obs: np.ndarray) -> Table:
    life = float(obs[_LIFE_IDX])
    sat  = float(obs[_SAT_IDX]) * 2.0 - 1.0   # denormalise → [-1, 1]
    val  = float(obs[_VAL_IDX]) * 2.0 - 1.0

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(width=10)
    tbl.add_column(width=_BAR_W + 2)
    tbl.add_column(width=6)

    life_bar = _unipolar_bar(life, "green" if life > 0.4 else "red")
    tbl.add_row(
        Text("Life", style=_LABEL_STYLE),
        life_bar,
        Text(f"{life:.2f}", style="grey85"),
    )

    sat_bar = _polar_bar(sat, "dark_orange", "green")
    tbl.add_row(
        Text("Satiation", style=_LABEL_STYLE),
        sat_bar,
        Text(f"{sat:+.2f}", style="dark_orange" if sat < 0 else "green"),
    )

    val_bar = _polar_bar(val, "red", "medium_purple1")
    tbl.add_row(
        Text("Valence", style=_LABEL_STYLE),
        val_bar,
        Text(f"{val:+.2f}", style="red" if val < 0 else "medium_purple1"),
    )

    return tbl


# ── Raw values block ──────────────────────────────────────────────────────────

def _values_block(obs: np.ndarray) -> Table:
    """
    Full numerical dump of all sensor values, displayed below the status bar.

    VIS type  · · w · f · · ·  …   (· nothing  w wall  f food  d danger  e entity)
    VIS prox  0.00 0.00 0.73 …
    TAC body  0.00 0.00 0.70 0.30 …  (receptor order: rear-L → fwd → rear-R)
    TAC prng  L:0.00  R:0.00
    METERS    life:0.82  sat:+0.34  val:+0.12
    """
    ray_types = obs[0:_VIS_END:2]
    ray_prox  = obs[1:_VIS_END:2]
    body      = obs[_VIS_END:_TAC_BODY_END]
    prongs    = obs[_TAC_BODY_END:_TAC_END]
    life      = float(obs[_LIFE_IDX])
    sat       = float(obs[_SAT_IDX]) * 2.0 - 1.0
    val       = float(obs[_VAL_IDX]) * 2.0 - 1.0

    n_cols  = _vision_cols()
    indices = np.round(np.linspace(0, N_RAYS - 1, n_cols)).astype(int)

    tbl = Table.grid(padding=(0, 0))
    tbl.add_column(width=10, style="grey50")   # label
    tbl.add_column()                            # values

    # Vision type row
    vtype = Text()
    for i in indices:
        hit = int(round(float(ray_types[i]) * 4))
        vtype.append(_TYPE_CHARS.get(hit, "?") + " ", style=_VIS_STYLE.get(hit, "grey30"))
    tbl.add_row(Text("VIS type", style="grey50"), vtype)

    # Vision proximity row
    vprox = Text()
    for i in indices:
        hit  = int(round(float(ray_types[i]) * 4))
        prox = float(ray_prox[i])
        style = _VIS_STYLE.get(hit, "grey30") if hit > 0 else "grey30"
        vprox.append(f"{prox:.2f} ", style=style)
    tbl.add_row(Text("VIS prox", style="grey50"), vprox)

    # Tactile body row (unfolded, same order as strip)
    half  = N_TOUCH_BODY // 2
    order = list(range(half, 0, -1)) + [0] + list(range(1, half + 1))
    tbody = Text()
    for idx in order:
        sig = float(body[idx])
        tbody.append(f"{sig:.2f} ", style=_tac_style(sig))
    tbl.add_row(Text("TAC body", style="grey50"), tbody)

    # Prong row
    l_sig, r_sig = float(prongs[0]), float(prongs[1])
    tprng = Text()
    tprng.append("L:", style="grey50")
    tprng.append(f"{l_sig:.2f}", style=_tac_style(l_sig))
    tprng.append("  R:", style="grey50")
    tprng.append(f"{r_sig:.2f}", style=_tac_style(r_sig))
    tbl.add_row(Text("TAC prng", style="grey50"), tprng)

    # Meters row
    mtext = Text()
    mtext.append("life:", style="grey50")
    mtext.append(f"{life:.3f}", style="green" if life > 0.4 else "red")
    mtext.append("  sat:", style="grey50")
    mtext.append(f"{sat:+.3f}", style="dark_orange" if sat < 0 else "green")
    mtext.append("  val:", style="grey50")
    mtext.append(f"{val:+.3f}", style="red" if val < 0 else "medium_purple1")
    tbl.add_row(Text("METERS", style="grey50"), mtext)

    return tbl


# ── Full layout ───────────────────────────────────────────────────────────────

def _build_display(
    obs: np.ndarray,
    reward: float,
    done: bool,
    step: int,
    fps: float,
) -> Table:
    root = Table.grid(padding=(0, 0))
    root.add_column()

    # Vision
    root.add_row(Panel(_vision_strip(obs), title="[bold cyan]VISION[/]",
                       title_align="left", border_style="grey42",
                       padding=(0, 1)))

    # Tactile
    root.add_row(Panel(_tactile_strip(obs), title="[bold cyan]TACTILE[/]  "
                       "[grey50](· no signal  [yellow]▇[/] food  "
                       "[green]▇[/] entity  [white]▇[/] wall  [red]▇[/] danger  "
                       "❬❭ prongs)[/]",
                       title_align="left", border_style="grey42",
                       padding=(0, 1)))

    # Meters
    mtbl = _meters_table(obs)
    root.add_row(Panel(mtbl, title="[bold cyan]METERS[/]",
                       title_align="left", border_style="grey42",
                       padding=(0, 1)))

    # Status bar
    status = Text()
    status.append(f" step {step:>7}   ", style="grey60")
    status.append("reward ", style="grey60")
    status.append(f"{reward:+.3f}", style="red" if reward < 0 else "medium_purple1")
    status.append(f"   {fps:.0f} fps", style="grey60")
    if done:
        status.append("   RESET", style="bold red")
    root.add_row(status)

    # Raw numerical values
    root.add_row(_values_block(obs))

    return root


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console = Console()
    client  = AgentConnector()

    console.print("[grey60]Connecting to game...[/]")
    client.connect()
    console.print("[green]Connected.[/]  Waiting for first frame…")

    # Receive first frame to confirm connection
    obs, reward, done, step = client.recv_obs(timeout_ms=10_000)

    step_prev = step
    t_prev    = time.monotonic()
    fps       = 0.0

    with Live(console=console, refresh_per_second=20, screen=True) as live:
        try:
            while True:
                obs, reward, done, step = client.recv_obs(timeout_ms=500)
                now = time.monotonic()

                # Estimate FPS from step counter jumps
                t_delta = max(1e-6, now - t_prev)
                if t_delta > 0.2:
                    fps       = (step - step_prev) / t_delta
                    step_prev = step
                    t_prev    = now

                display = _build_display(obs, reward, done, step, fps)
                live.update(display)

        except KeyboardInterrupt:
            pass
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")

    client.close()
    console.print("[grey60]Monitor closed.[/]")


if __name__ == "__main__":
    main()
