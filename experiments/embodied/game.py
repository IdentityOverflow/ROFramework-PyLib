"""
2D Embodied AI Environment — interactive viewer.

Controls:
  Arrow keys      move player  (while player is active)
  Space           eat          (player consumes overlapping food)
  T               toggle teleop of AI slot 0 using keyboard input
  Tab             toggle player on / off
  P               pause / unpause
  R               reset world  (respawns everything)
  Escape / Q      quit

AI entity (blue) uses zero actions by default.
Replace `ai_action` in the main loop with your model output:

    obs     = world.get_ai_observation()      # (263,) numpy array, all in [0,1]
    out     = model.step(obs)                 # (3,): [forward, turn, eat]
    ai_action = tuple(float(v) for v in out)

Requires:  pygame  (pip install pygame)
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import List

try:
    import pygame
except ImportError:
    raise SystemExit("pygame is required:  pip install pygame")

sys.path.insert(0, os.path.dirname(__file__))
from env import (
    World, Entity,
    WORLD_W, WORLD_H,
    VISION_RANGE, VISION_HALF_ANGLE, N_RAYS,
    FOOD_RADIUS, DANGER_RADIUS,
    HIT_NONE, HIT_WALL, HIT_FOOD, HIT_DANGER, HIT_OTHER,
    PRONG_ANGLE, PRONG_LENGTH, PRONG_BASE_W,
    N_TOUCH_BODY, N_TOUCH_PRONGS,
    TOUCH_WALL, TOUCH_FOOD, TOUCH_ENTITY, TOUCH_DANGER,
)

# ── Layout ────────────────────────────────────────────────────────────────────
FPS        = 60
HUD_HEIGHT = 140
SCREEN_W   = WORLD_W
SCREEN_H   = WORLD_H + HUD_HEIGHT

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG           = ( 26,  28,  38)
C_BORDER       = ( 65,  68,  88)
C_HUD_BG       = ( 18,  20,  28)
C_HUD_DIV      = ( 45,  48,  65)
C_TEXT         = (200, 202, 215)
C_TEXT_DIM     = ( 85,  88, 110)

C_AI           = ( 80, 140, 220)
C_AI_RING      = (120, 175, 255)
C_PLAYER       = ( 70, 210, 110)
C_PLAYER_OFF   = ( 48,  80,  52)
C_PLAYER_RING  = (110, 245, 150)

C_FOOD         = (240, 210,  55)
C_FOOD_RING    = (255, 235, 100)
C_DANGER       = (210,  55,  55)
C_DANGER_FILL  = ( 90,  22,  22)

C_METER_BG     = ( 38,  40,  52)
C_LIFE         = ( 55, 195,  75)
C_SAT_NEG      = (215, 135,  40)
C_SAT_POS      = ( 55, 195,  75)
C_VAL_NEG      = (215,  55,  55)
C_VAL_POS      = (155,  75, 215)

C_DEATH_FLASH  = (180,  30,  30)

# Per-slot AI palettes — slot 0 reuses C_AI / C_AI_RING
_AI_PALETTES = [
    (C_AI,                    C_AI_RING),                  # slot 0: blue
    ((210, 100,  50), (255, 150,  80)),                     # slot 1: orange
    ((160,  80, 200), (200, 130, 255)),                     # slot 2: purple
    (( 60, 200, 180), (100, 240, 220)),                     # slot 3: teal
    ((200, 200,  50), (255, 255,  80)),                     # slot 4: yellow
    ((200,  80, 150), (255, 130, 195)),                     # slot 5: pink
]

# Ray hit → colour
RAY_COLORS = {
    HIT_NONE:   ( 45,  50,  70),
    HIT_WALL:   (130, 132, 155),
    HIT_FOOD:   (240, 210,  55),
    HIT_DANGER: (210,  55,  55),
    HIT_OTHER:  ( 70, 210, 110),
}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_meter(
    surf: pygame.Surface,
    label: str,
    value: float,
    x: int, y: int,
    color: tuple,
    width: int = 110,
    height: int = 10,
) -> None:
    font = pygame.font.SysFont("monospace", 11)
    pygame.draw.rect(surf, C_METER_BG, (x, y, width, height), border_radius=3)
    fill = max(2, int(width * float(np.clip(value, 0, 1))))
    pygame.draw.rect(surf, color, (x, y, fill, height), border_radius=3)
    pygame.draw.rect(surf, (75, 78, 100), (x, y, width, height), 1, border_radius=3)
    txt = font.render(f"{label}  {value:.2f}", True, C_TEXT)
    surf.blit(txt, (x + width + 6, y))


def draw_polar_meter(
    surf: pygame.Surface,
    label: str,
    value: float,
    x: int, y: int,
    color_neg: tuple,
    color_pos: tuple,
    width: int = 110,
    height: int = 10,
) -> None:
    font = pygame.font.SysFont("monospace", 11)
    pygame.draw.rect(surf, C_METER_BG, (x, y, width, height), border_radius=3)
    mid = width // 2
    v   = float(np.clip(value, -1.0, 1.0))
    if v > 0:
        fill = max(2, int(mid * v))
        pygame.draw.rect(surf, color_pos, (x + mid, y, fill, height), border_radius=3)
    elif v < 0:
        fill = max(2, int(mid * (-v)))
        pygame.draw.rect(surf, color_neg, (x + mid - fill, y, fill, height), border_radius=3)
    pygame.draw.line(surf, (110, 115, 140), (x + mid, y), (x + mid, y + height - 1))
    pygame.draw.rect(surf, (75, 78, 100), (x, y, width, height), 1, border_radius=3)
    txt = font.render(f"{label}  {value:+.2f}", True, C_TEXT)
    surf.blit(txt, (x + width + 6, y))


def _tactile_color(sig: float) -> tuple:
    """Map signal strength to a display colour."""
    if sig < 0.05:
        return (55, 58, 78)          # no signal
    if sig >= 0.9:
        return (215, 55, 55)         # danger: red
    if sig >= 0.65:
        return (180, 182, 200)       # wall: light grey
    if sig >= 0.45:
        return (70, 210, 110)        # entity: green
    return (240, 210, 55)            # food: yellow


def _draw_vision_strip(
    surf: pygame.Surface,
    entity: Entity,
    world: World,
    others: List[Entity],
    sx: int, vy: int, strip_w: int, strip_h: int,
) -> None:
    angles = np.linspace(
        entity.heading - VISION_HALF_ANGLE,
        entity.heading + VISION_HALF_ANGLE,
        N_RAYS,
    )
    for i, angle in enumerate(angles):
        hit_type, dist = world._cast_ray(entity, angle, others)
        prox  = (1.0 - dist / VISION_RANGE) if hit_type != HIT_NONE else 0.0
        base  = RAY_COLORS[hit_type]
        alpha = 0.25 if hit_type == HIT_NONE else 0.35 + 0.65 * prox
        color = tuple(int(c * alpha) for c in base)
        cx = sx + int(i * strip_w / max(N_RAYS - 1, 1))
        cy = vy + strip_h // 2
        r  = 1 if hit_type == HIT_NONE else max(1, int(1 + 2 * prox))
        pygame.draw.circle(surf, color, (cx, cy), r)


def draw_sensor_strips(
    surf: pygame.Surface,
    entity: Entity,
    world: World,
    others: List[Entity],
    x: int, y: int,
    strip_w: int,
    font_s: pygame.font.Font,
) -> None:
    """
    Two horizontal sensor strips (first-person view), each strip_w pixels wide.

    Strip 1 (top)    — vision rays.  Centre dot = forward ray.  Colour by hit type.
    Strip 2 (bottom) — tactile body receptors.  Centre = forward, sides wrap to rear.
                        Two extra dots at far right = prong signals.

    Both strips: left edge = leftmost sensor, right edge = rightmost sensor,
    centre = directly ahead.  Dot brightness scales with signal strength.
    """
    DOT_STEP = max(1, strip_w // max(N_RAYS, N_TOUCH_BODY + 2))
    STRIP_H  = 10

    lbl_w = 52   # pixels reserved for label to the left

    sx = x + lbl_w    # strip left edge
    vy = y            # vision strip y
    ty = y + 20       # tactile strip y

    # ── labels ────────────────────────────────────────────────────────────────
    surf.blit(font_s.render("vision", True, C_TEXT_DIM), (x, vy + 1))
    surf.blit(font_s.render("touch",  True, C_TEXT_DIM), (x, ty + 1))

    # Background track
    pygame.draw.rect(surf, C_METER_BG, (sx, vy, strip_w, STRIP_H), border_radius=2)
    pygame.draw.rect(surf, C_METER_BG, (sx, ty, strip_w + DOT_STEP * 4, STRIP_H), border_radius=2)

    # ── vision strip ──────────────────────────────────────────────────────────
    _draw_vision_strip(surf, entity, world, others, sx, vy, strip_w, STRIP_H)

    # Centre marker (forward direction)
    mid_x = sx + strip_w // 2
    pygame.draw.line(surf, (80, 83, 108), (mid_x, vy), (mid_x, vy + STRIP_H - 1))

    # ── tactile body strip ────────────────────────────────────────────────────
    # Receptor 0 = forward.  Layout: centre=forward, left=CCW, right=CW.
    # The two halves of the circle (left/right of forward) are unfolded flat.
    # Receptor N//2 (opposite of forward = rear) appears at both edges, split.
    body = entity.tactile.body
    half = N_TOUCH_BODY // 2

    for i in range(N_TOUCH_BODY):
        sig = float(body[i])
        color = _tactile_color(sig)
        # Map receptor index to strip x:
        #   i=0   → centre  (forward)
        #   i=1…half  → right of centre  (CW / rightward)
        #   i=N-1…half+1 → left of centre  (CCW / leftward)
        if i == 0:
            rx = sx + strip_w // 2
        elif i <= half:
            rx = sx + strip_w // 2 + int(i * (strip_w // 2) / half)
        else:
            steps_from_right_edge = N_TOUCH_BODY - i
            rx = sx + strip_w // 2 - int(steps_from_right_edge * (strip_w // 2) / half)
        cy_t = ty + STRIP_H // 2
        r  = max(2, int(2 + 2 * sig))
        pygame.draw.circle(surf, color, (rx, cy_t), r)

    # ── prong dots — positioned at ±PRONG_ANGLE from centre, slightly above ────
    # PRONG_ANGLE rad from forward maps to ~1.5 receptor spacings from centre.
    prong_offset = int(PRONG_ANGLE * strip_w / (2 * np.pi))
    prong_sides  = (("L", -1), ("R", +1))
    for lbl, sign in prong_sides:
        pi_idx = 0 if sign < 0 else 1
        sig    = float(entity.tactile.prongs[pi_idx])
        color  = _tactile_color(sig)
        rx     = mid_x + sign * prong_offset
        py_    = ty - 7     # sit just above the tactile strip
        r      = max(4, int(4 + 3 * sig))
        pygame.draw.circle(surf, color, (rx, py_), r)
        pygame.draw.circle(surf, (100, 103, 130), (rx, py_), r, 1)
        surf.blit(font_s.render(lbl, True, C_TEXT_DIM), (rx - 3, py_ + r + 2))

    # Centre marker
    pygame.draw.line(surf, (80, 83, 108), (mid_x, ty), (mid_x, ty + STRIP_H - 1))


def draw_prongs(
    surf: pygame.Surface,
    entity: Entity,
    color: tuple,
) -> None:
    cx, cy = entity.x, entity.y
    r, h   = entity.radius, entity.heading
    for side in (-1.0, 1.0):
        angle = h + side * PRONG_ANGLE
        tip_x = cx + (r + PRONG_LENGTH) * np.cos(angle)
        tip_y = cy + (r + PRONG_LENGTH) * np.sin(angle)
        b1_x  = cx + r * np.cos(angle - PRONG_BASE_W)
        b1_y  = cy + r * np.sin(angle - PRONG_BASE_W)
        b2_x  = cx + r * np.cos(angle + PRONG_BASE_W)
        b2_y  = cy + r * np.sin(angle + PRONG_BASE_W)
        pts = [(int(tip_x), int(tip_y)), (int(b1_x), int(b1_y)), (int(b2_x), int(b2_y))]
        pygame.draw.polygon(surf, color, pts)


def draw_entity(
    surf: pygame.Surface,
    entity: Entity,
    color: tuple,
    ring: tuple,
) -> None:
    cx, cy = int(entity.x), int(entity.y)
    r = int(entity.radius)
    pygame.draw.circle(surf, color, (cx, cy), r)
    pygame.draw.circle(surf, ring,  (cx, cy), r, 2)
    draw_prongs(surf, entity, ring)
    hx = int(cx + np.cos(entity.heading) * (r - 4))
    hy = int(cy + np.sin(entity.heading) * (r - 4))
    pygame.draw.circle(surf, (255, 255, 255), (hx, hy), 3)


def draw_food(surf: pygame.Surface, food) -> None:
    cx, cy = int(food.x), int(food.y)
    r = int(FOOD_RADIUS)
    pygame.draw.circle(surf, C_FOOD,      (cx, cy), r)
    pygame.draw.circle(surf, C_FOOD_RING, (cx, cy), r, 1)


def draw_danger(surf: pygame.Surface, danger) -> None:
    cx, cy = int(danger.x), int(danger.y)
    r = int(DANGER_RADIUS)
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(surf, C_DANGER_FILL, pts)
    pygame.draw.polygon(surf, C_DANGER,      pts, 2)


def draw_vision(
    surf: pygame.Surface,
    ray_surf: pygame.Surface,
    entity: Entity,
    world: World,
    others: List[Entity],
) -> None:
    ray_surf.fill((0, 0, 0, 0))
    angles = np.linspace(
        entity.heading - VISION_HALF_ANGLE,
        entity.heading + VISION_HALF_ANGLE,
        N_RAYS,
    )
    ox, oy = int(entity.x), int(entity.y)
    for angle in angles:
        hit_type, dist = world._cast_ray(entity, angle, others)
        proximity = (1.0 - dist / VISION_RANGE) if hit_type != HIT_NONE else 0.0
        base = RAY_COLORS[hit_type]
        if hit_type == HIT_NONE:
            alpha_line = 60
        else:
            alpha_line = int(80 + 160 * proximity)
        ex = int(entity.x + np.cos(angle) * dist)
        ey = int(entity.y + np.sin(angle) * dist)
        pygame.draw.line(ray_surf, (*base, alpha_line), (ox, oy), (ex, ey), 1)
        if hit_type != HIT_NONE:
            dot_alpha = int(150 + 105 * proximity)
            dot_r = max(2, int(2 + 4 * proximity))
            pygame.draw.circle(ray_surf, (*base, dot_alpha), (ex, ey), dot_r)
    surf.blit(ray_surf, (0, 0))


def draw_entity_hud(
    surf: pygame.Surface,
    entity: Entity,
    world: World,
    others: List[Entity],
    label: str,
    color: tuple,
    font_s: pygame.font.Font,
    font_m: pygame.font.Font,
    x: int,
    y: int,
    panel_w: int,
) -> None:
    """Draw meters + sensor strips for one entity."""
    surf.blit(font_m.render(label, True, color), (x, y))

    m  = entity.meters
    my = y + 18

    draw_meter(      surf, "Life    ", m.life,      x, my,      C_LIFE)
    draw_polar_meter(surf, "Satiation", m.satiation, x, my + 16, C_SAT_NEG, C_SAT_POS)
    draw_polar_meter(surf, "Valence  ", m.valence,   x, my + 32, C_VAL_NEG, C_VAL_POS)

    # Sensor strips (vision + tactile) — full panel width, below meters
    strip_w = panel_w - 60   # leave margin for labels + prong dots
    draw_sensor_strips(surf, entity, world, others,
                       x, my + 56, strip_w, font_s)


def draw_hud(
    surf: pygame.Surface,
    world: World,
    font_s: pygame.font.Font,
    font_m: pygame.font.Font,
    slot_names: dict | None = None,
    teleop_enabled: bool = False,
) -> None:
    top = WORLD_H
    pygame.draw.rect(surf, C_HUD_BG, (0, top, SCREEN_W, HUD_HEIGHT))
    pygame.draw.line(surf, C_BORDER, (0, top), (SCREEN_W, top), 1)

    panel_w = SCREEN_W // 2

    # Panel 1: player
    pl_others = [a for a in world.agents if a is not None]
    p_color   = C_PLAYER if world.player_active else C_PLAYER_OFF
    draw_entity_hud(surf, world.player, world, pl_others,
                    "PLAYER", p_color,
                    font_s, font_m, x=14, y=top + 8,
                    panel_w=panel_w - 14)

    pygame.draw.line(surf, C_HUD_DIV,
                     (panel_w, top + 6), (panel_w, top + HUD_HEIGHT - 6), 1)

    # Panel 2: controls + agent status
    sx = panel_w + 14
    sy = top + 8
    player_on = world.player_active
    surf.blit(font_m.render(
        "PLAYER: " + ("ON " if player_on else "OFF"),
        True, C_PLAYER if player_on else C_TEXT_DIM),
        (sx, sy))

    teleop_label = "TELEOP: AI #0" if teleop_enabled else "TELEOP: OFF"
    teleop_color = C_AI_RING if teleop_enabled else C_TEXT_DIM
    surf.blit(font_m.render(teleop_label, True, teleop_color), (sx + 145, sy))

    lines = [
        "[TAB]      toggle player",
        "[↑↓]       move   [←→] turn",
        "[SPACE]    eat food",
        "[T]        teleop AI #0",
        "[E]        pat AI  (+pleasure)",
        "[F]        feed AI  (push food to mouth)",
        "[P] pause  [R] reset  [ESC] quit",
        f"step {world.step_count:>7}   deaths {world.death_count}",
    ]
    for i, line in enumerate(lines):
        surf.blit(font_s.render(line, True, C_TEXT_DIM), (sx, sy + 18 + i * 15))

    # Agent status lines (one per connected brain)
    active = [(i, a) for i, a in enumerate(world.agents) if a is not None]
    if not active:
        ay = sy + 18 + len(lines) * 15
        surf.blit(font_s.render("no agents connected", True, C_TEXT_DIM), (sx, ay))
    else:
        for row, (i, agent) in enumerate(active):
            color, _ = _AI_PALETTES[i % len(_AI_PALETTES)]
            label = slot_names.get(i) if slot_names else None
            label = label or f"#{i}"
            line = (f"{label}  L:{agent.meters.life:.2f}  "
                    f"V:{agent.meters.valence:+.2f}  "
                    f"S:{agent.meters.satiation:+.2f}")
            ay = sy + 18 + len(lines) * 15 + row * 13
            surf.blit(font_s.render(line, True, color), (sx, ay))


# ── Input helpers ─────────────────────────────────────────────────────────────

def _handle_events(world: World, paused: bool, teleop_enabled: bool) -> tuple:
    still_running = True
    reset_flash   = False
    pat           = False
    feed          = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            still_running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                still_running = False
            elif event.key == pygame.K_TAB:
                world.player_active = not world.player_active
            elif event.key == pygame.K_p:
                paused = not paused
                world.paused = paused
            elif event.key == pygame.K_r:
                world.reset()
                reset_flash = True
            elif event.key == pygame.K_e:
                pat = True
            elif event.key == pygame.K_f:
                feed = True
            elif event.key == pygame.K_t:
                teleop_enabled = not teleop_enabled
    return still_running, paused, reset_flash, pat, feed, teleop_enabled


def _manual_action(paused: bool) -> tuple:
    if paused:
        return (0.0, 0.0, 0.0)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        fwd = 1.0
    elif keys[pygame.K_DOWN]:
        fwd = -1.0
    else:
        fwd = 0.0
    if keys[pygame.K_RIGHT]:
        turn = 1.0
    elif keys[pygame.K_LEFT]:
        turn = -1.0
    else:
        turn = 0.0
    eat  = 1.0 if keys[pygame.K_SPACE] else 0.0
    return (fwd, turn, eat)


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _draw_world(
    screen: pygame.Surface,
    ray_surf: pygame.Surface,
    world: World,
    font_s: pygame.font.Font,
    slot_names: dict | None = None,
) -> None:
    screen.fill(C_BG)
    pygame.draw.rect(screen, C_BORDER, (0, 0, WORLD_W, WORLD_H), 2)

    for food in world.foods:
        if food.active:
            draw_food(screen, food)
    for danger in world.dangers:
        draw_danger(screen, danger)

    # Vision rays for slot 0 only (showing all would be cluttered)
    active_agents = [a for a in world.agents if a is not None]
    if active_agents:
        ai0     = active_agents[0]
        others0 = [e for e in ([world.player] if world.player_active else []) + active_agents
                   if e is not ai0]
        draw_vision(screen, ray_surf, ai0, world, others0)

    # Draw all AI bodies with slot-specific colours
    n_active = len(active_agents)
    for i, agent in enumerate(world.agents):
        if agent is None:
            continue
        color, ring = _AI_PALETTES[i % len(_AI_PALETTES)]
        draw_entity(screen, agent, color, ring)
        if n_active > 1:
            name = slot_names.get(i, f"#{i}") if slot_names else f"#{i}"
            lbl = font_s.render(name, True, ring)
            screen.blit(lbl, (int(agent.x) - lbl.get_width() // 2,
                               int(agent.y) - int(agent.radius) - 13))

    if world.player_active:
        draw_entity(screen, world.player, C_PLAYER, C_PLAYER_RING)


def _draw_death_flash(screen: pygame.Surface, death_flash: int) -> int:
    if death_flash <= 0:
        return 0
    alpha      = min(160, int(death_flash / 45 * 160))
    flash_surf = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
    flash_surf.fill((*C_DEATH_FLASH, alpha))
    screen.blit(flash_surf, (0, 0))
    font_big = pygame.font.SysFont("monospace", 48, bold=True)
    txt = font_big.render("DEAD — RESET", True, (255, 200, 200))
    screen.blit(txt, (WORLD_W // 2 - txt.get_width() // 2,
                       WORLD_H // 2 - txt.get_height() // 2))
    return death_flash - 1


def _draw_pause_overlay(screen: pygame.Surface) -> None:
    overlay = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    screen.blit(overlay, (0, 0))
    font_big = pygame.font.SysFont("monospace", 52, bold=True)
    txt = font_big.render("PAUSED", True, (200, 200, 230))
    screen.blit(txt, (WORLD_W // 2 - txt.get_width() // 2,
                       WORLD_H // 2 - txt.get_height() // 2))


# ── Main loop ─────────────────────────────────────────────────────────────────

def _load_ruleset(path: str) -> dict:
    import json as _json
    with open(path) as f:
        raw = _json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def main() -> None:
    import sys as _sys
    use_connector = "--connect" in _sys.argv
    no_reset      = "--no-reset" in _sys.argv
    rules_path    = None
    for i, arg in enumerate(_sys.argv[:-1]):
        if arg == "--rules":
            rules_path = _sys.argv[i + 1]
            break
    world_cfg = _load_ruleset(rules_path) if rules_path else {}

    conn = None
    if use_connector:
        from connector import MultiGameConnector
        conn = MultiGameConnector()
        conn.start()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("RO Framework — Embodied Environment")
    clock    = pygame.time.Clock()
    font_s   = pygame.font.SysFont("monospace", 11)
    font_m   = pygame.font.SysFont("monospace", 13, bold=True)
    world    = World(seed=42, no_reset_on_death=no_reset, cfg=world_cfg)
    ray_surf = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)

    paused               = False
    death_flash          = 0
    _prev_player_deaths  = 0
    teleop_enabled       = False

    running = True
    while running:
        running, paused, reset, pat, feed, teleop_enabled = _handle_events(
            world, paused, teleop_enabled,
        )
        if reset:
            death_flash = 0

        manual_act = _manual_action(paused)
        player_act = manual_act if world.player_active else (0.0, 0.0, 0.0)

        # Poll for newly connected brains — spawn a body per new slot
        if conn is not None:
            new_slots = conn.poll_registrations(world.step_count)
            for _ in new_slots:
                world.add_agent()

        # Gather actions for all registered agents
        if conn is not None:
            ai_actions = conn.recv_actions(len(world.agents), world.step_count)
            # Despawn bodies for brains that stopped responding
            for slot_id in conn.disconnected_slots(world.step_count):
                world.remove_agent(slot_id)
                conn.remove_slot(slot_id)
        else:
            ai_actions = [(0.0, 0.0, 0.0)] * len(world.agents)

        if teleop_enabled and ai_actions:
            ai_actions[0] = manual_act
        teleop_slots = {0} if teleop_enabled and ai_actions else set()

        n_agents_before = sum(1 for a in world.agents if a is not None)
        world.step(
            ai_actions=ai_actions,
            player_action=player_act,
            pat=pat,
            feed=feed,
            teleop_slots=teleop_slots,
        )

        # Detect player death: world.reset() clears all agents
        active_after = sum(1 for a in world.agents if a is not None)
        player_died = (n_agents_before > 0 and active_after == 0) or \
                      (n_agents_before == 0 and world.death_count > _prev_player_deaths)
        _prev_player_deaths = world.death_count

        # If the world reset (player died), re-spawn one body per registered brain
        if conn is not None and active_after < conn.n_slots:
            for slot_id in conn._slots:
                if slot_id >= len(world.agents) or world.agents[slot_id] is None:
                    world.add_agent()

        # Send observations to all connected brains
        if conn is not None:
            conn.send_obs_all(world)

        if player_died:
            death_flash = 45

        names = {sid: conn.slot_name(sid) for sid in conn._slots} if conn else None
        _draw_world(screen, ray_surf, world, font_s, slot_names=names)
        death_flash = _draw_death_flash(screen, death_flash)
        if paused:
            _draw_pause_overlay(screen)
        draw_hud(screen, world, font_s, font_m, slot_names=names,
                 teleop_enabled=teleop_enabled)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()
