"""
2D Embodied AI Environment — interactive viewer.

Controls:
  Arrow keys      move player  (while player is active)
  Space           eat          (player consumes overlapping food)
  Tab             toggle player on / off
  P               pause / unpause
  R               reset world  (respawns everything)
  Escape / Q      quit

AI entity (blue) uses zero actions by default.
Replace `ai_action` in the main loop with your ESN output:

    obs     = world.get_ai_observation()      # (50,) numpy array, all in [0,1]
    esn_out = esn.step(obs)                   # (3,): [forward, turn, eat]
    ai_action = tuple(float(v) for v in esn_out)

Requires:  pygame  (pip install pygame)
"""

from __future__ import annotations

import os
import sys
import numpy as np

try:
    import pygame
except ImportError:
    raise SystemExit("pygame is required:  pip install pygame")

sys.path.insert(0, os.path.dirname(__file__))
from env import (
    World, Entity,
    WORLD_W, WORLD_H,
    VISION_RANGE, VISION_HALF_ANGLE, N_RAYS,
    HIT_NONE, HIT_WALL, HIT_FOOD, HIT_DANGER, HIT_OTHER,
    PRONG_ANGLE, PRONG_LENGTH, PRONG_BASE_W,
)

# ── Layout ────────────────────────────────────────────────────────────────────
FPS        = 60
HUD_HEIGHT = 130
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
C_SAT_NEG      = (215, 135,  40)   # hungry  (orange)
C_SAT_POS      = ( 55, 195,  75)   # satiated (green)
C_VAL_NEG      = (215,  55,  55)   # pain    (red)
C_VAL_POS      = (155,  75, 215)   # pleasure (purple)

C_DEATH_FLASH  = (180,  30,  30)
C_PAUSE_OVERLAY= (  0,   0,   0)

# Ray hit → render colour
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
    width: int = 120,
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
    value: float,          # ∈ [-1, 1]
    x: int, y: int,
    color_neg: tuple,
    color_pos: tuple,
    width: int = 120,
    height: int = 10,
) -> None:
    """Centred bar: left half fills for negative values, right half for positive."""
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
    # Centre tick
    pygame.draw.line(surf, (110, 115, 140), (x + mid, y), (x + mid, y + height - 1))
    pygame.draw.rect(surf, (75, 78, 100), (x, y, width, height), 1, border_radius=3)
    txt = font.render(f"{label}  {value:+.2f}", True, C_TEXT)
    surf.blit(txt, (x + width + 6, y))


def draw_prongs(
    surf: pygame.Surface,
    entity: Entity,
    color: tuple,
) -> None:
    """Draw two triangular prongs (fangs/hands) on the front of the entity."""
    cx, cy = entity.x, entity.y
    r = entity.radius
    h = entity.heading
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
    # Heading dot
    hx = int(cx + np.cos(entity.heading) * (r - 3))
    hy = int(cy + np.sin(entity.heading) * (r - 3))
    pygame.draw.circle(surf, (255, 255, 255), (hx, hy), 3)


def draw_food(surf: pygame.Surface, food) -> None:
    cx, cy = int(food.x), int(food.y)
    pygame.draw.circle(surf, C_FOOD,      (cx, cy), 8)
    pygame.draw.circle(surf, C_FOOD_RING, (cx, cy), 8, 1)


def draw_danger(surf: pygame.Surface, danger) -> None:
    cx, cy = int(danger.x), int(danger.y)
    r = 10
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(surf, C_DANGER_FILL, pts)
    pygame.draw.polygon(surf, C_DANGER,      pts, 2)


def draw_vision(
    surf: pygame.Surface,
    ray_surf: pygame.Surface,
    entity: Entity,
    world: World,
    other: Entity,
) -> None:
    """Draw vision rays. Brightness scales with proximity (close = brighter)."""
    ray_surf.fill((0, 0, 0, 0))
    angles = np.linspace(
        entity.heading - VISION_HALF_ANGLE,
        entity.heading + VISION_HALF_ANGLE,
        N_RAYS,
    )
    ox, oy = int(entity.x), int(entity.y)
    for angle in angles:
        hit_type, dist = world._cast_ray(entity, angle, other)
        proximity = (1.0 - dist / VISION_RANGE) if hit_type != HIT_NONE else 0.0
        base = RAY_COLORS[hit_type]

        # Ray brightness: dim for background, scale with proximity for hits
        if hit_type == HIT_NONE:
            alpha_line = 60
        else:
            alpha_line = int(80 + 160 * proximity)   # 80…240

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
    label: str,
    color: tuple,
    font_s: pygame.font.Font,
    font_m: pygame.font.Font,
    x: int,
    y: int,
) -> None:
    """Draw meters + touch state for one entity at (x, y)."""
    surf.blit(font_m.render(label, True, color), (x, y))

    m  = entity.meters
    t  = entity.touch
    my = y + 18

    draw_meter(      surf, "Life    ", m.life,      x, my,      C_LIFE,    width=110)
    draw_polar_meter(surf, "Satiation", m.satiation, x, my + 16, C_SAT_NEG, C_SAT_POS, width=110)
    draw_polar_meter(surf, "Valence  ", m.valence,   x, my + 32, C_VAL_NEG, C_VAL_POS, width=110)

    # Touch indicators
    tx = x
    ty = my + 52
    touches = [
        ("wall",   t.wall,   C_BORDER),
        ("food",   t.food,   C_FOOD),
        ("danger", t.danger, C_DANGER),
        ("other",  t.other,  color),
    ]
    for name, active, tc in touches:
        c   = tc if active else C_TEXT_DIM
        dot = "●" if active else "○"
        surf.blit(font_s.render(f"{dot} {name}", True, c), (tx, ty))
        tx += 62


def draw_hud(
    surf: pygame.Surface,
    world: World,
    font_s: pygame.font.Font,
    font_m: pygame.font.Font,
    paused: bool,
) -> None:
    top = WORLD_H
    pygame.draw.rect(surf, C_HUD_BG, (0, top, SCREEN_W, HUD_HEIGHT))
    pygame.draw.line(surf, C_BORDER, (0, top), (SCREEN_W, top), 1)

    # ── AI (left third) ───────────────────────────────────────────────────────
    draw_entity_hud(surf, world.ai,     "AI ENTITY",     C_AI,
                    font_s, font_m, x=14, y=top + 8)

    # Vertical divider
    pygame.draw.line(surf, C_HUD_DIV,
                     (SCREEN_W // 3, top + 6), (SCREEN_W // 3, top + HUD_HEIGHT - 6), 1)

    # ── Player (middle third) ─────────────────────────────────────────────────
    p_color = C_PLAYER if world.player_active else C_PLAYER_OFF
    draw_entity_hud(surf, world.player, "PLAYER",         p_color,
                    font_s, font_m, x=SCREEN_W // 3 + 14, y=top + 8)

    # Vertical divider
    pygame.draw.line(surf, C_HUD_DIV,
                     (SCREEN_W * 2 // 3, top + 6), (SCREEN_W * 2 // 3, top + HUD_HEIGHT - 6), 1)

    # ── Controls + status (right third) ──────────────────────────────────────
    sx = SCREEN_W * 2 // 3 + 14
    sy = top + 8

    player_on = world.player_active
    surf.blit(font_m.render(
        "PLAYER: " + ("ON " if player_on else "OFF"),
        True, C_PLAYER if player_on else C_TEXT_DIM),
        (sx, sy))

    lines = [
        "[TAB]      toggle player",
        "[↑↓]       move   [←→] turn",
        "[SPACE]    eat food  (face it first)",
        "[E]        pat AI  (+pleasure)",
        "[F]        feed AI  (push food to AI mouth)",
        "[P] pause  [R] reset  [ESC] quit",
        f"step {world.step_count:>7}   deaths {world.death_count}",
    ]
    for i, line in enumerate(lines):
        c = (180, 80, 80) if paused and i == 3 else C_TEXT_DIM
        surf.blit(font_s.render(line, True, c), (sx, sy + 18 + i * 15))


# ── Input helpers ─────────────────────────────────────────────────────────────

def _handle_events(world: World, paused: bool) -> tuple:
    """
    Process pygame event queue.
    Returns (still_running, paused, reset_flash, pat, feed).
    pat / feed are True on the key-press frame (one-shot).
    """
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
    return still_running, paused, reset_flash, pat, feed


def _player_action(world: World, paused: bool) -> tuple:
    """Read held keys and return (forward, turn, eat) for the player."""
    if not world.player_active or paused:
        return (0.0, 0.0, 0.0)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        fwd = 1.0
    elif keys[pygame.K_DOWN]:
        fwd = -1.0
    else:
        fwd = 0.0
    if keys[pygame.K_LEFT]:
        turn = -1.0
    elif keys[pygame.K_RIGHT]:
        turn = 1.0
    else:
        turn = 0.0
    eat   =  1.0 if keys[pygame.K_SPACE] else 0.0
    return (fwd, turn, eat)


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _draw_world(
    screen: pygame.Surface,
    ray_surf: pygame.Surface,
    world: World,
) -> None:
    screen.fill(C_BG)
    pygame.draw.rect(screen, C_BORDER, (0, 0, WORLD_W, WORLD_H), 2)

    for food in world.foods:
        if food.active:
            draw_food(screen, food)
    for danger in world.dangers:
        draw_danger(screen, danger)

    draw_vision(screen, ray_surf, world.ai, world, world.player)

    draw_entity(screen, world.ai, C_AI, C_AI_RING)
    if world.player_active:
        draw_entity(screen, world.player, C_PLAYER, C_PLAYER_RING)


def _draw_death_flash(screen: pygame.Surface, death_flash: int) -> int:
    if death_flash <= 0:
        return 0
    alpha     = min(160, int(death_flash / 45 * 160))
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

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("RO Framework — Embodied Environment")
    clock  = pygame.time.Clock()

    font_s   = pygame.font.SysFont("monospace", 11)
    font_m   = pygame.font.SysFont("monospace", 13, bold=True)
    world    = World(seed=42)
    ray_surf = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)

    paused      = False
    death_flash = 0

    running = True
    while running:
        running, paused, reset, pat, feed = _handle_events(world, paused)
        if reset:
            death_flash = 0

        player_act = _player_action(world, paused)

        # ── AI action (placeholder — replace with ESN output) ─────────────────
        # obs        = world.get_ai_observation()    # (50,) array
        # esn_out    = esn.step(obs)                 # (3,) array
        # ai_action  = tuple(float(v) for v in esn_out)
        ai_action = (0.0, 0.0, 0.0)

        prev_deaths = world.death_count
        world.step(ai_action=ai_action, player_action=player_act,
                   pat=pat, feed=feed)
        if world.death_count > prev_deaths:
            death_flash = 45   # ~0.75 s at 60 fps

        _draw_world(screen, ray_surf, world)
        death_flash = _draw_death_flash(screen, death_flash)
        if paused:
            _draw_pause_overlay(screen)
        draw_hud(screen, world, font_s, font_m, paused)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
