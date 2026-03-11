"""
2D embodied environment for RO Framework AI experiments.

VISION CONE
-----------
The AI (and player) entities have a forward-facing cone of vision implemented
as N_RAYS independent ray-casts spread evenly across a 120° arc.  Each ray
reports: what it first hits and how far away that is.

TACTILE RECEPTORS
-----------------
Each entity has N_TOUCH_BODY = 16 receptors evenly spaced around its circular
body (receptor 0 = forward / heading direction, proceeding counter-clockwise in
math convention) plus N_TOUCH_PRONGS = 2 receptors — one at each prong tip.

Each receptor reports the *maximum* signal among stimuli within its angular
sector.  Signals are raw floats — the AI is expected to learn what each value
means, not read explicit labels:

  wall   → 0.7   (border contact)
  food   → 0.3   (edible item)
  entity → 0.5   (the other agent)
  danger → 1.0   (harmful, over-stimulation)

Non-cumulative: a corner touching two walls gives 0.7 on each relevant
receptor, never 1.4.  The strongest stimulus per sector wins.

Observation vector  (263 values, all in [0, 1]):
  [0 : 242]   Vision — 121 rays × [type_norm, proximity]
                type:      0=none · 0.25=wall · 0.5=food · 0.75=danger · 1=other
                proximity: 1 − (hit_dist / VISION_RANGE) when hit, else 0
  [242 : 260]  Tactile — 16 body receptors + 2 prong receptors
  [260 : 263]  Internal meters — [life, satiation_norm, valence_norm]
                satiation_norm = (satiation + 1) / 2  → 0=starving, 1=full
                valence_norm   = (valence   + 1) / 2  → 0=max pain, 1=max pleasure

Action vector  (3 values):
  [0] forward  ∈ [-1, 1]   +1 = full forward,  -1 = full reverse
  [1] turn     ∈ [-1, 1]   +1 = right,          -1 = left
  [2] eat      ∈  {0, 1}   1 = consume overlapping food (otherwise pushes it)

METER DYNAMICS
--------------
Danger contact
  • Valence immediately drops to min(current, -0.5) on first contact step.
  • After DANGER_LIFE_DELAY_STEPS consecutive contact steps (~3 s at 60 fps):
      life drains at LIFE_DRAIN_RATE each step, and
      valence = −(0.5 + 0.5 × (1 − life))   [pain rises as life falls]

Hunger
  • Satiation drains at SATIATION_DRAIN_RATE per step.
  • Once satiation ≤ −1.0, valence drifts negative at HUNGER_VALENCE_DRAIN rate.
  • Once valence ≤ −0.5 from starvation, life also drains.

Recovery
  • Eating food restores life, satiation, spikes valence.
  • Positive / negative valence both decay toward 0 when not under stress.
  • Leaving danger / eating food breaks the drain loop.

Death
  • When life reaches 0, the world auto-resets.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── World ─────────────────────────────────────────────────────────────────────
WORLD_W: int = 1200
WORLD_H: int = 900

# ── Entities ──────────────────────────────────────────────────────────────────
ENTITY_RADIUS:     float = 18.0
ENTITY_SPEED:      float = 3.0
ENTITY_TURN_SPEED: float = 0.06      # rad / step  (~3.4° → full turn ≈ 1.7 s)

# ── Items ─────────────────────────────────────────────────────────────────────
FOOD_RADIUS:        float = 12.0
DANGER_RADIUS:      float = 15.0
FOOD_COUNT:         int   = 20
DANGER_COUNT:       int   = 10
FOOD_RESPAWN_STEPS: int   = 300      # ~5 s at 60 fps

# ── Vision ────────────────────────────────────────────────────────────────────
VISION_RANGE:      float = 450.0
VISION_HALF_ANGLE: float = np.pi / 3  # 60° each side → 120° total cone
N_RAYS:            int   = 121        # ~1° between rays in the 120° cone

HIT_NONE:   int = 0
HIT_WALL:   int = 1
HIT_FOOD:   int = 2
HIT_DANGER: int = 3
HIT_OTHER:  int = 4   # the other entity
HIT_TYPES:  int = 5

# ── Tactile receptors ─────────────────────────────────────────────────────────
N_TOUCH_BODY:      int   = 16     # receptors evenly distributed around body
N_TOUCH_PRONGS:    int   = 2      # one at each prong tip
N_TOUCH_RECEPTORS: int   = N_TOUCH_BODY + N_TOUCH_PRONGS  # 18 total

# Signal strengths — raw floats, no explicit labels in the AI observation
TOUCH_WALL:    float = 0.7
TOUCH_FOOD:    float = 0.3
TOUCH_ENTITY:  float = 0.5
TOUCH_DANGER:  float = 1.0
TOUCH_MARGIN:  float = 4.0   # extra px beyond radius sum for contact detection

# ── Interface sizes ───────────────────────────────────────────────────────────
OBS_SIZE:    int = N_RAYS * 2 + N_TOUCH_RECEPTORS + 3
ACTION_SIZE: int = 3

# ── Meter dynamics ────────────────────────────────────────────────────────────
SATIATION_DRAIN_RATE: float = 0.0001
VALENCE_DECAY_NEG:    float = 0.001
VALENCE_DECAY_POS:    float = 0.001
HUNGER_VALENCE_DRAIN: float = 0.002   # net change when starving: −0.001/step

DANGER_LIFE_DELAY_STEPS: int   = 180
LIFE_DRAIN_RATE:         float = 0.001

# Instant rewards
FOOD_PLEASURE:       float = 0.6
FOOD_SATIATION_GAIN: float = 0.35
FOOD_LIFE_GAIN:      float = 0.15

OTHER_ENTITY_PLEASURE:  float = 0.005   # valence per step while touching
OTHER_ENTITY_TOUCH_CAP: float = 0.12    # touch alone cannot push valence above this

# ── Prongs / eat zone ─────────────────────────────────────────────────────────
PRONG_ANGLE:     float = 0.60   # rad from heading (±34°)
PRONG_LENGTH:    float = 18.0   # px beyond entity radius
PRONG_BASE_W:    float = 0.35   # half-angle of prong base on entity surface (rad)
PRONG_CAPSULE_R: float = 9.0    # collision capsule radius along each prong
EAT_ZONE_ANGLE:  float = 0.60
EAT_ZONE_REACH:  float = ENTITY_RADIUS + FOOD_RADIUS + PRONG_LENGTH + 2.0

# ── Social actions ────────────────────────────────────────────────────────────
PAT_PLEASURE:    float = 0.1
PAT_TOUCH_DIST:  float = ENTITY_RADIUS * 2 + 24.0
FEED_REACH_BONUS: float = 20.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Food:
    x: float
    y: float
    active: bool = True
    respawn_timer: int = 0


@dataclass
class Danger:
    x: float
    y: float


@dataclass
class Meters:
    life:      float = 1.0
    valence:   float = 0.0   # ∈ [-1, 1]
    satiation: float = 0.0   # ∈ [-1, 1]

    def as_array(self) -> np.ndarray:
        return np.array([
            self.life,
            (self.satiation + 1.0) * 0.5,
            (self.valence   + 1.0) * 0.5,
        ], dtype=np.float32)

    def reset(self) -> None:
        self.life      = 1.0
        self.valence   = 0.0
        self.satiation = 0.0


class TactileState:
    """
    Float array of directional tactile receptor signals, all ∈ [0, 1].

    body[i] covers an angular sector centered on heading + i*(2π/N_TOUCH_BODY).
    prongs[0] = left prong,  prongs[1] = right prong.

    Signal strengths intentionally carry no type label — the AI learns the
    mapping from strength to object identity through experience.
    """

    __slots__ = ("body", "prongs")

    def __init__(self) -> None:
        self.body   = np.zeros(N_TOUCH_BODY,   dtype=np.float32)
        self.prongs = np.zeros(N_TOUCH_PRONGS, dtype=np.float32)

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.body, self.prongs])   # (18,)

    def reset(self) -> None:
        self.body[:]   = 0.0
        self.prongs[:] = 0.0

    # ── Convenience flags for internal game logic ─────────────────────────────
    def has_danger(self) -> bool:
        """Any receptor at danger-level signal (≥ 0.9)?"""
        return bool(np.any(self.body >= 0.9) or np.any(self.prongs >= 0.9))

    def has_other(self) -> bool:
        """Any body receptor in the entity-contact range [0.45, 0.55]?"""
        return bool(np.any((self.body >= 0.45) & (self.body <= 0.55)))


# ── Entity ────────────────────────────────────────────────────────────────────

class Entity:
    def __init__(self, x: float, y: float, heading: float = 0.0) -> None:
        self.x       = float(x)
        self.y       = float(y)
        self.heading = float(heading)
        self.radius  = ENTITY_RADIUS
        self.speed   = ENTITY_SPEED
        self.turn_speed = ENTITY_TURN_SPEED
        self.meters  = Meters()
        self.tactile = TactileState()
        self.alive   = True
        self.danger_contact_steps: int = 0

    def apply_action(self, forward: float, turn: float) -> None:
        self.heading = (self.heading + turn * self.turn_speed) % (2.0 * np.pi)
        dx = np.cos(self.heading) * forward * self.speed
        dy = np.sin(self.heading) * forward * self.speed
        self.x = float(np.clip(self.x + dx, self.radius, WORLD_W - self.radius))
        self.y = float(np.clip(self.y + dy, self.radius, WORLD_H - self.radius))


# ── World ─────────────────────────────────────────────────────────────────────

_AI_START     = (WORLD_W * 0.25, WORLD_H * 0.5, 0.0)
_PLAYER_START = (WORLD_W * 0.75, WORLD_H * 0.5, np.pi)


class World:
    """
    Main interface:

        obs = world.get_observation(world.ai)     # shape (263,), all in [0, 1]
        world.step(ai_action=(fwd, turn, eat),
                   player_action=(fwd, turn, eat),
                   pat=False, feed=False)

    Actions: (forward ∈ [-1,1], turn ∈ [-1,1], eat ∈ {0,1}).

    When either entity's life reaches 0 the world auto-resets.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng  = np.random.default_rng(seed)
        self.paused: bool = False

        self.ai     = Entity(*_AI_START)
        self.player = Entity(*_PLAYER_START)
        self.player_active: bool = True

        self.foods   = self._spawn_foods(FOOD_COUNT)
        self.dangers = self._spawn_dangers(DANGER_COUNT)
        self.step_count: int  = 0
        self.death_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        ai_action:     Tuple[float, float, float] = (0.0, 0.0, 0.0),
        player_action: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        pat:  bool = False,
        feed: bool = False,
    ) -> None:
        if self.paused:
            return

        ai_fwd,  ai_turn,  ai_eat  = ai_action
        pl_fwd,  pl_turn,  pl_eat  = player_action

        self.ai.apply_action(
            float(np.clip(ai_fwd,  -1, 1)),
            float(np.clip(ai_turn, -1, 1)),
        )
        if self.player_active:
            self.player.apply_action(
                float(np.clip(pl_fwd,  -1, 1)),
                float(np.clip(pl_turn, -1, 1)),
            )

        self._resolve_entities()

        if pat:
            self._try_pat_ai()
        if feed:
            self._try_feed_ai()

        ai_ate = self._resolve_food(self.ai,    bool(ai_eat > 0.5))
        pl_ate = (self._resolve_food(self.player, bool(pl_eat > 0.5))
                  if self.player_active else False)

        self._update_entity(self.ai,    other=self.player, ate_food=ai_ate)
        if self.player_active:
            self._update_entity(self.player, other=self.ai,    ate_food=pl_ate)

        self._tick_food_respawns()
        self.step_count += 1

        if not self.ai.alive or not self.player.alive:
            self.death_count += 1
            self.reset(keep_counts=True)

    def reset(self, keep_counts: bool = False) -> None:
        counts = (self.step_count, self.death_count) if keep_counts else (0, 0)
        self.ai     = Entity(*_AI_START)
        self.player = Entity(*_PLAYER_START)
        self.foods   = self._spawn_foods(FOOD_COUNT)
        self.dangers = self._spawn_dangers(DANGER_COUNT)
        self.step_count  = counts[0] if keep_counts else 0
        self.death_count = counts[1] if keep_counts else 0

    def get_observation(self, entity: Entity) -> np.ndarray:
        """263-dim observation vector, all values ∈ [0, 1]."""
        other  = self.player if entity is self.ai else self.ai
        vision  = self._cast_rays(entity, other).flatten()   # (242,)
        tactile = entity.tactile.as_array()                  # (18,)
        meters  = entity.meters.as_array()                   # (3,)
        return np.concatenate([vision, tactile, meters])     # (263,)

    def get_ai_observation(self) -> np.ndarray:
        return self.get_observation(self.ai)

    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    # ── Ray casting ───────────────────────────────────────────────────────────

    def _cast_rays(self, entity: Entity, other: Entity) -> np.ndarray:
        """Returns (N_RAYS, 2) float32: [type_norm, proximity] per ray."""
        angles = np.linspace(
            entity.heading - VISION_HALF_ANGLE,
            entity.heading + VISION_HALF_ANGLE,
            N_RAYS,
        )
        result = np.zeros((N_RAYS, 2), dtype=np.float32)
        for i, angle in enumerate(angles):
            hit_type, dist = self._cast_ray(entity, angle, other)
            result[i, 0] = hit_type / (HIT_TYPES - 1)
            result[i, 1] = (1.0 - dist / VISION_RANGE) if hit_type != HIT_NONE else 0.0
        return result

    @staticmethod
    def _ray_nearest(
        ox: float, oy: float, dx: float, dy: float,
        items: list, radius: float,
        skip_dist: float, best_dist: float,
        active_only: bool = False,
    ) -> Optional[float]:
        nearest = None
        for item in items:
            if active_only and not item.active:
                continue
            d = _ray_circle_dist(ox, oy, dx, dy, item.x, item.y, radius)
            if d is not None and skip_dist < d < (nearest or best_dist):
                nearest = d
        return nearest

    def _cast_ray(
        self, entity: Entity, angle: float, other: Entity
    ) -> Tuple[int, float]:
        dx, dy    = float(np.cos(angle)), float(np.sin(angle))
        ox, oy    = entity.x, entity.y
        skip_dist = entity.radius
        best_dist = VISION_RANGE
        best_type = HIT_NONE

        d = _ray_wall_dist(ox, oy, dx, dy, WORLD_W, WORLD_H)
        if skip_dist < d < best_dist:
            best_dist, best_type = d, HIT_WALL

        d = self._ray_nearest(ox, oy, dx, dy, self.foods,   FOOD_RADIUS,
                              skip_dist, best_dist, active_only=True)
        if d is not None:
            best_dist, best_type = d, HIT_FOOD

        d = self._ray_nearest(ox, oy, dx, dy, self.dangers, DANGER_RADIUS,
                              skip_dist, best_dist)
        if d is not None:
            best_dist, best_type = d, HIT_DANGER

        if not (other is self.player and not self.player_active):
            d = _ray_circle_dist(ox, oy, dx, dy, other.x, other.y, other.radius)
            if d is not None and skip_dist < d < best_dist:
                best_dist, best_type = d, HIT_OTHER

        return best_type, best_dist

    # ── Food physics ──────────────────────────────────────────────────────────

    def _prong_tips(self, entity: Entity) -> List[Tuple[float, float]]:
        """Return tip (x, y) of both prongs (used by game.py for rendering)."""
        r, h = entity.radius, entity.heading
        return [
            (
                entity.x + (r + PRONG_LENGTH) * float(np.cos(h + side * PRONG_ANGLE)),
                entity.y + (r + PRONG_LENGTH) * float(np.sin(h + side * PRONG_ANGLE)),
            )
            for side in (-1.0, 1.0)
        ]

    @staticmethod
    def _consume_food(food: Food) -> None:
        food.active = False
        food.respawn_timer = FOOD_RESPAWN_STEPS

    @staticmethod
    def _push_food(food: Food, ox: float, oy: float, min_dist: float) -> None:
        dx = food.x - ox
        dy = food.y - oy
        dist = float(np.hypot(dx, dy))
        nx = dx / dist if dist > 0.1 else 1.0
        ny = dy / dist if dist > 0.1 else 0.0
        push = min_dist - dist + 0.5
        food.x = float(np.clip(food.x + nx * push, FOOD_RADIUS + 2, WORLD_W - FOOD_RADIUS - 2))
        food.y = float(np.clip(food.y + ny * push, FOOD_RADIUS + 2, WORLD_H - FOOD_RADIUS - 2))

    def _apply_prong_push(self, food: Food, entity: Entity) -> None:
        min_d = PRONG_CAPSULE_R + FOOD_RADIUS
        r, h  = entity.radius, entity.heading
        for side in (-1.0, 1.0):
            angle = h + side * PRONG_ANGLE
            cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            p1x = entity.x + r * cos_a;        p1y = entity.y + r * sin_a
            p2x = entity.x + (r + PRONG_LENGTH) * cos_a
            p2y = entity.y + (r + PRONG_LENGTH) * sin_a
            sx, sy     = p2x - p1x, p2y - p1y
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 1e-6:
                continue
            t = float(np.clip(((food.x - p1x) * sx + (food.y - p1y) * sy) / seg_len_sq, 0.0, 1.0))
            dist = float(np.hypot(food.x - (p1x + t * sx), food.y - (p1y + t * sy)))
            if 0.1 < dist < min_d:
                self._push_food(food, p1x + t * sx, p1y + t * sy, min_d)
                break

    def _resolve_food(self, entity: Entity, eat_triggered: bool) -> bool:
        ate = False
        for food in self.foods:
            if not food.active:
                continue
            dist     = float(np.hypot(food.x - entity.x, food.y - entity.y))
            min_body = entity.radius + FOOD_RADIUS
            if dist < min_body:
                if eat_triggered and not ate and self._in_eat_zone(entity, food):
                    self._consume_food(food)
                    ate = True
                else:
                    self._push_food(food, entity.x, entity.y, min_body)
            elif eat_triggered and not ate and self._in_eat_zone(entity, food):
                self._consume_food(food)
                ate = True
            else:
                self._apply_prong_push(food, entity)
        return ate

    def _in_eat_zone(self, entity: Entity, food: Food) -> bool:
        dx   = food.x - entity.x
        dy   = food.y - entity.y
        dist = float(np.hypot(dx, dy))
        if dist > EAT_ZONE_REACH or dist < 0.1:
            return False
        food_angle = float(np.arctan2(dy, dx))
        diff = (food_angle - entity.heading + np.pi) % (2.0 * np.pi) - np.pi
        return abs(diff) <= EAT_ZONE_ANGLE

    def _resolve_entities(self) -> None:
        if not self.player_active:
            return
        dx   = self.player.x - self.ai.x
        dy   = self.player.y - self.ai.y
        dist = float(np.hypot(dx, dy))
        min_dist = self.ai.radius + self.player.radius
        if 0 < dist < min_dist:
            overlap = (min_dist - dist + 0.5) * 0.5
            nx, ny  = dx / dist, dy / dist
            r = self.ai.radius
            self.ai.x = float(np.clip(self.ai.x - nx * overlap, r, WORLD_W - r))
            self.ai.y = float(np.clip(self.ai.y - ny * overlap, r, WORLD_H - r))
            r = self.player.radius
            self.player.x = float(np.clip(self.player.x + nx * overlap, r, WORLD_W - r))
            self.player.y = float(np.clip(self.player.y + ny * overlap, r, WORLD_H - r))

    def _try_pat_ai(self) -> None:
        dist = float(np.hypot(self.ai.x - self.player.x, self.ai.y - self.player.y))
        if dist < PAT_TOUCH_DIST:
            self.ai.meters.valence = min(1.0, self.ai.meters.valence + PAT_PLEASURE)

    def _try_feed_ai(self) -> None:
        for food in self.foods:
            if not food.active:
                continue
            in_player = (float(np.hypot(food.x - self.player.x, food.y - self.player.y))
                         < self.player.radius + FOOD_RADIUS + FEED_REACH_BONUS)
            if in_player and self._in_eat_zone(self.ai, food):
                food.active = False
                food.respawn_timer = FOOD_RESPAWN_STEPS
                m = self.ai.meters
                m.valence   = min(1.0, m.valence   + FOOD_PLEASURE)
                m.satiation = min(1.0, m.satiation + FOOD_SATIATION_GAIN)
                m.life      = min(1.0, m.life      + FOOD_LIFE_GAIN)
                return

    # ── Tactile sensing ───────────────────────────────────────────────────────

    def _update_tactile(self, entity: Entity, other: Entity) -> None:
        """Recompute all receptor signals for entity."""
        body   = np.zeros(N_TOUCH_BODY,   dtype=np.float32)
        prongs = np.zeros(N_TOUCH_PRONGS, dtype=np.float32)
        self._sense_walls(entity, body)
        self._sense_foods(entity, body)
        self._sense_dangers(entity, body)
        self._sense_other_entity(entity, other, body)
        self._sense_prongs(entity, other, prongs)
        entity.tactile.body[:]   = body
        entity.tactile.prongs[:] = prongs

    def _sense_walls(self, entity: Entity, body: np.ndarray) -> None:
        r, m = entity.radius, TOUCH_MARGIN
        ex, ey = entity.x, entity.y
        if ex <= r + m:
            _write_receptor(body, entity.heading, np.pi, TOUCH_WALL)
        if ex >= WORLD_W - r - m:
            _write_receptor(body, entity.heading, 0.0, TOUCH_WALL)
        if ey <= r + m:
            _write_receptor(body, entity.heading, -np.pi * 0.5, TOUCH_WALL)
        if ey >= WORLD_H - r - m:
            _write_receptor(body, entity.heading, np.pi * 0.5, TOUCH_WALL)

    def _sense_foods(self, entity: Entity, body: np.ndarray) -> None:
        threshold = entity.radius + FOOD_RADIUS + TOUCH_MARGIN
        for food in self.foods:
            if not food.active:
                continue
            if float(np.hypot(food.x - entity.x, food.y - entity.y)) < threshold:
                angle = float(np.arctan2(food.y - entity.y, food.x - entity.x))
                _write_receptor(body, entity.heading, angle, TOUCH_FOOD)

    def _sense_dangers(self, entity: Entity, body: np.ndarray) -> None:
        threshold = entity.radius + DANGER_RADIUS + TOUCH_MARGIN
        for danger in self.dangers:
            if float(np.hypot(danger.x - entity.x, danger.y - entity.y)) < threshold:
                angle = float(np.arctan2(danger.y - entity.y, danger.x - entity.x))
                _write_receptor(body, entity.heading, angle, TOUCH_DANGER)

    def _sense_other_entity(
        self, entity: Entity, other: Entity, body: np.ndarray
    ) -> None:
        if other is self.player and not self.player_active:
            return
        threshold = entity.radius + other.radius + TOUCH_MARGIN
        if float(np.hypot(other.x - entity.x, other.y - entity.y)) < threshold:
            angle = float(np.arctan2(other.y - entity.y, other.x - entity.x))
            _write_receptor(body, entity.heading, angle, TOUCH_ENTITY)

    def _sense_prongs(self, entity: Entity, other: Entity, prongs: np.ndarray) -> None:
        for pi, side in enumerate((-1.0, 1.0)):
            prongs[pi] = self._prong_signal(entity, side, other)

    def _prong_signal(self, entity: Entity, side: float, other: Entity) -> float:
        """
        Max signal anywhere on the prong capsule (tip, sides, base).
        The capsule radius (PRONG_CAPSULE_R) already exceeds the visual
        triangle half-width, so the full triangular area is covered.
        Sense threshold = PRONG_CAPSULE_R + obj_radius + TOUCH_MARGIN so
        the sensor fires even after the push settles food at the boundary.
        The extra TOUCH_MARGIN also gives a small proximity halo beyond contact.
        """
        r, h = entity.radius, entity.heading
        angle = h + side * PRONG_ANGLE
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        p1x = entity.x + r * cos_a
        p1y = entity.y + r * sin_a
        p2x = entity.x + (r + PRONG_LENGTH) * cos_a
        p2y = entity.y + (r + PRONG_LENGTH) * sin_a
        max_sig = 0.0
        for food in self.foods:
            if not food.active:
                continue
            if _capsule_dist(food.x, food.y, p1x, p1y, p2x, p2y) < PRONG_CAPSULE_R + FOOD_RADIUS + TOUCH_MARGIN:
                max_sig = max(max_sig, TOUCH_FOOD)
        for danger in self.dangers:
            if _capsule_dist(danger.x, danger.y, p1x, p1y, p2x, p2y) < PRONG_CAPSULE_R + DANGER_RADIUS + TOUCH_MARGIN:
                max_sig = max(max_sig, TOUCH_DANGER)
        if not (other is self.player and not self.player_active):
            if _capsule_dist(other.x, other.y, p1x, p1y, p2x, p2y) < PRONG_CAPSULE_R + other.radius + TOUCH_MARGIN:
                max_sig = max(max_sig, TOUCH_ENTITY)
        return max_sig

    # ── Entity update ─────────────────────────────────────────────────────────

    def _step_valence_and_life(self, entity: Entity) -> None:
        m        = entity.meters
        in_danger = entity.tactile.has_danger()
        draining = (
            (in_danger and entity.danger_contact_steps >= DANGER_LIFE_DELAY_STEPS)
            or (m.satiation <= -1.0 and m.valence <= -0.5)
        )
        if draining:
            m.life    = max(0.0, m.life - LIFE_DRAIN_RATE)
            m.valence = -(0.5 + 0.5 * (1.0 - m.life))
        elif in_danger:
            m.valence = min(m.valence, -0.5)
        elif m.satiation <= -1.0:
            net = HUNGER_VALENCE_DRAIN - VALENCE_DECAY_NEG
            m.valence = float(np.clip(m.valence - net, -1.0, 1.0))
        elif m.valence > 0.0:
            m.valence = max(0.0, m.valence - VALENCE_DECAY_POS)
        elif m.valence < 0.0:
            m.valence = min(0.0, m.valence + VALENCE_DECAY_NEG)
        if m.life <= 0.0:
            entity.alive = False

    def _update_entity(
        self, entity: Entity, other: Entity, ate_food: bool
    ) -> None:
        m = entity.meters
        self._update_tactile(entity, other)

        entity.danger_contact_steps = (
            entity.danger_contact_steps + 1 if entity.tactile.has_danger() else 0
        )

        m.satiation = max(-1.0, m.satiation - SATIATION_DRAIN_RATE)

        if ate_food:
            m.valence   = min(1.0, m.valence   + FOOD_PLEASURE)
            m.satiation = min(1.0, m.satiation + FOOD_SATIATION_GAIN)
            m.life      = min(1.0, m.life      + FOOD_LIFE_GAIN)
        if entity.tactile.has_other() and m.valence < OTHER_ENTITY_TOUCH_CAP:
            m.valence = min(OTHER_ENTITY_TOUCH_CAP, m.valence + OTHER_ENTITY_PLEASURE)

        self._step_valence_and_life(entity)

    # ── Food respawn ──────────────────────────────────────────────────────────

    def _tick_food_respawns(self) -> None:
        for food in self.foods:
            if not food.active:
                food.respawn_timer -= 1
                if food.respawn_timer <= 0:
                    food.active = True
                    food.x = float(self._rng.uniform(
                        FOOD_RADIUS + 40, WORLD_W - FOOD_RADIUS - 40))
                    food.y = float(self._rng.uniform(
                        FOOD_RADIUS + 40, WORLD_H - FOOD_RADIUS - 40))

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_foods(self, count: int) -> List[Food]:
        return [
            Food(
                x=float(self._rng.uniform(FOOD_RADIUS + 40, WORLD_W - FOOD_RADIUS - 40)),
                y=float(self._rng.uniform(FOOD_RADIUS + 40, WORLD_H - FOOD_RADIUS - 40)),
            )
            for _ in range(count)
        ]

    def _spawn_dangers(self, count: int) -> List[Danger]:
        return [
            Danger(
                x=float(self._rng.uniform(DANGER_RADIUS + 40, WORLD_W - DANGER_RADIUS - 40)),
                y=float(self._rng.uniform(DANGER_RADIUS + 40, WORLD_H - DANGER_RADIUS - 40)),
            )
            for _ in range(count)
        ]


# ── Pure geometry helpers ─────────────────────────────────────────────────────

def _write_receptor(
    body: np.ndarray, heading: float, world_angle: float, strength: float
) -> None:
    """Write signal strength to the nearest body receptor for world_angle."""
    sector = 2.0 * np.pi / N_TOUCH_BODY
    rel    = (world_angle - heading) % (2.0 * np.pi)
    idx    = int(round(rel / sector)) % N_TOUCH_BODY
    if strength > body[idx]:
        body[idx] = strength


def _capsule_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Distance from point (px,py) to line segment (ax,ay)→(bx,by)."""
    sx, sy     = bx - ax, by - ay
    seg_len_sq = sx * sx + sy * sy
    if seg_len_sq < 1e-6:
        return float(np.hypot(px - ax, py - ay))
    t = float(np.clip(((px - ax) * sx + (py - ay) * sy) / seg_len_sq, 0.0, 1.0))
    return float(np.hypot(px - (ax + t * sx), py - (ay + t * sy)))


def _ray_wall_dist(
    ox: float, oy: float, dx: float, dy: float, w: int, h: int
) -> float:
    candidates: List[float] = []
    if dx > 0:   candidates.append((w - ox) / dx)
    elif dx < 0: candidates.append(-ox / dx)
    if dy > 0:   candidates.append((h - oy) / dy)
    elif dy < 0: candidates.append(-oy / dy)
    pos = [t for t in candidates if t > 1e-6]
    return min(pos) if pos else VISION_RANGE


def _ray_circle_dist(
    ox: float, oy: float, dx: float, dy: float,
    cx: float, cy: float, r: float,
) -> Optional[float]:
    ex = ox - cx;  ey = oy - cy
    b  = 2.0 * (ex * dx + ey * dy)
    c  = ex * ex + ey * ey - r * r
    d  = b * b - 4.0 * c
    if d < 0:
        return None
    sq = float(np.sqrt(d))
    t  = (-b - sq) * 0.5
    if t > 1e-6:
        return t
    t = (-b + sq) * 0.5
    return t if t > 1e-6 else None
