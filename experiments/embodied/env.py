"""
2D embodied environment for RO Framework AI experiments.

VISION CONE
-----------
The AI (and player) entities have a forward-facing cone of vision implemented
as N_RAYS independent ray-casts spread evenly across a 120° arc. Each ray
travels up to VISION_RANGE pixels and reports:
  - what it first hits  (wall / food / danger / other entity)
  - how far away that is

This gives the entity a "strip" of directional distance readings — like a
simplified LiDAR. With N_RAYS=21 there is ~5.7° between rays, which is enough
to resolve individual food and danger items at moderate range.

Observation vector  (50 values, all in [0, 1]):
  [0 : 42]  Vision — 21 rays × [type_norm, proximity]
              type:      0=none · 0.25=wall · 0.5=food · 0.75=danger · 1=other-entity
              proximity: 1 − (hit_distance / VISION_RANGE) when something is hit, else 0
                         → strong signal (≈1) when object is close,
                           weak signal (≈0) when far or nothing detected
  [42 : 46] Touch sensors — [wall, food, danger, other_entity]
  [46 : 49] Internal meters — [life, satiation_norm, valence_norm]
              satiation_norm = (satiation + 1) / 2  → 0=starving, 0.5=neutral, 1=full
              valence_norm   = (valence   + 1) / 2  → 0=max pain, 0.5=neutral, 1=max pleasure

Action vector  (3 values):
  [0] forward  ∈ [-1, 1]   +1 = full forward,  -1 = full reverse
  [1] turn     ∈ [-1, 1]   +1 = right,          -1 = left
  [2] eat      ∈  {0, 1}   1 = consume overlapping food (otherwise pushes it)

METER DYNAMICS
--------------
Danger contact
  • Pain immediately jumps to max(current, 0.5) on first contact step.
  • After DANGER_LIFE_DELAY_STEPS consecutive contact steps (~3 s at 60 fps):
      life drains at LIFE_DRAIN_RATE each step, and
      pain = 0.5 + 0.5 × (1 − life)   [pain rises inversely with remaining life]

Hunger
  • Hunger increases at HUNGER_RATE per step.
  • Once hunger reaches 1.0, pain rises at HUNGER_PAIN_INCREASE_RATE − PAIN_DECAY.
  • Once pain ≥ 0.5 (from any cause), life drains as above, and
      pain = 0.5 + 0.5 × (1 − life)

Recovery
  • Eating food restores life, drops hunger, spikes pleasure.
  • Pain decays at PAIN_DECAY per step when not in a drain phase.
  • Leaving danger / eating food can break the drain loop.

Death
  • When life reaches 0, the world auto-resets: both entities return to their
    start positions with fresh meters; all items re-spawn at random locations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── World ─────────────────────────────────────────────────────────────────────
WORLD_W: int = 800
WORLD_H: int = 600

# ── Entities ──────────────────────────────────────────────────────────────────
ENTITY_RADIUS:     float = 12.0
ENTITY_SPEED:      float = 2.5
ENTITY_TURN_SPEED: float = 0.06      # rad / step  (~3.4° → full turn ≈ 1.7 s)

# ── Items ─────────────────────────────────────────────────────────────────────
FOOD_RADIUS:        float = 8.0
DANGER_RADIUS:      float = 10.0
FOOD_COUNT:         int   = 15
DANGER_COUNT:       int   = 8
FOOD_RESPAWN_STEPS: int   = 300      # ~5 s at 60 fps

# ── Vision ────────────────────────────────────────────────────────────────────
VISION_RANGE:      float = 300.0
VISION_HALF_ANGLE: float = np.pi / 3  # 60° each side → 120° total cone
N_RAYS:            int   = 121         # ~5.7° between rays

HIT_NONE:   int = 0
HIT_WALL:   int = 1
HIT_FOOD:   int = 2
HIT_DANGER: int = 3
HIT_OTHER:  int = 4   # the other entity (player seen by AI, AI seen by player)
HIT_TYPES:  int = 5

# ── Interface sizes ───────────────────────────────────────────────────────────
OBS_SIZE:    int = N_RAYS * 2 + 4 + 3   # 49  (42 vision + 4 touch + 3 meters)
ACTION_SIZE: int = 3                     # forward, turn, eat

# ── Meter dynamics ────────────────────────────────────────────────────────────
# Gradual drift rates (intentionally slow — instant events are larger)
SATIATION_DRAIN_RATE: float = 0.0001   # satiation lost per step  (full→starving ≈ 3 min)
VALENCE_DECAY_NEG:    float = 0.001    # negative valence recovers toward 0
VALENCE_DECAY_POS:    float = 0.001    # positive valence decays toward 0
HUNGER_VALENCE_DRAIN: float = 0.002    # extra valence drop per step when starving
                                       # net valence change when starving: −(0.002−0.001) = −0.001/step

DANGER_LIFE_DELAY_STEPS: int   = 180   # 3 s of danger contact before life drains
LIFE_DRAIN_RATE:         float = 0.001 # life lost per step during drain

# Instant rewards (eating, patting, feeding)
FOOD_PLEASURE:       float = 0.6    # valence spike from eating
FOOD_SATIATION_GAIN: float = 0.35   # satiation restored by eating
FOOD_LIFE_GAIN:      float = 0.15

OTHER_ENTITY_PLEASURE:   float = 0.005   # valence per step while touching other entity
OTHER_ENTITY_TOUCH_DIST: float = ENTITY_RADIUS * 2 + 2.0

# ── Prongs / eat zone ─────────────────────────────────────────────────────────
PRONG_ANGLE:      float = 0.60   # rad from heading (±34°) — where prongs sit
PRONG_LENGTH:     float = 12.0    # px beyond entity radius
PRONG_BASE_W:     float = 0.35   # half-angle of prong base on entity surface (rad)
PRONG_CAPSULE_R:  float = 6.0    # collision capsule radius along each prong
EAT_ZONE_ANGLE:   float = 0.60   # food must be within ±34° of heading to be eatable
EAT_ZONE_REACH:   float = ENTITY_RADIUS + FOOD_RADIUS + PRONG_LENGTH + 2.0

# ── Social actions ────────────────────────────────────────────────────────────
PAT_PLEASURE:    float = 0.1    # pleasure given to AI on a single pat press
PAT_TOUCH_DIST:  float = ENTITY_RADIUS * 2 + 20.0  # generous proximity for pat
FEED_REACH_BONUS: float = 16.0  # extra reach for food-near-player detection on feed


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
    valence:   float = 0.0   # ∈ [-1, 1]: negative = pain, positive = pleasure
    satiation: float = 0.0   # ∈ [-1, 1]: negative = hungry, positive = satiated

    def as_array(self) -> np.ndarray:
        """Return [life, satiation_norm, valence_norm] each ∈ [0, 1]."""
        return np.array([
            self.life,
            (self.satiation + 1.0) * 0.5,
            (self.valence   + 1.0) * 0.5,
        ], dtype=np.float32)

    def reset(self) -> None:
        self.life      = 1.0
        self.valence   = 0.0
        self.satiation = 0.0


@dataclass
class TouchState:
    wall:  bool = False
    food:  bool = False   # overlapping with food (can eat)
    danger: bool = False
    other: bool = False   # touching the other entity

    def as_array(self) -> np.ndarray:
        return np.array(
            [float(self.wall), float(self.food),
             float(self.danger), float(self.other)],
            dtype=np.float32,
        )


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
        self.touch   = TouchState()
        self.alive   = True
        self.danger_contact_steps: int = 0

    def apply_action(self, forward: float, turn: float) -> None:
        self.heading = (self.heading + turn * self.turn_speed) % (2.0 * np.pi)
        dx = np.cos(self.heading) * forward * self.speed
        dy = np.sin(self.heading) * forward * self.speed
        self.x = float(np.clip(self.x + dx, self.radius, WORLD_W - self.radius))
        self.y = float(np.clip(self.y + dy, self.radius, WORLD_H - self.radius))


# ── World ─────────────────────────────────────────────────────────────────────

# Starting positions (used on reset)
_AI_START     = (WORLD_W * 0.25, WORLD_H * 0.5, 0.0)
_PLAYER_START = (WORLD_W * 0.75, WORLD_H * 0.5, np.pi)


class World:
    """
    Main interface:

        obs = world.get_observation(world.ai)     # shape (49,), all in [0, 1]
        world.step(ai_action=(fwd, turn, eat),
                   player_action=(fwd, turn, eat),
                   pat=False, feed=False)

    Actions: (forward ∈ [-1,1], turn ∈ [-1,1], eat ∈ {0,1}).

    When either entity's life reaches 0 the world auto-resets.
    Call world.reset() explicitly to force a reset.
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
        self.step_count: int = 0
        self.death_count: int = 0   # how many resets have happened

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

        # Apply movement
        self.ai.apply_action(
            float(np.clip(ai_fwd, -1, 1)),
            float(np.clip(ai_turn, -1, 1)),
        )
        if self.player_active:
            self.player.apply_action(
                float(np.clip(pl_fwd, -1, 1)),
                float(np.clip(pl_turn, -1, 1)),
            )

        # Entity–entity collision (symmetric push)
        self._resolve_entities()

        # Social player actions — before food physics so positions are stable
        if pat:
            self._try_pat_ai()
        if feed:
            self._try_feed_ai()

        # Food interaction — player body only participates when active
        ai_ate = self._resolve_food(self.ai, bool(ai_eat > 0.5))
        pl_ate = (self._resolve_food(self.player, bool(pl_eat > 0.5))
                  if self.player_active else False)

        # Update touch + meters — same guard for player
        self._update_entity(self.ai, other=self.player, ate_food=ai_ate)
        if self.player_active:
            self._update_entity(self.player, other=self.ai, ate_food=pl_ate)

        self._tick_food_respawns()
        self.step_count += 1

        # Auto-reset on death
        if not self.ai.alive or not self.player.alive:
            self.death_count += 1
            self.reset(keep_counts=True)

    def reset(self, keep_counts: bool = False) -> None:
        """Reset both entities and respawn all items at random positions."""
        counts = (self.step_count, self.death_count) if keep_counts else (0, 0)
        self.ai     = Entity(*_AI_START)
        self.player = Entity(*_PLAYER_START)
        self.foods   = self._spawn_foods(FOOD_COUNT)
        self.dangers = self._spawn_dangers(DANGER_COUNT)
        self.step_count  = 0 if not keep_counts else counts[0]
        self.death_count = counts[1] if keep_counts else 0

    def get_observation(self, entity: Entity) -> np.ndarray:
        """49-dim observation for the given entity (AI or player)."""
        other  = self.player if entity is self.ai else self.ai
        vision = self._cast_rays(entity, other).flatten()   # (42,)
        touch  = entity.touch.as_array()                    # (4,)
        meters = entity.meters.as_array()                   # (3,)
        return np.concatenate([vision, touch, meters])      # (49,)

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
        """
        Returns (N_RAYS, 2) float32.
          col 0: hit_type / (HIT_TYPES - 1)
          col 1: distance / VISION_RANGE
        """
        angles = np.linspace(
            entity.heading - VISION_HALF_ANGLE,
            entity.heading + VISION_HALF_ANGLE,
            N_RAYS,
        )
        result = np.zeros((N_RAYS, 2), dtype=np.float32)
        for i, angle in enumerate(angles):
            hit_type, dist = self._cast_ray(entity, angle, other)
            result[i, 0] = hit_type / (HIT_TYPES - 1)
            # Proximity: strong when close, zero when nothing detected
            result[i, 1] = (1.0 - dist / VISION_RANGE) if hit_type != HIT_NONE else 0.0
        return result

    @staticmethod
    def _ray_nearest(
        ox: float, oy: float, dx: float, dy: float,
        items: list, radius: float,
        skip_dist: float, best_dist: float,
        active_only: bool = False,
    ) -> Optional[float]:
        """Return the closest ray-circle hit among items, or None."""
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

        d = self._ray_nearest(ox, oy, dx, dy, self.foods,   FOOD_RADIUS,   skip_dist, best_dist, active_only=True)
        if d is not None:
            best_dist, best_type = d, HIT_FOOD

        d = self._ray_nearest(ox, oy, dx, dy, self.dangers, DANGER_RADIUS, skip_dist, best_dist)
        if d is not None:
            best_dist, best_type = d, HIT_DANGER

        if not (other is self.player and not self.player_active):
            d = _ray_circle_dist(ox, oy, dx, dy, other.x, other.y, other.radius)
            if d is not None and skip_dist < d < best_dist:
                best_dist, best_type = d, HIT_OTHER

        return best_type, best_dist

    # ── Food physics ──────────────────────────────────────────────────────────

    def _prong_tips(self, entity: Entity) -> List[Tuple[float, float]]:
        """Return tip (x, y) of both prongs (used for visual sync in game.py)."""
        r = entity.radius
        h = entity.heading
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
        """Push food away from origin (ox, oy) until separation >= min_dist."""
        dx = food.x - ox
        dy = food.y - oy
        dist = float(np.hypot(dx, dy))
        nx = dx / dist if dist > 0.1 else 1.0
        ny = dy / dist if dist > 0.1 else 0.0
        push = min_dist - dist + 0.5
        food.x = float(np.clip(food.x + nx * push, FOOD_RADIUS + 2, WORLD_W - FOOD_RADIUS - 2))
        food.y = float(np.clip(food.y + ny * push, FOOD_RADIUS + 2, WORLD_H - FOOD_RADIUS - 2))

    def _apply_prong_push(self, food: Food, entity: Entity) -> None:
        """
        Treat each prong as a capsule (line segment + radius PRONG_CAPSULE_R).
        Find the closest point on the segment to the food centre; push food
        away if overlapping.  Stops after the first prong that overlaps.
        """
        min_d = PRONG_CAPSULE_R + FOOD_RADIUS
        cx, cy = entity.x, entity.y
        r = entity.radius
        h = entity.heading
        for side in (-1.0, 1.0):
            angle  = h + side * PRONG_ANGLE
            cos_a  = float(np.cos(angle))
            sin_a  = float(np.sin(angle))
            # Segment: base at body surface → tip
            p1x, p1y = cx + r * cos_a, cy + r * sin_a
            p2x, p2y = cx + (r + PRONG_LENGTH) * cos_a, cy + (r + PRONG_LENGTH) * sin_a
            # Closest point on segment to food
            sx, sy     = p2x - p1x, p2y - p1y
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 1e-6:
                continue
            t = ((food.x - p1x) * sx + (food.y - p1y) * sy) / seg_len_sq
            t = float(np.clip(t, 0.0, 1.0))
            near_x = p1x + t * sx
            near_y = p1y + t * sy
            dist = float(np.hypot(food.x - near_x, food.y - near_y))
            if 0.1 < dist < min_d:
                self._push_food(food, near_x, near_y, min_d)
                break  # one prong push per food per step

    def _resolve_food(self, entity: Entity, eat_triggered: bool) -> bool:
        """
        Handle entity↔food interaction per food item:
          1. Body overlap + eat zone + eat_triggered → consume
          2. Body overlap only → push from body centre
          3. In eat zone (prong reach) + eat_triggered → consume
          4. Otherwise → push from prong tip if overlapping
        Returns True if a food item was consumed this step.
        """
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
        """True when food is within the frontal eat arc (between the prongs)."""
        dx   = food.x - entity.x
        dy   = food.y - entity.y
        dist = float(np.hypot(dx, dy))
        if dist > EAT_ZONE_REACH or dist < 0.1:
            return False
        food_angle = float(np.arctan2(dy, dx))
        diff = (food_angle - entity.heading + np.pi) % (2.0 * np.pi) - np.pi
        return abs(diff) <= EAT_ZONE_ANGLE

    def _resolve_entities(self) -> None:
        """Symmetric push between AI and player when they overlap."""
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
            self.ai.x = float(np.clip(self.ai.x - nx * overlap,     r, WORLD_W - r))
            self.ai.y = float(np.clip(self.ai.y - ny * overlap,     r, WORLD_H - r))
            r = self.player.radius
            self.player.x = float(np.clip(self.player.x + nx * overlap, r, WORLD_W - r))
            self.player.y = float(np.clip(self.player.y + ny * overlap, r, WORLD_H - r))

    def _try_pat_ai(self) -> None:
        """Player pats the AI: one press adds PAT_PLEASURE to AI valence."""
        dist = float(np.hypot(self.ai.x - self.player.x, self.ai.y - self.player.y))
        if dist < PAT_TOUCH_DIST:
            self.ai.meters.valence = min(1.0, self.ai.meters.valence + PAT_PLEASURE)

    def _try_feed_ai(self) -> None:
        """
        Player feeds AI: consume a food item that is touching the player AND
        inside the AI's eat zone.  AI gets the full food reward.
        """
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
                return   # feed one item per press

    # ── Entity update ─────────────────────────────────────────────────────────

    def _update_touch(self, entity: Entity, other: Entity) -> None:
        """Refresh the entity's touch-state flags."""
        t = entity.touch
        r = entity.radius
        t.wall = (
            entity.x <= r + 1 or entity.x >= WORLD_W - r - 1 or
            entity.y <= r + 1 or entity.y >= WORLD_H - r - 1
        )
        t.food = any(
            f.active and np.hypot(entity.x - f.x, entity.y - f.y) < r + FOOD_RADIUS
            for f in self.foods
        )
        t.danger = any(
            np.hypot(entity.x - d.x, entity.y - d.y) < r + DANGER_RADIUS
            for d in self.dangers
        )
        other_present = not (other is self.player and not self.player_active)
        t.other = other_present and np.hypot(entity.x - other.x, entity.y - other.y) < OTHER_ENTITY_TOUCH_DIST

    def _step_valence_and_life(self, entity: Entity) -> None:
        """
        Update valence and life based on current danger/starvation state.
        Also sets entity.alive = False when life reaches 0.
        """
        m = entity.meters
        t = entity.touch
        draining = (
            (t.danger and entity.danger_contact_steps >= DANGER_LIFE_DELAY_STEPS)
            or (m.satiation <= -1.0 and m.valence <= -0.5)
        )
        if draining:
            m.life    = max(0.0, m.life - LIFE_DRAIN_RATE)
            m.valence = -(0.5 + 0.5 * (1.0 - m.life))
        elif t.danger:
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
        self._update_touch(entity, other)

        # ── Danger contact counter ────────────────────────────────────────────
        entity.danger_contact_steps = (
            entity.danger_contact_steps + 1 if entity.touch.danger else 0
        )

        # ── Satiation drifts negative over time ──────────────────────────────
        m.satiation = max(-1.0, m.satiation - SATIATION_DRAIN_RATE)

        # ── Food + social rewards ─────────────────────────────────────────────
        if ate_food:
            m.valence   = min(1.0, m.valence   + FOOD_PLEASURE)
            m.satiation = min(1.0, m.satiation + FOOD_SATIATION_GAIN)
            m.life      = min(1.0, m.life      + FOOD_LIFE_GAIN)
        if entity.touch.other:
            m.valence = min(1.0, m.valence + OTHER_ENTITY_PLEASURE)

        self._step_valence_and_life(entity)

    # ── Food respawn ──────────────────────────────────────────────────────────

    def _tick_food_respawns(self) -> None:
        for food in self.foods:
            if not food.active:
                food.respawn_timer -= 1
                if food.respawn_timer <= 0:
                    food.active = True
                    food.x = float(self._rng.uniform(
                        FOOD_RADIUS + 30, WORLD_W - FOOD_RADIUS - 30))
                    food.y = float(self._rng.uniform(
                        FOOD_RADIUS + 30, WORLD_H - FOOD_RADIUS - 30))

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_foods(self, count: int) -> List[Food]:
        return [
            Food(
                x=float(self._rng.uniform(FOOD_RADIUS + 30, WORLD_W - FOOD_RADIUS - 30)),
                y=float(self._rng.uniform(FOOD_RADIUS + 30, WORLD_H - FOOD_RADIUS - 30)),
            )
            for _ in range(count)
        ]

    def _spawn_dangers(self, count: int) -> List[Danger]:
        return [
            Danger(
                x=float(self._rng.uniform(DANGER_RADIUS + 30, WORLD_W - DANGER_RADIUS - 30)),
                y=float(self._rng.uniform(DANGER_RADIUS + 30, WORLD_H - DANGER_RADIUS - 30)),
            )
            for _ in range(count)
        ]


# ── Pure geometry helpers ─────────────────────────────────────────────────────

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
