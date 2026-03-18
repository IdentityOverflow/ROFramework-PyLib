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
  • Danger visible in forward cone: valence −= DANGER_VISION_PENALTY × proximity
  • After DANGER_LIFE_DELAY_STEPS consecutive contact steps (~3 s at 60 fps):
      life drains at LIFE_DRAIN_RATE each step, and
      valence = −(0.5 + 0.5 × (1 − life))   [pain rises as life falls]

Hunger
  • Satiation drains at SATIATION_DRAIN_RATE per step.
  • Once satiation ≤ −1.0, valence drifts negative at HUNGER_VALENCE_DRAIN rate.
  • Once valence ≤ −0.5 from starvation, life also drains.

Recovery
  • Food visible in forward cone: valence += FOOD_VISION_REWARD × proximity
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
WORLD_H: int = 1200

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
SATIATION_DRAIN_RATE: float = 0.0001   # hunger kicks in at ~20000 steps (~5.5 min at 60fps)
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

# ── Spawn behaviour ───────────────────────────────────────────────────────────
# When True the AI spawns adjacent to a random food item (facing it) so the
# eat reflex gets a reward signal within the first few seconds of each episode.
SPAWN_NEAR_FOOD: bool  = True
SPAWN_NEAR_DIST: float = ENTITY_RADIUS * 4   # ~72 px  — close but not on top

# ── Vision-based valence gradients ────────────────────────────────────────────
# Symmetric proximity signals in the forward cone so the brain gets a learnable
# gradient toward food and away from danger — not just the contact spike/penalty.
#
# Danger: −0.002 / step at max proximity (≈ −0.12 / s at 60 fps)
# Food:   +0.001 / step at max proximity (≈ +0.06 / s at 60 fps)
#   Kept half the danger penalty so food-seeking doesn't override avoidance when
#   both are visible.  No effect while already eating (contact spike dominates).
DANGER_VISION_PENALTY:        float = 0.002
FOOD_VISION_REWARD:           float = 0.002
DANGER_CONTACT_VALENCE_FLOOR: float = -0.5   # valence is clamped to this on danger touch


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
        self.last_action: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_action_teacher_forced: bool = False

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

    When either entity's life reaches 0 the world auto-resets unless
    ``no_reset_on_death`` is True, in which case the AI is kept alive at
    life=0 with maximum pain (valence=−1) so it can learn to recover by eating.
    """

    def __init__(self, seed: int = 42, no_reset_on_death: bool = False,
                 cfg: Optional[dict] = None) -> None:
        c = cfg or {}
        self._seed = seed
        self._rng  = np.random.default_rng(seed)
        self.paused: bool = False
        self.no_reset_on_death: bool = no_reset_on_death

        # ── Configurable world parameters (overridable via ruleset JSON) ───────
        # World layout
        self._food_count             = int(c.get("food_count",            FOOD_COUNT))
        self._danger_count           = int(c.get("danger_count",          DANGER_COUNT))
        self._food_respawn_steps     = int(c.get("food_respawn_steps",    FOOD_RESPAWN_STEPS))
        self._spawn_near_food        = bool(c.get("spawn_near_food",      SPAWN_NEAR_FOOD))
        # Entity movement
        self._entity_speed           = float(c.get("entity_speed",        ENTITY_SPEED))
        self._entity_turn_speed      = float(c.get("entity_turn_speed",   ENTITY_TURN_SPEED))
        # Vision
        self._vision_range           = float(c.get("vision_range",        VISION_RANGE))
        # Meter dynamics
        self._satiation_drain_rate   = float(c.get("satiation_drain_rate",   SATIATION_DRAIN_RATE))
        self._valence_decay_neg      = float(c.get("valence_decay_neg",      VALENCE_DECAY_NEG))
        self._valence_decay_pos      = float(c.get("valence_decay_pos",      VALENCE_DECAY_POS))
        self._hunger_valence_drain   = float(c.get("hunger_valence_drain",   HUNGER_VALENCE_DRAIN))
        self._danger_life_delay_steps= int(c.get("danger_life_delay_steps",  DANGER_LIFE_DELAY_STEPS))
        self._life_drain_rate        = float(c.get("life_drain_rate",        LIFE_DRAIN_RATE))
        # Rewards
        self._food_pleasure          = float(c.get("food_pleasure",          FOOD_PLEASURE))
        self._food_satiation_gain    = float(c.get("food_satiation_gain",    FOOD_SATIATION_GAIN))
        self._food_life_gain         = float(c.get("food_life_gain",         FOOD_LIFE_GAIN))
        self._other_entity_pleasure  = float(c.get("other_entity_pleasure",  OTHER_ENTITY_PLEASURE))
        self._other_entity_touch_cap = float(c.get("other_entity_touch_cap", OTHER_ENTITY_TOUCH_CAP))
        self._danger_vis_penalty          = float(c.get("danger_vision_penalty",        DANGER_VISION_PENALTY))
        self._food_vis_reward             = float(c.get("food_vision_reward",           FOOD_VISION_REWARD))
        self._danger_contact_valence_floor= float(c.get("danger_contact_valence_floor", DANGER_CONTACT_VALENCE_FLOOR))
        # ──────────────────────────────────────────────────────────────────────

        self.dangers = self._spawn_dangers(self._danger_count)
        self.foods   = self._spawn_foods(self._food_count)
        self.agents: List[Entity] = []
        self.player = Entity(*_PLAYER_START)
        self._configure_entity(self.player)
        self.player_active: bool = True
        self.step_count: int  = 0
        self.death_count: int = 0

    @property
    def ai(self) -> Entity:
        """Convenience accessor for the first active agent."""
        for a in self.agents:
            if a is not None:
                return a
        raise IndexError("No active agents")

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        ai_action:     Tuple[float, float, float] = (0.0, 0.0, 0.0),
        player_action: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        pat:  bool = False,
        feed: bool = False,
        ai_actions: Optional[List[Tuple[float, float, float]]] = None,
        teleop_slots: Optional[set[int]] = None,
    ) -> None:
        if self.paused:
            return

        # Build per-agent action list: ai_actions overrides ai_action for all slots
        if ai_actions is None:
            ai_actions = [ai_action] + [(0.0, 0.0, 0.0)] * (len(self.agents) - 1)

        pl_fwd, pl_turn, pl_eat = player_action
        teleop_slots = teleop_slots or set()

        for i, agent in enumerate(self.agents):
            if agent is None:
                continue
            if i < len(ai_actions):
                raw_fwd, raw_turn, raw_eat = ai_actions[i]
            else:
                raw_fwd, raw_turn, raw_eat = 0.0, 0.0, 0.0
            fwd = float(np.clip(raw_fwd, -1, 1))
            turn = float(np.clip(raw_turn, -1, 1))
            eat = 1.0 if float(raw_eat) > 0.5 else 0.0
            agent.last_action = (fwd, turn, eat)
            agent.last_action_teacher_forced = (i in teleop_slots)
            agent.apply_action(fwd, turn)

        if self.player_active:
            p_fwd = float(np.clip(pl_fwd,  -1, 1))
            p_turn = float(np.clip(pl_turn, -1, 1))
            p_eat = 1.0 if float(pl_eat) > 0.5 else 0.0
            self.player.last_action = (p_fwd, p_turn, p_eat)
            self.player.last_action_teacher_forced = False
            self.player.apply_action(p_fwd, p_turn)

        self._resolve_entities()

        if pat:
            self._try_pat_ai()
        if feed:
            self._try_feed_ai()

        agent_ate = [
            (self._resolve_food(
                agent,
                bool(ai_actions[i][2] > 0.5) if i < len(ai_actions) else False,
            ) if agent is not None else False)
            for i, agent in enumerate(self.agents)
        ]
        pl_ate = (self._resolve_food(self.player, bool(pl_eat > 0.5))
                  if self.player_active else False)

        # Build per-entity "others" list (all entities visible to each entity)
        active_agents = [a for a in self.agents if a is not None]
        all_entities = ([self.player] if self.player_active else []) + active_agents
        for i, agent in enumerate(self.agents):
            if agent is None:
                continue
            others = [e for e in all_entities if e is not agent]
            self._update_entity(agent, others, agent_ate[i])
        if self.player_active:
            self._update_entity(self.player, active_agents, pl_ate)

        self._tick_food_respawns()
        self.step_count += 1

        # Death handling
        dead_slots = [i for i, a in enumerate(self.agents) if a is not None and not a.alive]
        if not self.player.alive:
            self.death_count += 1
            self.reset(keep_counts=True)
            return

        if dead_slots:
            for i in dead_slots:
                self.death_count += 1
                if self.no_reset_on_death:
                    self.agents[i].alive = True
                    self.agents[i].meters.life = 0.0
                    self.agents[i].meters.valence = -1.0
                else:
                    self._respawn_agent(i)

    def reset(self, keep_counts: bool = False) -> None:
        counts = (self.step_count, self.death_count) if keep_counts else (0, 0)
        self.dangers = self._spawn_dangers(self._danger_count)
        self.foods   = self._spawn_foods(self._food_count)
        self.agents  = []
        self.player  = Entity(*_PLAYER_START)
        self._configure_entity(self.player)
        self.step_count  = counts[0] if keep_counts else 0
        self.death_count = counts[1] if keep_counts else 0

    def add_agent(self, x: Optional[float] = None, y: Optional[float] = None,
                  heading: Optional[float] = None) -> int:
        """Spawn a new AI entity and return its slot index in self.agents."""
        if x is None:
            start = self._near_food_start() if self._spawn_near_food else _AI_START
            sx, sy, sh = start
        else:
            sx, sy, sh = x, y, heading or 0.0
        entity = Entity(float(sx), float(sy), float(sh))
        self._configure_entity(entity)
        # Reuse first empty slot to preserve slot→index mapping for live agents
        for i, a in enumerate(self.agents):
            if a is None:
                self.agents[i] = entity
                return i
        self.agents.append(entity)
        return len(self.agents) - 1

    def remove_agent(self, slot: int) -> None:
        """Remove an AI entity by slot index (leaves None placeholder to preserve higher-slot indices)."""
        if 0 <= slot < len(self.agents):
            self.agents[slot] = None

    def _respawn_agent(self, slot: int) -> None:
        """Replace a dead agent with a fresh entity at a spawn position."""
        start = self._near_food_start() if self._spawn_near_food else _AI_START
        entity = Entity(*start)
        self._configure_entity(entity)
        self.agents[slot] = entity

    def _configure_entity(self, entity: Entity) -> None:
        """Apply world-config movement settings to a newly created entity."""
        entity.speed      = self._entity_speed
        entity.turn_speed = self._entity_turn_speed

    def get_observation(self, entity: Entity) -> np.ndarray:
        """263-dim observation vector, all values ∈ [0, 1]."""
        active   = [a for a in self.agents if a is not None]
        all_ents = ([self.player] if self.player_active else []) + active
        others   = [e for e in all_ents if e is not entity]
        vision   = self._cast_rays(entity, others).flatten()   # (242,)
        tactile  = entity.tactile.as_array()                   # (18,)
        meters   = entity.meters.as_array()                    # (3,)
        return np.concatenate([vision, tactile, meters])       # (263,)

    def get_agent_observation(self, slot: int) -> np.ndarray:
        return self.get_observation(self.agents[slot])

    def get_ai_observation(self) -> np.ndarray:
        return self.get_observation(self.ai)

    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    # ── Ray casting ───────────────────────────────────────────────────────────

    def _cast_rays(self, entity: Entity, others: List[Entity]) -> np.ndarray:
        """Returns (N_RAYS, 2) float32: [type_norm, proximity] per ray."""
        angles = np.linspace(
            entity.heading - VISION_HALF_ANGLE,
            entity.heading + VISION_HALF_ANGLE,
            N_RAYS,
        )
        result = np.zeros((N_RAYS, 2), dtype=np.float32)
        for i, angle in enumerate(angles):
            hit_type, dist = self._cast_ray(entity, angle, others)
            result[i, 0] = hit_type / (HIT_TYPES - 1)
            result[i, 1] = (1.0 - dist / self._vision_range) if hit_type != HIT_NONE else 0.0
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
        self, entity: Entity, angle: float, others: List[Entity]
    ) -> Tuple[int, float]:
        dx, dy    = float(np.cos(angle)), float(np.sin(angle))
        ox, oy    = entity.x, entity.y
        skip_dist = entity.radius
        best_dist = self._vision_range
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

        for other in others:
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

    def _consume_food(self, food: Food) -> None:
        food.active = False
        food.respawn_timer = self._food_respawn_steps

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

    def _push_apart(self, a: Entity, b: Entity) -> None:
        dx   = b.x - a.x
        dy   = b.y - a.y
        dist = float(np.hypot(dx, dy))
        min_dist = a.radius + b.radius
        if 0 < dist < min_dist:
            overlap = (min_dist - dist + 0.5) * 0.5
            nx, ny  = dx / dist, dy / dist
            ra = a.radius
            a.x = float(np.clip(a.x - nx * overlap, ra, WORLD_W - ra))
            a.y = float(np.clip(a.y - ny * overlap, ra, WORLD_H - ra))
            rb = b.radius
            b.x = float(np.clip(b.x + nx * overlap, rb, WORLD_W - rb))
            b.y = float(np.clip(b.y + ny * overlap, rb, WORLD_H - rb))

    def _push_from_danger(self, entity: Entity) -> None:
        """Push entity out of any overlapping danger object (dangers are static)."""
        min_dist = entity.radius + DANGER_RADIUS
        for danger in self.dangers:
            dx = entity.x - danger.x
            dy = entity.y - danger.y
            dist = float(np.hypot(dx, dy))
            if 0 < dist < min_dist:
                nx, ny = dx / dist, dy / dist
                push = min_dist - dist + 0.5
                r = entity.radius
                entity.x = float(np.clip(entity.x + nx * push, r, WORLD_W - r))
                entity.y = float(np.clip(entity.y + ny * push, r, WORLD_H - r))

    def _resolve_entities(self) -> None:
        if self.player_active:
            for agent in self.agents:
                if agent is not None:
                    self._push_apart(agent, self.player)
        for i in range(len(self.agents)):
            if self.agents[i] is None:
                continue
            for j in range(i + 1, len(self.agents)):
                if self.agents[j] is not None:
                    self._push_apart(self.agents[i], self.agents[j])
        all_entities = ([self.player] if self.player_active else []) + [
            a for a in self.agents if a is not None
        ]
        for entity in all_entities:
            self._push_from_danger(entity)

    def _try_pat_ai(self) -> None:
        for agent in self.agents:
            if agent is None:
                continue
            dist = float(np.hypot(agent.x - self.player.x, agent.y - self.player.y))
            if dist < PAT_TOUCH_DIST:
                agent.meters.valence = min(1.0, agent.meters.valence + PAT_PLEASURE)

    def _try_feed_ai(self) -> None:
        for food in self.foods:
            if not food.active:
                continue
            in_player = (float(np.hypot(food.x - self.player.x, food.y - self.player.y))
                         < self.player.radius + FOOD_RADIUS + FEED_REACH_BONUS)
            for agent in self.agents:
                if agent is None:
                    continue
                if in_player and self._in_eat_zone(agent, food):
                    food.active = False
                    food.respawn_timer = self._food_respawn_steps
                    m = agent.meters
                    m.valence   = min(1.0, m.valence   + self._food_pleasure)
                    m.satiation = min(1.0, m.satiation + self._food_satiation_gain)
                    m.life      = min(1.0, m.life      + self._food_life_gain)
                    return

    # ── Tactile sensing ───────────────────────────────────────────────────────

    def _update_tactile(self, entity: Entity, others: List[Entity]) -> None:
        """Recompute all receptor signals for entity."""
        body   = np.zeros(N_TOUCH_BODY,   dtype=np.float32)
        prongs = np.zeros(N_TOUCH_PRONGS, dtype=np.float32)
        self._sense_walls(entity, body)
        self._sense_foods(entity, body)
        self._sense_dangers(entity, body)
        for other in others:
            self._sense_other_entity(entity, other, body)
        self._sense_prongs(entity, others, prongs)
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
        threshold = entity.radius + other.radius + TOUCH_MARGIN
        if float(np.hypot(other.x - entity.x, other.y - entity.y)) < threshold:
            angle = float(np.arctan2(other.y - entity.y, other.x - entity.x))
            _write_receptor(body, entity.heading, angle, TOUCH_ENTITY)

    def _sense_prongs(self, entity: Entity, others: List[Entity], prongs: np.ndarray) -> None:
        for pi, side in enumerate((-1.0, 1.0)):
            prongs[pi] = max(
                (self._prong_signal(entity, side, other) for other in others),
                default=0.0,
            )

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
        if _capsule_dist(other.x, other.y, p1x, p1y, p2x, p2y) < PRONG_CAPSULE_R + other.radius + TOUCH_MARGIN:
            max_sig = max(max_sig, TOUCH_ENTITY)
        return max_sig

    # ── Vision-based danger aversion ─────────────────────────────────────────

    def _danger_vision_penalty(self, entity: Entity) -> float:
        """
        Proximity-scaled valence penalty for danger items visible in the
        forward cone.  Returns a negative value ∈ [−DANGER_VISION_PENALTY, 0].
        """
        max_prox = 0.0
        for danger in self.dangers:
            dx = danger.x - entity.x
            dy = danger.y - entity.y
            dist = float(np.hypot(dx, dy))
            if dist >= self._vision_range:
                continue
            angle = float(np.arctan2(dy, dx))
            diff = abs((angle - entity.heading + np.pi) % (2.0 * np.pi) - np.pi)
            if diff <= VISION_HALF_ANGLE:
                max_prox = max(max_prox, 1.0 - dist / self._vision_range)
        return -self._danger_vis_penalty * max_prox

    def _food_vision_reward(self, entity: Entity) -> float:
        """
        Proximity-scaled valence reward for active food visible in the forward
        cone.  Symmetric counterpart to _danger_vision_penalty.
        Returns a positive value ∈ [0, FOOD_VISION_REWARD].
        """
        max_prox = 0.0
        for food in self.foods:
            if not food.active:
                continue
            dx = food.x - entity.x
            dy = food.y - entity.y
            dist = float(np.hypot(dx, dy))
            if dist >= self._vision_range:
                continue
            angle = float(np.arctan2(dy, dx))
            diff = abs((angle - entity.heading + np.pi) % (2.0 * np.pi) - np.pi)
            if diff <= VISION_HALF_ANGLE:
                max_prox = max(max_prox, 1.0 - dist / self._vision_range)
        return self._food_vis_reward * max_prox

    # ── Entity update ─────────────────────────────────────────────────────────

    def _step_valence_and_life(self, entity: Entity) -> None:
        m        = entity.meters
        in_danger = entity.tactile.has_danger()
        draining = (
            (in_danger and entity.danger_contact_steps >= self._danger_life_delay_steps)
            or (m.satiation <= -1.0 and m.valence <= -0.5)
        )
        if draining:
            m.life    = max(0.0, m.life - self._life_drain_rate)
            m.valence = -(0.5 + 0.5 * (1.0 - m.life))
        elif in_danger:
            m.valence = min(m.valence, self._danger_contact_valence_floor)
        elif m.satiation <= -1.0:
            net = self._hunger_valence_drain - self._valence_decay_neg
            m.valence = float(np.clip(m.valence - net, -1.0, 1.0))
        elif m.valence > 0.0:
            m.valence = max(0.0, m.valence - self._valence_decay_pos)
        elif m.valence < 0.0:
            m.valence = min(0.0, m.valence + self._valence_decay_neg)
        if m.life <= 0.0:
            entity.alive = False

    def _update_entity(
        self, entity: Entity, others: List[Entity], ate_food: bool
    ) -> None:
        m = entity.meters
        self._update_tactile(entity, others)

        entity.danger_contact_steps = (
            entity.danger_contact_steps + 1 if entity.tactile.has_danger() else 0
        )

        m.satiation = max(-1.0, m.satiation - self._satiation_drain_rate)

        if ate_food:
            m.valence   = min(1.0, m.valence   + self._food_pleasure)
            m.satiation = min(1.0, m.satiation + self._food_satiation_gain)
            m.life      = min(1.0, m.life      + self._food_life_gain)
        if entity.tactile.has_other() and m.valence < self._other_entity_touch_cap:
            m.valence = min(self._other_entity_touch_cap, m.valence + self._other_entity_pleasure)

        self._step_valence_and_life(entity)

        # Vision-based valence gradients — only when not in tactile contact
        # (contact dynamics in _step_valence_and_life already dominate then).
        if not entity.tactile.has_danger():
            penalty = self._danger_vision_penalty(entity)
            if penalty < 0.0:
                m.valence = max(-1.0, m.valence + penalty)
        if not ate_food:
            reward = self._food_vision_reward(entity)
            if reward > 0.0:
                m.valence = min(1.0, m.valence + reward)

    # ── Food respawn ──────────────────────────────────────────────────────────

    def _tick_food_respawns(self) -> None:
        for food in self.foods:
            if not food.active:
                food.respawn_timer -= 1
                if food.respawn_timer <= 0:
                    food.active = True
                    food.x, food.y = self._safe_food_pos()

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _near_food_start(self) -> Tuple[float, float, float]:
        """Return (x, y, heading) for the AI spawn point adjacent to a random food."""
        active = [f for f in self.foods if f.active]
        if not active:
            return _AI_START
        food = active[int(self._rng.integers(len(active)))]
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        x = float(np.clip(food.x + SPAWN_NEAR_DIST * np.cos(angle),
                          ENTITY_RADIUS + 5, WORLD_W - ENTITY_RADIUS - 5))
        y = float(np.clip(food.y + SPAWN_NEAR_DIST * np.sin(angle),
                          ENTITY_RADIUS + 5, WORLD_H - ENTITY_RADIUS - 5))
        heading = float(np.arctan2(food.y - y, food.x - x))   # face the food
        return x, y, heading

    def _safe_food_pos(self) -> Tuple[float, float]:
        """Sample a food position that doesn't overlap any danger (max 20 tries)."""
        min_dist = FOOD_RADIUS + DANGER_RADIUS + 2.0
        x, y = 0.0, 0.0
        for _ in range(20):
            x = float(self._rng.uniform(FOOD_RADIUS + 40, WORLD_W - FOOD_RADIUS - 40))
            y = float(self._rng.uniform(FOOD_RADIUS + 40, WORLD_H - FOOD_RADIUS - 40))
            if all(np.hypot(x - d.x, y - d.y) >= min_dist for d in self.dangers):
                break  # accept first non-overlapping position
        return x, y

    def _spawn_foods(self, count: int) -> List[Food]:
        return [Food(*self._safe_food_pos()) for _ in range(count)]

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
