# Embodied Environment — Integration Guide

The game exposes the **AI entity's** sensory stream and accepts motor commands
from an external process over ZeroMQ.  The human player keeps full keyboard
control; only the AI slot is wired through the connector.

---

## Architecture

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│       game.py  (60 fps)     │         │     your agent process       │
│                             │  ──obs─▶ │                             │
│  GameConnector  PUSH :5557  │         │  AgentConnector  PULL :5557  │
│                 PULL :5558  │ ◀─act── │                 PUSH :5558  │
└─────────────────────────────┘         └─────────────────────────────┘
```

The game publishes one observation packet per step (~60 fps) without waiting.
Your agent can consume and respond at any rate — the game uses the most recent
action it has received, or holds the previous one if nothing new has arrived.

---

## Starting the game with the connector enabled

```bash
cd experiments/embodied
python game.py --connect
```

The terminal will print the ports once the sockets are bound:

```
[connector] listening on ports 5557 / 5558
```

The game window opens normally.  Without `--connect`, behaviour is identical to
before — no ZeroMQ overhead.

---

## Connecting from Python

```python
from connector import AgentConnector
import numpy as np

client = AgentConnector()   # defaults: host=127.0.0.1, ports 5557/5558
client.connect()

obs, reward, done = client.recv_obs()   # wait for first frame

while True:
    # --- your agent logic here ---
    fwd  =  1.0   # ∈ [-1, 1]  forward/reverse
    turn =  0.0   # ∈ [-1, 1]  right/left
    eat  =  0.0   # ∈  {0, 1}  attempt to eat/grab food

    client.send_action((fwd, turn, eat))
    obs, reward, done = client.recv_obs()   # blocks until next frame

client.close()
```

---

## Observation packet — `obs` (float32 array, shape `(263,)`)

| Slice | Size | Content |
|-------|------|---------|
| `[0:242]` | 242 | Vision — 121 rays × 2 values: `[type_norm, proximity]` |
| `[242:258]` | 16 | Body tactile receptors (receptor 0 = forward, going CW) |
| `[258:260]` | 2 | Prong tactile receptors `[left_prong, right_prong]` |
| `[260:263]` | 3 | Internal meters `[life, satiation_norm, valence_norm]` |

All values are `∈ [0, 1]`.

### Vision encoding (per ray)

| Field | Value |
|-------|-------|
| `type_norm` | 0 = nothing · 0.25 = wall · 0.5 = food · 0.75 = danger · 1.0 = other entity |
| `proximity` | `1 − distance/range`; 0 when nothing detected, ~1 when very close |

### Tactile encoding (per receptor)

Signal strengths carry no explicit labels — the agent learns what each value
means through experience:

| Signal | Object |
|--------|--------|
| `0.0`  | no contact |
| `0.3`  | food |
| `0.5`  | other entity |
| `0.7`  | wall |
| `1.0`  | danger (pain / over-stimulation) |

Body receptors are body-relative: receptor 0 = directly forward, 8 = directly
behind.  Prong receptors fire when objects are within or near the prong triangle.

### Internal meters

| Index | Meaning |
|-------|---------|
| `[260]` | `life` — 1.0 = full health, 0.0 = dead → triggers reset |
| `[261]` | `satiation_norm` = `(satiation + 1) / 2` — 0.0 = starving, 1.0 = full |
| `[262]` | `valence_norm`   = `(valence + 1) / 2` — 0.0 = max pain, 1.0 = max pleasure |

---

## Reward signal

`reward` is the raw **valence** value `∈ [-1, 1]` (not normalised):

- `+0.6` spike on eating food
- `+0.1` spike on receiving a pat from the player
- `−0.5` or below when touching danger
- Drifts slowly toward 0 otherwise

For RL, `reward` is the natural intrinsic reward.  You can also compute a
shaping reward from `obs[262]` (valence_norm) directly.

---

## `done` flag

`done = True` is sent on the step **after** a reset (life reached 0).  The
next observation will be the fresh starting state.  Your agent should reset
its hidden state (RNN, ESN, etc.) when `done` is True.

---

## Connecting from another language

The wire protocol is plain binary — no ZeroMQ framing beyond the socket type.

**OBS packet** (total: 5 + 4 + 263×4 = 1061 bytes):

```
offset  size  type    field
0       4     int32   step_count  (game steps since last reset)
4       1     uint8   done        (1 = world just reset)
5       4     float32 reward      (valence ∈ [-1, 1])
9       1052  float32[263]  obs
```

**ACT packet** (12 bytes):

```
offset  size  type    field
0       4     float32 fwd   ∈ [-1, 1]
4       4     float32 turn  ∈ [-1, 1]
8       4     float32 eat   ∈  {0, 1}
```

All values are little-endian.

---

## Using your own render loop (headless)

If you want to run without a display — for training — instantiate `World`
directly and skip `game.py` entirely:

```python
from env import World

world = World(seed=42)

obs   = world.get_ai_observation()   # (263,) float32
reward = world.ai.meters.valence
done   = not world.ai.alive          # normally False right after reset

action = (1.0, 0.0, 0.0)            # your agent's output
world.step(ai_action=action)        # player_action defaults to (0,0,0)
```

The `World.step()` call auto-resets when life reaches 0; there is no separate
`reset()` you need to call.

---

## Ports

| Port | Direction | Socket type |
|------|-----------|-------------|
| 5557 | game → agent | PUSH / PULL |
| 5558 | agent → game | PULL / PUSH |

Override via constructor arguments:

```python
conn   = GameConnector(host="0.0.0.0", obs_port=6000, act_port=6001)
client = AgentConnector(host="192.168.1.10", obs_port=6000, act_port=6001)
```
