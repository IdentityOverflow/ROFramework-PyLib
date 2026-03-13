# Embodied Environment — Integration Guide

The game supports **multiple simultaneous AI agents** connecting over ZeroMQ.  The
human player keeps full keyboard control; AI slots are wired through the connector.

---

## Architecture

```
┌──────────────────────────────────────┐        ┌─────────────────────────────┐
│          game.py  (60 fps)           │        │     your agent process       │
│                                      │        │                             │
│  MultiGameConnector                  │        │  AgentConnector             │
│    REP   :5556  ◀── register ──────────────── │    REQ   :5556              │
│    PUB   :5555  ─── monitor ──────────────── │    SUB   :5555  (optional)  │
│    PUSH  :5557  ─── obs slot 0 ───────────── │    PULL  :5557              │
│    PULL  :5558  ◀── act slot 0 ───────────── │    PUSH  :5558              │
│    PUSH  :5559  ─── obs slot 1 ─────────────   second agent (PULL :5559)   │
│    PULL  :5560  ◀── act slot 1 ─────────────   second agent (PUSH :5560)   │
│    ...                               │        └─────────────────────────────┘
└──────────────────────────────────────┘
```

Each agent registers on port 5556, receives a **slot ID**, and is then assigned a
dedicated PUSH/PULL pair.  Slot N uses ports `5557 + 2N` (obs) and `5558 + 2N` (act).

The game publishes one observation packet per active agent per step (~60 fps) without
waiting.  Your agent can consume and respond at any rate — the game uses the most
recent action it has received, or holds the previous one if nothing new has arrived.

---

## Starting the game with the connector enabled

```bash
cd experiments/embodied
python game.py --connect
```

The terminal will print the ports once the sockets are bound:

```
[connector] multi-agent connector ready  reg=5556  monitor=5555
```

The game window opens normally.  Without `--connect`, behaviour is identical to
before — no ZeroMQ overhead.

---

## Connecting from Python

`AgentConnector` handles registration, slot assignment, and per-slot port binding
automatically:

```python
from connector import AgentConnector
import numpy as np

client = AgentConnector()   # defaults: host=127.0.0.1, reg_port=5556
client.connect()            # registers, receives slot ID, opens obs/act sockets

print(f"registered as slot {client.slot_id}")   # 0, 1, 2, ...

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

Multiple agents simply connect in separate processes — each receives its own slot and
its own obs/act port pair.  There is no coordination required between agent processes.

---

## Registration protocol

When `client.connect()` is called, a single REQ/REP exchange happens:

**REQ → game** (1 byte):  `b"\x01"`  (register request)

**REP ← game** (2 bytes):

```
offset  size  type   field
0       1     uint8  slot_id   (0, 1, 2, ...)
1       1     uint8  reserved  (always 0)
```

The slot ID determines which port pair to use:

- obs port: `5557 + 2 * slot_id`
- act port: `5558 + 2 * slot_id`

---

## Disconnect and despawn

If an agent process exits or stops sending/receiving for ~3 seconds (180 steps at
60 fps), the game detects the timeout and **automatically despawns** that agent's body.
The slot ID is freed and will be reused for the next agent that registers.

No explicit deregistration message is needed — simply closing the process is enough.

---

## Monitor stream (optional)

Port 5555 broadcasts a compact status packet for all active agents every step.
Connect a `MonitorConnector` to receive it — no registration, no body spawned:

```python
from connector import MonitorConnector

mon = MonitorConnector()   # host=127.0.0.1, port=5555
mon.connect()

while True:
    packets = mon.recv_all()   # list of dicts, one per active agent
    for p in packets:
        print(p["slot_id"], p["obs"], p["reward"], p["done"])
```

The monitor client is read-only and does not affect the simulation.  The included
`monitor.py` script renders a live Rich dashboard using this stream.

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

## Wire protocol

All values are little-endian.

**OBS packet** (total: 5 + 4 + 263×4 = 1061 bytes):

```
offset  size  type          field
0       4     int32         step_count  (game steps since last reset)
4       1     uint8         done        (1 = world just reset)
5       4     float32       reward      (valence ∈ [-1, 1])
9       1052  float32[263]  obs
```

**ACT packet** (12 bytes):

```
offset  size  type     field
0       4     float32  fwd   ∈ [-1, 1]
4       4     float32  turn  ∈ [-1, 1]
8       4     float32  eat   ∈  {0, 1}
```

---

## Using your own render loop (headless)

If you want to run without a display — for training — instantiate `World`
directly and skip `game.py` entirely:

```python
from env import World

world = World(seed=42)
world.add_agent()           # spawns the first AI entity

obs    = world.get_observation(slot=0)   # (263,) float32
reward = world.agents[0].meters.valence
done   = not world.agents[0].alive       # normally False right after reset

action = (1.0, 0.0, 0.0)               # your agent's output
world.step(ai_actions=[(0, action)])    # list of (slot, action) tuples
```

The `World.step()` call auto-resets when life reaches 0; there is no separate
`reset()` you need to call.

---

## Ports

| Port | Direction | Socket type | Purpose |
|------|-----------|-------------|---------|
| 5555 | game → all | PUB / SUB | Monitor broadcast (all agents) |
| 5556 | agent → game | REQ / REP | Registration |
| 5557 + 2N | game → agent N | PUSH / PULL | Observations for slot N |
| 5558 + 2N | agent N → game | PULL / PUSH | Actions from slot N |

Override base ports via constructor arguments:

```python
conn   = MultiGameConnector(host="0.0.0.0", base_obs_port=6000, base_act_port=6001,
                             reg_port=5900, monitor_port=5899)
client = AgentConnector(host="192.168.1.10", reg_port=5900)
```
