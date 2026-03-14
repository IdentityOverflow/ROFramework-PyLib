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
[connector] registration port 5556  monitor port 5555  (obs base: 5557, act base: 5558)
```

The game window opens normally.  Without `--connect`, behaviour is identical —
no ZeroMQ overhead.

---

## Connecting from Python

`AgentConnector` handles registration, slot assignment, and per-slot port binding
automatically:

```python
from connector import AgentConnector

client = AgentConnector()             # defaults: host=127.0.0.1
client.connect(name="my-agent")       # registers, receives slot ID, opens obs/act sockets

print(f"registered as slot {client.slot_id}")   # 0, 1, 2, ...

obs, reward, done, step = client.recv_obs()   # wait for first frame

while True:
    # --- your agent logic here ---
    fwd  =  1.0   # ∈ [-1, 1]  forward/reverse
    turn =  0.0   # ∈ [-1, 1]  right/left
    eat  =  0.0   # ∈  {0, 1}  attempt to eat/grab food

    client.send_action((fwd, turn, eat))
    obs, reward, done, step = client.recv_obs()   # blocks until next frame

client.close()
```

Multiple agents simply connect in separate processes — each receives its own slot and
its own obs/act port pair.  There is no coordination required between agent processes.

The `name` argument is optional.  If provided, it is shown above the body in the game
window, in the HUD status lines, and in the monitor panel headers.

---

## Registration protocol

When `client.connect()` is called, a single REQ/REP exchange happens:

**REQ → game** — UTF-8 bytes of the brain name (up to 63 bytes).  If no name is
provided, `b"R"` is sent (legacy single-byte sentinel, treated as unnamed).

**REP ← game** (5 bytes, little-endian):

```
offset  size  type    field
0       1     uint8   slot_id    (0, 1, 2, ...)
1       2     uint16  obs_port   (5557 + 2 * slot_id)
3       2     uint16  act_port   (5558 + 2 * slot_id)
```

The slot ID determines which port pair to use.  It is recycled from disconnected
brains when possible, so a reconnecting brain gets the same slot and same colour.

---

## Disconnect and despawn

If an agent process exits or stops sending/receiving for ~3 seconds (180 steps at
60 fps), the game detects the timeout and **automatically despawns** that agent's body.
The slot ID is freed and reused for the next brain that registers.

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
    step, agents = mon.recv()   # list of dicts, one per active agent
    for agent in agents:
        print(agent["name"], agent["reward"], agent["obs"][260:263])
```

Each agent dict contains:

| Key | Type | Content |
|-----|------|---------|
| `"name"` | str | Brain name from config, or `"#N"` if unnamed |
| `"reward"` | float | Raw valence ∈ [-1, 1] |
| `"obs"` | ndarray (263,) float32 | Full observation vector |

The monitor client is read-only and does not affect the simulation.  The included
`monitor.py` script renders a live Rich dashboard using this stream.

---

## Monitor broadcast wire format

Each frame published on port 5555:

**Header** (5 bytes):

```
offset  size  type    field
0       1     uint8   n_agents
1       4     int32   step_count
```

**Per-agent block** (repeated `n_agents` times):

```
offset  size  type          field
0       32    bytes         name  (UTF-8, null-padded to 32 bytes)
32      4     float32       reward  (valence ∈ [-1, 1])
36      1052  float32[263]  obs
```

Total per-agent block: 1088 bytes.

---

## Observation packet — `obs` (float32 array, shape `(263,)`)

| Slice | Size | Content |
|-------|------|---------|
| `[0:242]` | 242 | Vision — 121 rays × 2 values: `[type_norm, proximity]` |
| `[242:258]` | 16 | Body tactile receptors (receptor 0 = forward, going CW) |
| `[258:260]` | 2 | Prong tactile receptors `[left_prong, right_prong]` |
| `[260:263]` | 3 | Internal meters `[life, satiation_norm, valence_norm]` |

All values are ∈ [0, 1].

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

`reward` is the raw **valence** value ∈ [-1, 1] (not normalised):

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

## OBS/ACT packet wire format

All values are little-endian.

**OBS packet** (total: 5 + 4 + 263×4 = 1061 bytes):

```
offset  size  type          field
0       4     int32         step_count  (game steps since start)
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

## Headless (no display)

Instantiate `World` directly and skip `game.py` entirely:

```python
from env import World

world = World(seed=42)
world.add_agent()                          # spawn one AI body

obs    = world.get_ai_observation()        # float32 (263,)
reward = float(world.ai.meters.valence)    # ∈ [-1, 1]
done   = not world.ai.alive               # normally False right after reset

action = (1.0, 0.0, 0.0)                  # fwd, turn, eat
world.step(ai_action=action)              # advances one step; auto-resets on death
```

For multiple agents use `ai_actions` (list indexed by slot):

```python
world.add_agent()   # slot 0
world.add_agent()   # slot 1
world.step(ai_actions=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
```

---

## Ports

| Port | Direction | Socket type | Purpose |
|------|-----------|-------------|---------|
| 5555 | game → all | PUB / SUB | Monitor broadcast (all agents, every step) |
| 5556 | agent → game | REQ / REP | Registration — brain sends name, game responds with slot+ports |
| 5557 + 2N | game → agent N | PUSH / PULL | Observations for slot N |
| 5558 + 2N | agent N → game | PUSH / PULL | Actions from slot N |

Override base ports via constructor arguments:

```python
conn   = MultiGameConnector(host="0.0.0.0", register_port=5900, monitor_port=5899)
client = AgentConnector(host="192.168.1.10")
client.connect(name="remote-brain", register_port=5900)
```
