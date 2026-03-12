# Embodied AI Experiment

A 2D pygame environment for embodied AI research. A human-controlled player and an
AI entity share the same world. The AI has rich sensory input and intrinsic reward
(valence) as its only learning signal. An external brain process connects over
ZeroMQ — no game code needs to change to swap the agent.

---

## Files

| File | Purpose |
|------|---------|
| `env.py` | Core world simulation — `World`, entities, physics, meters |
| `game.py` | Pygame display, HUD with live sensor strips, keyboard input |
| `connector.py` | ZeroMQ `GameConnector` (game-side) and `AgentConnector` (brain-side) |
| `brain.py` | Multi-reservoir ESN brain — numpy/CPU, default reservoir sizes |
| `brain_gpu.py` | GPU-accelerated brain — PyTorch CUDA, 4× larger reservoirs (V1=1024, central=2048, motor=1024) |
| `brain_viz.py` | Live pygame visualiser — reservoir heatmaps, graph view, rolling signal plots |
| `monitor.py` | Rich terminal sensor monitor (read-only, second terminal) |
| `INTEGRATION.md` | Full wire protocol reference (packet layout, port table) |

---

## Quick Start

Three terminals:

```bash
# Terminal 1 — game window
cd experiments/embodied
python game.py --connect

# Terminal 2 — ESN brain
python brain.py --save brain.npz

# Terminal 3 — sensor monitor (optional)
python monitor.py
```

The brain connects automatically and starts learning. No pretraining needed.

### With brain visualiser

```bash
# Terminal 1 — game window
python game.py --connect

# Terminal 2 — brain + visualiser (replaces brain.py)
python brain_viz.py --save brain.npz
```

The visualiser window shows:

- **Reservoir heatmaps** — each of the 5 reservoirs as a row of neurons (green=active, red=inhibited)
- **Motor PCA 2-D** — trajectory of motor reservoir state projected to 2D; a loop = spinning attractor, scattered = exploring
- **Rolling plots** — fwd, turn, reward, RPE over the last 400 steps
- **Spinning alert** — red banner when sustained turn bias is detected

### Headless (no display)

```bash
python brain.py --headless 18000 --save brain.npz --log-path brain_log.csv
```

Imports `World` directly from `env.py`, bypasses pygame entirely. Useful for
overnight training runs.

---

## Installation

```bash
# Connector + monitor only (no pygame)
pip install -e ".[embodied]"

# Full game window
pip install -e ".[embodied-game]"
```

---

## The World

The 2D environment contains:

- **Walls** — boundary and interior obstacles
- **Food** — eating increases satiation, triggers a +0.6 pleasure spike, heals
- **Danger** — contact causes pain (−0.5+ valence), sustained contact drains life
- **Player body** — human-controlled; can carry and deliver food, trigger "pat" (+0.1 pleasure)
- **AI body** — sensor-driven; life/satiation/valence meters; dies and resets at life=0

The player can accelerate learning by feeding the AI and patting it.

---

## AI Sensory System

The AI's observation is a `float32[263]` vector every step (~60 fps):

```
obs[0:242]    — Vision:   121 rays × (type_norm, proximity)
obs[242:258]  — Tactile:  16 body receptors (receptor 0 = forward, CW)
obs[258:260]  — Tactile:  2 prong receptors
obs[260:263]  — Internal: [life, satiation_norm, valence_norm]
```

All values ∈ [0, 1]. See `INTEGRATION.md` for encoding details.

---

## Brain Architecture

`brain.py` implements a **multi-reservoir Echo State Network** — five fixed random
reservoirs wired in a sensory hierarchy, with a single trained readout layer.

```
vision (242)  →  [V1_res,       256]  ─┐
tactile (18)  →  [tac_res,      128]  ─┤  concat(448)
state (3)     →  [val_res,       64]  ─┘
                                          → [central_res,  512]
                                          → [motor_res,    256]
                                          → W_out (3 × 256)
                                          → (fwd, turn, eat)
```

### Why this architecture?

| Property | Implementation |
|----------|---------------|
| Fixed reservoir weights | Spectral radius ≈ 0.99 (edge of chaos), never updated |
| Fixed bottleneck connections | Random projections between reservoirs, never updated |
| Only W_out trains | 3 × 256 = 768 parameters — entire learning budget |
| Background noise | Per-reservoir Gaussian noise keeps the system active |
| Memory | Fading echoes in recurrent state — natural short-term memory |

The architecture maps onto the Organic Cognitive Architecture (OCA) in
`docs/organic_cognitive_architecture_oca.md`:
- `val_res` ≈ amygdala / valence system
- `central_res` ≈ association cortex
- `motor_res` ≈ motor cortex / action system

### Reservoir dynamics

Each reservoir step uses a leaky integrator:

```
pre   = tanh(W_in @ x  +  W_res @ h  +  bias  +  noise)
h_new = (1 − α) · h  +  α · pre
```

`α = 1.0` (the default) recovers the standard ESN equation. Lower values slow
the reservoir's response, increasing its effective time constant — useful for
experimenting with different integration speeds per layer (e.g. slow central,
fast sensory). `W_res` is scaled to the configured spectral radius; `bias` is a
small fixed random vector that gives each seed a distinct "personality"
(consistent turn preference, activity level, etc.).

### Online learning — RPE-gated eligibility traces

No pretraining. W_out updates on every step using a simple actor-critic rule:

```
rpe             = reward - valence_pred          # reward prediction error
valence_pred   += CRITIC_LR * rpe               # moving baseline (persists across episodes)
trace           = TRACE_DECAY * trace  +  outer(action, h_motor)
W_out          += LEARN_LR * rpe * trace
W_out          *= (1 - WEIGHT_DECAY)             # L2 regularisation
```

`reward` is the game's raw valence ∈ [−1, 1] (see `INTEGRATION.md §Reward`).
`action` is the post-tanh output `(fwd, turn, eat)` — bounded ∈ [−1, 1] —
which keeps the trace and W_out numerically stable.
The eligibility trace links past motor activations to future rewards with a
0.99 decay, giving a ~100-step (~1.7s) credit-assignment window at 60 fps —
long enough to connect "food visible ahead" with "food eaten a moment later".

### Hyperparameters

All constants are at the top of `brain.py` and overridable via `--config` or the
`config={}` argument to `EmbodiedBrain`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `SPECTRAL_RADIUS` | 0.90 | Reservoir echo length (~10 steps). Lower than the classic 0.99 "edge of chaos" — at 0.99 the reservoir echoes circular motion for 100 steps, locking the agent into spinning attractors. Credit assignment window is set by `TRACE_DECAY` (independent of ρ). |
| `VAL_SPECTRAL_RADIUS` | 0.85 | Value reservoir — faster reset, more reactive to current internal state |
| `V1_SIZE` | 256 | Vision reservoir neurons |
| `TAC_SIZE` | 128 | Tactile reservoir neurons |
| `VAL_SIZE` | 64 | Value reservoir neurons |
| `CENTRAL_SIZE` | 512 | Central integration reservoir |
| `MOTOR_SIZE` | 256 | Motor reservoir |
| `V1_ALPHA` … `MOTOR_ALPHA` | 1.0 | Leaky integrator α per reservoir — 1.0 = standard ESN; lower = slower integration |
| `V1_BIAS_SCALE` … `MOTOR_BIAS_SCALE` | 0.0 | All reservoir biases zeroed — bias on any upstream reservoir propagates into motor output and locks the agent into persistent spinning; noise (ρ=0.99, noise=0.01) keeps reservoirs active without bias |
| `V1_NOISE` … `MOTOR_NOISE` | 0.01 / 0.005 | Per-reservoir background noise scale |
| `EXPLORE_NOISE` | 0.2 | Gaussian std added to fwd/turn before tanh |
| `LEARN_LR` | 1e-4 | W_out learning rate |
| `CRITIC_LR` | 1e-3 | Valence prediction EMA rate |
| `TRACE_DECAY` | 0.99 | Eligibility trace decay (~100-step / ~1.7s window at 60fps) |
| `WEIGHT_DECAY` | 1e-5 | L2 regularisation on W_out per step |

Sensor counts (`N_RAYS`, `N_TOUCH_BODY`, `N_TOUCH_PRONGS`, `STATE_SIZE`) are also
constants — all obs slice indices are derived from them automatically.

---

## Brain CLI

```
python brain.py [options]

  --save PATH      Save W_out + critic state periodically and on exit
  --load PATH      Load previously saved .npz before starting
  --no-learn       Inference only — W_out frozen
  --headless N     Run N steps against World directly (no display)
  --seed N         Reservoir init seed (default 42)
  --log-every N    Print status every N steps (default 300)
  --save-every N   Save to disk every N steps (default 3600 ≈ 1 min at 60 fps)
  --log-path PATH  Append one CSV row per log interval to PATH
```

### Save / load

```bash
# Start a new run
python brain.py --save brain.npz

# Resume from checkpoint
python brain.py --load brain.npz --save brain.npz

# Save on demand without interrupting a running process
kill -USR1 $(pgrep -f "brain.py")
```

Only `W_out`, `b_out`, and `valence_pred` are saved. Reservoir weights are fixed
and reproducible from `--seed`, so they don't need saving.

### Log output

```
step     300  reward +0.031  valence_pred +0.012  |W_out| 0.031  fwd:+0.73  turn:+0.05  eat:1  ep:0
step     600  reward +0.123  valence_pred +0.089  |W_out| 0.034  fwd:+0.68  turn:-0.12  eat:3  ep:1
```

`|W_out|` is the Frobenius norm of the readout weights — initialises around
0.03 for the default 3×256 size, then oscillates: spikes up on positive RPE
events, pulled back down by weight decay between them. `ep` counts episode
resets (life → 0) in the interval.

`fwd` and `turn` are the mean action values over the interval (each ∈ [−1, 1]).
They are the primary diagnostic for spinning attractors:

| Pattern               | Meaning                               |
| --------------------- | ------------------------------------- |
| `turn:+0.9`           | stuck in a right circle               |
| `turn:-0.9`           | stuck in a left circle                |
| `fwd:+0.1`            | spinning in place, barely translating |
| `fwd:+0.6  turn:+0.1` | healthy forward exploration           |

### Persistent CSV log

```bash
python brain.py --save brain.npz --log-path brain_log.csv
```

Appends one row per log interval to a CSV file:

```
step,mean_reward,valence_pred,w_norm,eat_count,episodes,mean_fwd,mean_turn
300,-0.0012,-0.0010,0.0314,1,0,+0.6821,+0.0312
600,+0.0312,0.0089,0.0341,3,1,+0.7103,-0.0891
```

The file is created if it doesn't exist and appended to on restart, so the
full training history accumulates across runs. At the default `--log-every 300`
and 60 fps, an overnight 8-hour run produces ≈ 5 700 rows ≈ 350 KB.

---

## Tips for Training

**Learning is slow at first** — the brain starts with near-zero W_out and
relies entirely on exploration noise. At 60 fps, 3600 steps ≈ 1 minute.
Expect meaningful behavioural change after tens of thousands of steps.

**Feed and pat the AI** — each food delivery gives a +0.6 reward spike; each
pat gives +0.1. These create strong positive RPEs that drive learning. Dragging
the AI toward food is the fastest way to teach food-seeking behaviour.

**Try different seeds** — `--seed` changes the reservoir init, which changes
the resting motor bias (turn preference, activity level). Some seeds explore
more naturally than others.

**Headless pretraining** — run headless overnight to accumulate steps, then
load the weights and connect to the live game:

```bash
python brain.py --headless 360000 --save brain.npz --log-path brain_log.csv
python brain.py --load brain.npz --save brain.npz --log-path brain_log.csv
```

The CSV log appends across restarts, giving you a continuous training history.

**Analysing progress** — load the CSV in Python or any spreadsheet tool:

```python
import pandas as pd
df = pd.read_csv("brain_log.csv")
df["mean_reward"].plot()          # valence trend
df["w_norm"].plot()               # weight growth
df["episodes"].cumsum().plot()    # total deaths over time
```

---

## Wire Protocol Summary

The game (`game.py --connect`) and brain (`brain.py`) communicate over ZeroMQ:

| Port | Direction | Content |
|------|-----------|---------|
| 5557 | game → brain | obs packet (1061 bytes): step, done, reward, obs[263] |
| 5558 | brain → game | act packet (12 bytes): fwd, turn, eat |

The brain can lag or disconnect freely — the game uses the last received action.
Full packet layout: see `INTEGRATION.md`.

---

## Writing Your Own Agent

```python
from connector import AgentConnector
import numpy as np

client = AgentConnector()
client.connect()
obs, reward, done, step = client.recv_obs()

while True:
    # --- your agent logic ---
    fwd, turn, eat = 1.0, 0.0, 0.0

    client.send_action((fwd, turn, eat))
    obs, reward, done, step = client.recv_obs()
```

Or headless, without ZeroMQ:

```python
from env import World

world = World(seed=42)
obs = world.get_ai_observation()   # float32 (263,)
world.step(ai_action=(1.0, 0.0, 0.0))
```
