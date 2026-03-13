# Embodied AI Experiment

A 2D pygame environment for embodied AI research. A human-controlled player and one or
more AI entities share the same world. Each AI has rich sensory input and intrinsic
reward (valence) as its only learning signal. External brain processes connect over
ZeroMQ — no game code needs to change to swap or add agents.

---

## Files

| File | Purpose |
|------|---------|
| `env.py` | Core world simulation — `World`, entities, physics, meters |
| `game.py` | Pygame display, HUD with player sensors, per-agent status, keyboard input |
| `connector.py` | ZeroMQ connectors: `MultiGameConnector` (game), `AgentConnector` (brain), `MonitorConnector` (read-only) |
| `dofs.py` | DoF schema — named external/internal degrees of freedom + obs/action converters |
| `reservoir.py` | `SingleReservoir` — fixed-weight ESN reservoir (PyTorch, GPU-first) |
| `brain.py` | `EmbodiedBrain` — single reservoir + RO Framework Observer + run loops + CLI |
| `brain_viz.py` | Live pygame visualiser — reservoir heatmap, graph view, rolling signal plots |
| `monitor.py` | Rich terminal sensor monitor (read-only, no body spawned) |
| `INTEGRATION.md` | Full wire protocol reference (packet layout, port table) |

---

## Quick Start

### Single brain

```bash
# Terminal 1 — game window
cd experiments/embodied
python game.py --connect

# Terminal 2 — ESN brain (GPU if available, falls back to CPU)
python brain.py --save brain.npz

# Terminal 3 — sensor monitor (optional, read-only)
python monitor.py
```

### Multiple brains

Each `brain.py` process connects independently and gets its own body spawned in the
world. Brains can be started and stopped at any time.

```bash
# Terminal 1 — game
python game.py --connect

# Terminal 2 — first brain (slot 0, blue body)
python brain.py --save brain_a.npz

# Terminal 3 — second brain (slot 1, orange body)
python brain.py --save brain_b.npz

# Terminal 4 — monitor showing all agents
python monitor.py
```

Bodies are despawned automatically ~3 seconds after a brain disconnects. The slot
ID is recycled so a reconnecting brain gets the same slot and same colour.

### With brain visualiser

```bash
# Terminal 1 — game window
python game.py --connect

# Terminal 2 — brain + visualiser (replaces brain.py)
python brain_viz.py --save brain.npz
```

The visualiser shows:

- **Reservoir heatmap** — single reservoir activity (green=+1, red=−1)
- **Node-edge graph** — spring layout of W_res connections, coloured by weight sign
- **Rolling plots** — fwd, turn, reward, RPE over the last 600 steps
- **Spinning alert** — red banner when sustained turn bias is detected

### Headless (no display)

```bash
python brain.py --headless 18000 --save brain.npz --log-path brain_log.csv
```

Imports `World` directly from `env.py`, bypasses pygame entirely. Useful for
overnight training runs.

### No-reset mode

Normally the world resets when an AI's life reaches 0. With `--no-reset`, the AI
stays alive at life=0 / valence=−1 (maximum pain) and must eat to recover:

```bash
# Connected mode
python game.py --connect --no-reset

# Headless
python brain.py --headless 18000 --no-reset --save brain.npz
```

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

The 1200 × 900 px environment contains:

- **Walls** — hard boundaries
- **Food** (20 items) — eating restores life +0.15, satiation +0.35, spikes valence +0.6; respawns after ~5 s
- **Danger** (10 items) — contact triggers pain (valence ≤ −0.5); sustained contact drains life
- **Player body** — human-controlled; can carry and deliver food (`F`), trigger "pat" (`E`, +0.1 pleasure)
- **AI bodies** — one per connected brain; each has its own sensor inputs, meters, and actions

The player can accelerate learning by feeding AIs and patting them.

---

## Game HUD

The bottom HUD has two panels:

| Panel | Content |
|-------|---------|
| Left (player) | Life / Satiation / Valence bars, vision strip, tactile strip |
| Right (controls) | Key bindings, step / death counter, one status line per connected AI |

Each AI status line shows slot number, life, valence, and satiation in its slot colour.
For full per-agent sensor detail use `monitor.py`.

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

`brain.py` implements a **single-reservoir Echo State Network** connected to the
RO Framework library.  A large fixed reservoir maps the full 263-dim observation
to a rich hidden state; only the readout layer W_out is trained.

```
obs[263]  →  SingleReservoir(N)  →  h[N]
                                  →  W_out (3 × N)
                                  →  (fwd, turn, eat)
```

Default reservoir size N = 4096 (64 MB W_res, well within an RTX 4090's L2 cache).
Larger sizes are selectable with `--res-size`:

| Preset       | N     | W_res  | Notes                                |
|--------------|-------|--------|--------------------------------------|
| `RES_TINY`   | 512   | 1 MB   | fast unit tests / CPU                |
| `RES_SMALL`  | 1024  | 4 MB   | quick experiments                    |
| `RES_MEDIUM` | 2187  | 18 MB  | 3^7, matches old hierarchy           |
| `RES_LARGE`  | 4096  | 64 MB  | **default** — RTX 4090               |
| `RES_XL`     | 8192  | 256 MB | comfortable on 4090                  |
| `RES_XXL`    | 16384 | 1 GB   | 4090 has 24 GB, fine for experiments |

### RO Framework integration

`brain.py` uses the library's Observer and KnowledgeTracker to turn the brain into a
named, DoF-typed observer of its own world:

- **External DoFs** — what the brain perceives: `food_max_proximity`, `danger_max_proximity`,
  `life`, `satiation_norm`, `valence_norm` (defined in `dofs.py`)
- **Internal DoFs** — what the brain produces: `fwd_output`, `turn_output`, `eat_output`
- **K(d_ext) = (ρ, ε, σ, C)** — knowledge assessment printed every `--log-every` steps.
  For example, `K(food_max_proximity): ρ=0.45 [weak]` means the brain is beginning to
  correlate food visibility with forward movement.

```
step     300  reward +0.341  valence_pred +0.092  |W_out| 34.65  fwd:+0.77  turn:+0.68  eat:242  ep:0
  K(food_max_proximity):   ρ=0.454  [weak]  ε=0.579  C=0.480
  K(danger_max_proximity): ρ=0.217  [uncertain]  ε=0.808  C=0.525
  K(life):                 ρ=0.000  [weak]  ε=0.000  C=0.000
  K(satiation_norm):       ρ=0.758  [false]  ε=0.770  C=0.000
  K(valence_norm):         ρ=0.747  [false]  ε=0.663  C=0.066
```

### Reservoir dynamics

Each reservoir step uses a leaky integrator:

```
noise  = N(0, noise_scale)
pre    = tanh(W_in @ x  +  W_res @ h  +  bias  +  noise)
h_new  = (1 − α) · h  +  α · pre
```

`α = 1.0` (default) is the standard ESN equation. `W_res` is rescaled to the
configured spectral radius (0.99 — edge of chaos). `bias_scale = 0.0` is mandatory:
any nonzero bias propagates through W_out and locks the motor output into a
persistent turn direction (spinning attractor).

### Online learning — RPE-gated eligibility traces

No pretraining. W_out updates on every step using a simple actor-critic rule:

```
rpe             = reward − valence_pred          # reward prediction error
valence_pred   += CRITIC_LR × rpe               # moving baseline (persists across episodes)
trace           = TRACE_DECAY × trace  +  outer(action, h)
W_out          += LEARN_LR × rpe × trace
W_out          ×= (1 − WEIGHT_DECAY)             # L2 regularisation
```

`reward` is the game's raw valence ∈ [−1, 1]. The eligibility trace gives a ~100-step
(~1.7 s at 60 fps) credit-assignment window — long enough to connect "food visible
ahead" with "food eaten a moment later".

### Hyperparameters

| Constant | Default | Meaning |
|----------|---------|---------|
| `RESERVOIR_SIZE` | 4096 | Number of reservoir neurons |
| `SPECTRAL_RADIUS` | 0.99 | Echo length; close to 1.0 = long memory |
| `NOISE_SCALE` | 0.9 | Background noise — keeps dead neurons active, prevents attractor lock-in |
| `ALPHA` | 1.0 | Leaky integrator rate; 1.0 = standard ESN |
| `BIAS_SCALE` | 0.0 | Fixed reservoir bias — must stay 0 (see above) |
| `EXPLORE_NOISE` | 0.9 | Std of Gaussian noise on fwd/turn before tanh |
| `EAT_THRESHOLD` | 0.0 | Raw eat output threshold for triggering eat action |
| `LEARN_LR` | 1e-4 | W_out learning rate |
| `CRITIC_LR` | 1e-3 | Valence prediction EMA rate |
| `TRACE_DECAY` | 0.99 | Eligibility trace decay (~100-step credit window at 60fps) |
| `WEIGHT_DECAY` | 1e-5 | L2 regularisation on W_out per step |
| `LOG_CAPACITY` | 5000 | Observer log depth (~5 min at 60fps; enough for stable K) |
| `ASSESS_EVERY` | 300 | K assessment interval (matched to `--log-every` by default) |

---

## Brain CLI

```
python brain.py [options]

  --device cuda|cpu        Compute device (default: cuda)
  --res-size N             Reservoir size (default: 4096)
  --action-feedback        Append previous (fwd,turn,eat) to obs → input_dim = 266
  --carrier                [Phase 2 stub] Carrier wave scaffold — no-op in Phase 1
  --assess-every N         K assessment interval in steps (default: 300)
  --no-learn               Freeze W_out (inference only)
  --no-reset               Keep AI alive at life=0/valence=-1 on death (headless only)
  --save PATH              Save W_out + critic state periodically and on exit
  --load PATH              Load previously saved .npz before starting
  --headless N             Run N steps against World directly (no display)
  --seed N                 Reservoir init seed (default 42)
  --log-every N            Print status every N steps (default 300)
  --save-every N           Save to disk every N steps (default 3600)
  --log-path PATH          Append one CSV row per log interval to PATH
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

---

## Game CLI

```
python game.py [options]

  --connect      Enable the multi-agent ZeroMQ connector (required for brains)
  --no-reset     Keep AIs alive at life=0/valence=-1 on death instead of respawning
```

---

## Tips for Training

**Learning is slow at first** — the brain starts with near-zero W_out and relies
entirely on exploration noise. At 60 fps, 3600 steps ≈ 1 minute.  Meaningful
behavioural change typically appears after tens of thousands of steps.

**Feed and pat the AI** — each food delivery gives a +0.6 reward spike, each pat
+0.1. These create strong positive RPEs. Dragging the AI toward food is the fastest
way to teach food-seeking.

**Try different seeds** — `--seed` changes the reservoir init, which changes the
resting motor bias. Some seeds explore more naturally than others.

**Compare multiple seeds simultaneously** — run several `brain.py` processes at once
with different seeds, all connected to the same game. Watch which seeds develop
food-seeking behaviour first.

**Headless pretraining** — run headless overnight, then connect to the live game:

```bash
python brain.py --headless 360000 --save brain.npz --log-path brain_log.csv
python brain.py --load brain.npz --save brain.npz --log-path brain_log.csv
```

---

## Writing Your Own Agent

```python
from connector import AgentConnector

client = AgentConnector()
client.connect()   # registers with the game, gets a slot + dedicated ports
obs, reward, done, step = client.recv_obs()

while True:
    fwd, turn, eat = 1.0, 0.0, 0.0   # your agent here
    client.send_action((fwd, turn, eat))
    obs, reward, done, step = client.recv_obs()
```

Or headless, without ZeroMQ:

```python
from env import World

world = World(seed=42)
world.add_agent()              # spawn one AI body
obs = world.get_ai_observation()   # float32 (263,)
world.step(ai_actions=[(1.0, 0.0, 0.0)])
```

---

## Roadmap

The current single-reservoir brain is Phase 1 of a staged experiment plan.
Each phase is a minimal extension of the previous one, designed to test one
hypothesis at a time.

### Phase 1 — Single Reservoir (current)

One large reservoir (N=4096, GPU) maps the full 263-dim observation to a hidden
state. W_out is trained online via RPE-gated eligibility traces. The RO Framework
Observer records (percept, action) pairs; KnowledgeTracker reports K(d_ext) every
`--assess-every` steps, giving a continuous measure of what the brain has learned.

Open questions from this baseline:
- Does `food_max_proximity → fwd_output` knowledge rise reliably over time?
- Does `danger_max_proximity → turn_output` rise? (avoidance)
- Is there a phase transition (weak → strong) similar to grokking?
- How does K evolve across seeds?

---

### Phase 2 — Modality-Specific Carrier Waves

**Hypothesis:** vision (242 dims) drowns out tactile (18 dims) and internal state
(3 dims) in a flat reservoir.  Injecting a slow oscillatory carrier onto each
modality's W_in columns will create frequency-separated sub-populations that the
Goertzel algorithm can track, effectively giving the reservoir modality-specific
"bands" analogous to gamma/theta/delta in cortex.

Implementation:
- `--carrier` flag activates sinusoidal noise modulation per `input_slices`
- `reservoir.py` already accepts `input_slices`, `carrier_freqs`, `carrier_amps`
- `FrequencyTracker` (new file) computes per-neuron Goertzel amplitude online
- Internal DoFs switch from action outputs to `vis_band_power`, `tac_band_power`,
  `val_band_power` (already defined in `dofs.py`)
- K(valence_norm → val_band_power) measures whether the value band is actually
  driven by internal state signals vs. visual leakage

Carrier frequency suggestions (at 60 fps):
| Band | Freq (cycles/step) | Approx rate | Modality |
|------|--------------------|-------------|----------|
| fast | 0.20 | 12 Hz | vision |
| mid  | 0.07 | 4 Hz | tactile |
| slow | 0.02 | 1 Hz | internal state |

---

### Phase 3 — Squeezed Callosum (Two Hemispheres, Shared Bottleneck)

**Hypothesis:** splitting the reservoir into two pools with a narrow bottleneck
connection (the "callosum") forces the two hemispheres to develop complementary
representations, similar to left/right specialisation in biological brains.

Architecture:
```
obs[263]  →  [left_res, N/2]  ──┐
                                  ├─ callosum (k×k, k << N/2) ─┐
obs[263]  →  [right_res, N/2] ──┘                              │
                                                                 ↓
                                                          W_out (3 × N)
```

The callosum is a small fixed random matrix; information passing through it is
compressed and noisy, encouraging functional specialisation. K(d_ext) on each
hemisphere separately will reveal whether they diverge (left = spatial/vision,
right = valence/state) or stay redundant.

---

### Phase 4 — Recurrent Callosum (Adaptive Bottleneck)

**Hypothesis:** if the callosum itself is a small trained reservoir (not just a
fixed random matrix), it can learn to selectively gate inter-hemisphere
communication based on reward history — a structural model of attention.

The callosum reservoir weight is the only additional trainable parameter beyond
W_out. It is updated by the same RPE-gated trace rule, but gated by the magnitude
of the cross-hemisphere mismatch (Δh_left − h_right).

K(satiation_norm → callosum_state) and K(valence_norm → callosum_state) would
measure whether the callosum preferentially routes valence-relevant information.

---

### Phase 5 — Reservoir Self-Model (Structural Consciousness)

**Hypothesis:** a small "meta-reservoir" that receives the hidden state h as its
input (instead of raw obs) constitutes a structural self-model in the RO sense:
it is an observer whose external DoFs are the primary reservoir's internal state.
K(h → h_meta) measures how well the meta-reservoir tracks the main reservoir.

This maps directly onto `ConsciousnessEvaluator.recursive_depth()` in the library:
the depth increases from 1 to 2 once the meta-reservoir achieves non-trivial K.

---

### Phase 6 — Toward OCA (Organic Cognitive Architecture)

Full multi-reservoir architecture with functional specialisation enforced by the
connectivity topology, not by the training signal:

```
[sensory_res]  →  [valence_res]  →  [motor_res]
                       ↓
               [self_model_res]  ←  h_motor
```

This mirrors the architecture in `docs/organic_cognitive_architecture_oca.md`.
The `brains/` directory will hold one subdirectory per architecture variant for
systematic comparison across phases.

---

## Wire Protocol Summary

| Port | Direction | Content |
|------|-----------|---------|
| 5555 | game → monitor | PUB broadcast — all agents' obs each step (read-only, no body) |
| 5556 | brain → game | REQ/REP registration — brain sends `b"R"`, game replies with slot+ports |
| 5557 + 2N | game → brain N | PUSH/PULL obs packet (1061 bytes): step, done, reward, obs[263] |
| 5558 + 2N | brain N → game | PUSH/PULL act packet (12 bytes): fwd, turn, eat |

Slot 0 uses ports 5557/5558, slot 1 uses 5559/5560, and so on.
Full packet layout: see `INTEGRATION.md`.
