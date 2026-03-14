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
| `brains/configs/` | Per-brain JSON config files (hyperparameters, paths, name) |
| `rulesets/` | World ruleset JSON files (game parameters — food density, meter rates, etc.) |
| `INTEGRATION.md` | Full wire protocol reference (packet layout, port table) |

---

## Quick Start

### Single brain

```bash
# Terminal 1 — game window (default ruleset)
cd experiments/embodied
python game.py --connect

# Terminal 1 — game window with a custom ruleset
python game.py --connect --rules rulesets/default.json

# Terminal 2 — ESN brain using a config file
python brain.py --config brains/configs/bob-16k.json

# Terminal 3 — sensor monitor (optional, read-only)
python monitor.py
```

The brain reads `brain_path` and `log_path` from its config.  If the checkpoint file
already exists it is loaded automatically — no `--load` needed.

### Multiple brains

Each `brain.py` process connects independently and gets its own body spawned in the
world. Brains can be started and stopped at any time.

```bash
# Terminal 1 — game
python game.py --connect

# Terminal 2 — first brain (slot 0, blue body)
python brain.py --config brains/configs/boop-256.json

# Terminal 3 — second brain (slot 1, orange body)
python brain.py --config brains/configs/bob-16k.json

# Terminal 4 — monitor showing all agents
python monitor.py
```

Bodies are despawned automatically ~3 seconds after a brain disconnects. The slot
ID is recycled so a reconnecting brain gets the same slot and same colour.

Names from each config (`"name": "Bob"`) are shown above the agent's body in the game
window, in the HUD status lines, and in the monitor panel headers.

### With brain visualiser

```bash
# Terminal 1 — game window
python game.py --connect

# Terminal 2 — brain + visualiser (replaces brain.py)
python brain_viz.py --config brains/configs/bob-16k.json
```

The visualiser shows:

- **Reservoir heatmap** — single reservoir activity (green=+1, red=−1)
- **Node-edge graph** — spring layout of W_res connections, coloured by weight sign
- **Rolling plots** — fwd, turn, reward, RPE over the last 600 steps
- **Spinning alert** — red banner when sustained turn bias is detected

It accepts the same flags as `brain.py` — `--config`, `--device`, `--no-learn`, `--headless`,
`--no-reset`, `--load`, `--save`.  Auto-load, config save, and world ruleset all work identically.

```bash
# Headless with live visualiser (useful for watching noise patterns without the game)
python brain_viz.py --config brains/configs/bob-16k.json --headless 18000
```

### Headless (no display)

```bash
python brain.py --config brains/configs/bob-16k.json --headless 18000
```

Imports `World` directly from `env.py`, bypasses pygame entirely. Useful for
overnight training runs.  Log path and save path come from the config.

### No-reset mode

Normally the world resets when an AI's life reaches 0. With `--no-reset`, the AI
stays alive at life=0 / valence=−1 (maximum pain) and must eat to recover:

```bash
# Connected mode
python game.py --connect --no-reset

# Headless (no-reset is a brain-side flag in headless mode)
python brain.py --config brains/configs/bob-16k.json --headless 18000 --no-reset
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
- **Food** — eating restores life, satiation, and spikes valence; respawns after a configurable delay
- **Danger** — contact triggers pain (valence ≤ −0.5); sustained contact drains life
- **Player body** — human-controlled; can carry and deliver food (`F`), trigger "pat" (`E`, +0.1 pleasure)
- **AI bodies** — one per connected brain; each has its own sensor inputs, meters, and actions

Default counts and reward values are set in `rulesets/default.json` and can be changed without
retraining — see the **Rulesets** section below.

The player can accelerate learning by feeding AIs and patting them.

---

## Game HUD

The bottom HUD has two panels:

| Panel | Content |
|-------|---------|
| Left (player) | Life / Satiation / Valence bars, vision strip, tactile strip |
| Right (controls) | Key bindings, step / death counter, one status line per connected AI |

Each AI status line shows the brain's name (from config), life, valence, and satiation
in its slot colour.  When multiple agents are connected, each body is labelled with its
name in the game world.  For full per-agent sensor detail use `monitor.py`.

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
Size is set in the config file:

| Config name  | N     | W_res  | Notes                                |
|--------------|-------|--------|--------------------------------------|
| `boop-256`   | 256   | 0.25 MB | CPU toy / fast iteration            |
| `alice-729`  | 729   | 2 MB   | small CPU experiment                 |
| `zuzu-2k`    | 2187  | 18 MB  | 3^7, medium                          |
| `bob-16k`    | 16384 | 1 GB   | large GPU (RTX 4090 comfortable)     |
| `xl-8k`      | 8192  | 256 MB | mid GPU                              |

### RO Framework integration

`brain.py` uses the library's Observer and KnowledgeTracker to turn the brain into a
named, DoF-typed observer of its own world:

- **External DoFs** — what the brain perceives: `food_max_proximity`, `danger_max_proximity`,
  `life`, `satiation_norm`, `valence_norm` (defined in `dofs.py`)
- **Internal DoFs** — what the brain produces: `fwd_output`, `turn_output`, `eat_output`
- **K(d_ext) = (ρ, ε, σ, C)** — knowledge assessment printed every `log_every` steps.
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
pre    = tanh(W_in @ x  +  W_res @ h  +  noise)
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
valence_pred   += critic_lr × rpe               # moving baseline (persists across episodes)
trace           = trace_decay × trace  +  outer(action, h)
W_out          += learn_lr × rpe × trace
W_out          ×= (1 − weight_decay)             # L2 regularisation
```

`reward` is the game's raw valence ∈ [−1, 1]. The eligibility trace gives a ~100-step
(~1.7 s at 60 fps) credit-assignment window — long enough to connect "food visible
ahead" with "food eaten a moment later".

---

## Config Files

All brain hyperparameters live in a JSON config file.  A documented template is at
`brains/configs/config-template.json`.  Ready-to-use configs are provided for each
reservoir size in `brains/configs/`.

```json
{
  "name":         "Bob",
  "brain_path":   "brains/Bob-16k.npz",
  "log_path":     "brains/logs/Bob-16k.log",
  "world_config": "rulesets/default.json",
  "res_size":     16384,
  "device":       "cuda",
  ...
}
```

The `world_config` field records which ruleset the brain was trained on.  In connected
mode the game owns the rules (`--rules` flag) — this field is informational only.  In
headless mode (`--headless N`) brain.py loads it automatically to configure the World.

**Safe to change between runs** (even after a checkpoint exists):
`explore_noise`, `eat_threshold`, `learn_lr`, `critic_lr`, `trace_decay`,
`weight_decay`, `assess_every`, `log_capacity`, `name`, `brain_path`, `log_path`,
`world_config`, `device`.

**Do not change after a checkpoint exists** — these define the reservoir structure
and changing them invalidates the saved W_out:
`res_size`, `seed`, `spectral_radius`, `alpha`, `action_feedback`.

| Config key | Default | Meaning |
|------------|---------|---------|
| `res_size` | 4096 | Number of reservoir neurons |
| `spectral_radius` | 0.99 | Echo length; close to 1.0 = long memory |
| `noise_scale` | 90.0 | Background noise injected into reservoir each step |
| `alpha` | 1.0 | Leaky integrator rate; 1.0 = standard ESN |
| `explore_noise` | 90.0 | Std of Gaussian noise added to fwd/turn before tanh |
| `eat_threshold` | 0.0 | Raw eat output threshold for triggering eat action |
| `learn_lr` | 1e-4 | W_out learning rate |
| `critic_lr` | 1e-3 | Valence prediction EMA rate |
| `trace_decay` | 0.99 | Eligibility trace decay (~100-step credit window at 60fps) |
| `weight_decay` | 1e-5 | L2 regularisation on W_out per step |
| `assess_every` | 300 | K assessment interval in steps |
| `log_capacity` | 5000 | Observer log depth (~83 s at 60fps) |
| `seed` | 42 | Reservoir weight init seed |
| `action_feedback` | false | Feed previous (fwd,turn,eat) back into reservoir input |
| `device` | "cuda" | Compute device |

---

## Rulesets

World parameters live in a separate JSON file, independent of any brain.  The
annotated template is at `rulesets/ruleset_template.json`; `rulesets/default.json`
contains the clean defaults ready to copy and edit.

```bash
# Run game with a custom ruleset
python game.py --connect --rules rulesets/my-ruleset.json
```

All ruleset parameters are **safe to change at any time** — they do not affect the
observation vector size or the brain checkpoint.  Changing them between runs with the
same brain just changes the environment the brain is placed in.

| Parameter | Default | Effect |
| --------- | ------- | ------ |
| `food_count` | 20 | Number of food items in the world |
| `danger_count` | 10 | Number of danger zones |
| `food_respawn_steps` | 300 | Steps before eaten food reappears (~5 s) |
| `spawn_near_food` | true | Spawn AI adjacent to food (early eating signal) |
| `entity_speed` | 3.0 | px/step at full forward input |
| `entity_turn_speed` | 0.06 | rad/step at full turn (~3.4°) |
| `vision_range` | 450.0 | Max ray-cast distance in pixels |
| `satiation_drain_rate` | 0.0001 | Hunger speed (lower = longer before starvation) |
| `valence_decay_neg` | 0.001 | Recovery rate from negative valence per step |
| `valence_decay_pos` | 0.001 | Fade rate from positive valence per step |
| `hunger_valence_drain` | 0.002 | Pain increase rate when starving |
| `danger_life_delay_steps` | 180 | Contact steps before life starts draining (~3 s) |
| `life_drain_rate` | 0.001 | Life lost per step once drain is active |
| `food_pleasure` | 0.6 | Valence spike on eating |
| `food_satiation_gain` | 0.35 | Satiation restored on eating |
| `food_life_gain` | 0.15 | Life restored on eating |
| `danger_vision_penalty` | 0.002 | Valence penalty per step for danger in forward cone |
| `food_vision_reward` | 0.002 | Valence reward per step for food in forward cone |

---

## Brain CLI

```
python brain.py [options]

  --config PATH        JSON config file (see brains/configs/config-template.json)
  --device cuda|cpu    Override device from config (useful for quick CPU testing)
  --no-learn           Freeze W_out (inference only)
  --no-reset           Keep AI alive at life=0/valence=-1 on death (headless only)
  --headless N         Run N steps against World directly (no display)
  --save PATH          Override brain_path from config for this run
  --load PATH          Force-load weights (skips auto-load from config brain_path)
  --log-path PATH      Override log_path from config for this run
  --log-every N        Print status every N steps (default 300)
  --save-every N       Save to disk every N steps (default 3600)
  --carrier            [Phase 2 stub] Carrier wave scaffold — no-op in Phase 1
```

### Typical workflows

```bash
# New brain — reads brain_path + log_path from config
python brain.py --config brains/configs/bob-16k.json

# Resume — auto-loads checkpoint if brain_path exists in config
python brain.py --config brains/configs/bob-16k.json

# Force CPU (e.g. testing on a machine without a GPU)
python brain.py --config brains/configs/boop-256.json --device cpu

# Headless overnight run
python brain.py --config brains/configs/bob-16k.json --headless 360000

# Inference only (no learning)
python brain.py --config brains/configs/bob-16k.json --no-learn

# Save on demand without interrupting a running process
kill -USR1 $(pgrep -f "brain.py")
```

Only `W_out`, `b_out`, and `valence_pred` are saved. Reservoir weights are fixed
and reproducible from `seed`, so they don't need saving.  The config JSON is always
written alongside the `.npz` checkpoint, so resuming never requires repeating the
config path.

---

## CSV Log

When `log_path` is set (in config or via `--log-path`), one row is appended every
`log_every` steps:

| Column | Content |
|--------|---------|
| `timestamp` | Wall-clock time of the log row |
| `step` | Total steps elapsed |
| `mean_reward` | Mean valence over the last window |
| `valence_pred` | Current critic baseline |
| `w_norm` | \|W_out\| Frobenius norm |
| `eat_count` | Eat actions triggered in window |
| `episodes` | Episode resets in window |
| `mean_fwd`, `mean_turn` | Mean motor outputs |
| `K_*` | Pearson ρ per external DoF (standard K tracker) |
| `K_sat_eat` | K(satiation → eat) — hunger drives eating? |
| `K_food_eat_cond` | K(food_prox → eat \| food visible) — conditional |
| `K_food_eat_lag2` | K(food_prox[t] → eat[t+2]) — delayed response |
| `K_fwd_autocorr` | ρ(fwd[t], fwd[t+1]) — motor persistence / attractor strength |
| `K_turn_autocorr` | ρ(turn[t], turn[t+1]) — spiral vs chaotic? |

Plot all logs with:
```bash
python brains/plot_logs.py
```

---

## Game CLI

```
python game.py [options]

  --connect          Enable the multi-agent ZeroMQ connector (required for brains)
  --no-reset         Keep AIs alive at life=0/valence=-1 on death instead of respawning
  --rules PATH       Load world ruleset JSON (see rulesets/ruleset_template.json)
```

---

## Tips for Training

**Learning is slow at first** — the brain starts with near-zero W_out and relies
entirely on exploration noise. At 60 fps, 3600 steps ≈ 1 minute.  Meaningful
behavioural change typically appears after tens of thousands of steps.

**Feed and pat the AI** — each food delivery gives a +0.6 reward spike, each pat
+0.1. These create strong positive RPEs. Dragging the AI toward food is the fastest
way to teach food-seeking.

**Try different seeds** — `seed` in the config changes the reservoir init, which
changes the resting motor bias. Some seeds explore more naturally than others.

**Compare multiple seeds simultaneously** — run several `brain.py` processes at once
with different configs, all connected to the same game. Watch which ones develop
food-seeking behaviour first.

**Headless pretraining, then connect**:

```bash
# Train overnight headless
python brain.py --config brains/configs/bob-16k.json --headless 360000

# Connect to the live game (auto-loads from brain_path in config)
python game.py --connect
python brain.py --config brains/configs/bob-16k.json
```

**Reduce noise for exploitation** — once a brain has learned a strategy, set
`explore_noise` lower in the config to reduce jitter without restarting.

---

## Writing Your Own Agent

```python
from connector import AgentConnector

client = AgentConnector()
client.connect(name="my-agent")   # registers with the game, gets a slot + dedicated ports
obs, reward, done, step = client.recv_obs()

while True:
    fwd, turn, eat = 1.0, 0.0, 0.0   # your agent here
    client.send_action((fwd, turn, eat))
    obs, reward, done, step = client.recv_obs()
```

Or headless, without ZeroMQ:

```python
import json
from env import World

with open("rulesets/default.json") as f:
    cfg = json.load(f)

world = World(seed=42, cfg=cfg)
world.add_agent()                          # spawn one AI body
obs = world.get_ai_observation()           # float32 (263,)
world.step(ai_action=(1.0, 0.0, 0.0))     # single-agent shorthand
```

---

## Roadmap

The current single-reservoir brain is Phase 1 of a staged experiment plan.
Each phase is a minimal extension of the previous one, designed to test one
hypothesis at a time.

### Phase 1 — Single Reservoir (current)

One large reservoir maps the full 263-dim observation to a hidden state. W_out is
trained online via RPE-gated eligibility traces. The RO Framework Observer records
(percept, action) pairs; KnowledgeTracker reports K(d_ext) every `assess_every` steps,
giving a continuous measure of what the brain has learned.

Open questions from this baseline:
- Does `food_max_proximity → fwd_output` knowledge rise reliably over time?
- Does `danger_max_proximity → turn_output` rise? (avoidance)
- Is there a phase transition (weak → strong) similar to grokking?
- How does K evolve across reservoir sizes?

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
| 5556 | brain → game | REQ/REP registration — brain sends its name, game replies with slot+ports |
| 5557 + 2N | game → brain N | PUSH/PULL obs packet (1061 bytes): step, done, reward, obs[263] |
| 5558 + 2N | brain N → game | PUSH/PULL act packet (12 bytes): fwd, turn, eat |

Slot 0 uses ports 5557/5558, slot 1 uses 5559/5560, and so on.
Full packet layout: see `INTEGRATION.md`.
