# Holographic Wave Memory — Findings

Experiments applying the holography ↔ HRR connection to the wave reservoir
substrate (`experiments/embodied/wim_brain.py`'s spring lattice). See the
holography thread for context; this is experiment **#2a** (wave substrate,
static associative memory).

## #2a — Does the phase the readout throws away buy binding capacity?

**File:** [`01_wave_associative_memory.py`](01_wave_associative_memory.py)

### Setup

The wave reservoir is a 2D spring lattice whose state is `(disp, vel)`. For a
harmonic medium these are the in-phase and quadrature parts of a complex phasor
`z = disp + i·vel` — the medium already holds the full complex wave. But the
standard readout is `W_out @ tanh(disp)`: it keeps `disp` and discards `vel`
(the phase quadrature). It reads the hologram's amplitude and never uses phase.

A vocabulary of items is injected into the lattice as random initial
displacements; after wave propagation, each item's `(disp, vel)` state is its
phasor (a point spanning many standing modes). We store M key→value
associations one-shot in a correlation-matrix / holographic associative memory
`H = Σ vₘ kₘ^H`, recall `r = H k_a`, and decode by nearest value prototype.
Four representations of the same medium state are compared:

| arm | representation | DOF |
|---|---|---|
| disp | amplitude only (current wim readout) | N |
| \|z\| | magnitude, phase-blind | N |
| **z = disp+i·vel** | **full complex phasor (holographic)** | 2N |
| [disp,vel] | concatenated real, DOF-matched control | 2N |

### Result (grid 12×12, N=144, vocab 3000, 6 trials)

Recall accuracy vs. number of stored associations M:

| M | 200 | 300 | 400 | 600 | 800 | 1000 | 1400 |
|---|---|---|---|---|---|---|---|
| disp (amplitude) | 100% | 97% | 90% | 69% | 50% | 35% | 19% |
| \|z\| (phase-blind) | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| **z (holographic)** | 100% | 100% | 99% | **88%** | **73%** | 58% | 36% |
| [disp,vel] (control) | 100% | 100% | 98% | 89% | 72% | 58% | 36% |

Capacity (interpolated M at 50% recall): disp **796**, holographic **1145**,
DOF-control **1138**, phase-blind **0**.

K(d_ext) on holographic one-shot recall (M=40): ρ=0.87, ε=0.002, σ=0.50,
C=0.87 → **strong**.

### Verdict against the success criteria

- **Capacity — PASS, but honest.** The full wave state gives ~1.4× the capacity
  of the amplitude-only readout (and ~2× the *accuracy* in the breakdown region,
  M=600–800). Using `(disp, vel)` clearly beats discarding `vel`.
- **No magic beyond DOF.** Holographic complex `z` ≈ DOF-matched real `[disp,vel]`
  to within noise (1145 vs 1138; 88% vs 89%, 73% vs 72% at every M). For *static*
  linear associative memory, "going complex" buys exactly what doubling the real
  degrees of freedom buys — nothing more. This matches VSA theory (complex random
  vectors ≈ real 2N for linear recall).
- **Phase carries information — PASS, strongly.** The phase-blind `|z|` arm
  collapses to ~0% recall immediately, while phase-bearing `disp` works. The
  relative timing of the excited modes (phase) is essential; magnitude alone is
  nearly useless.
- **One-shot K = strong — PASS.**

### What this means

1. **Actionable for the wave brain:** `wim_brain`'s readout throws away roughly
   half its memory capacity by discarding `vel`. Reading the full `(disp, vel)`
   phasor ~doubles binding capacity for free — a concrete, cheap improvement.
2. **The holographic framing is not yet earned.** In the *static* setting, the
   complex/holographic structure gives no advantage over matched real DOF. If
   there is a genuinely holographic win, it must come from **recall through the
   medium's own dynamics** — re-injecting the reference wave and letting the
   adapted couplings reconstruct the object wave (phase-gated Hebbian) — not from
   static complex vectors. This sharpens the bar for the next experiment.
3. **Phase is genuinely the missing ingredient** an amplitude-only/ESN readout
   discards — the phase-blind collapse makes that falsifiable claim concrete.

### Next: #2b — recall through the dynamics

The static memory here is a stored matrix `H`. The genuinely holographic test is
to put the memory *in the medium*: record key→value interference into adaptive
couplings (phase-gated Hebbian, recording only the `R*O` cross term), then recall
by re-injecting the reference (key) wave and reading the reconstructed object
wave at the output. Success criterion for #2b: dynamics-based reconstruction beats
the DOF-matched static control — i.e. an advantage that is *not* explained by DOF
count alone, which #2a shows the static complex representation does not provide.

---

## #2b — Literal lattice re-propagation: a negative result

**File:** [`02_lattice_repropagation_completion.py`](02_lattice_repropagation_completion.py)

### Setup

A 2D spring lattice with **local (neighbour-only)** couplings is the medium.
Smooth wavefield patterns (band-limited superpositions of low spatial modes —
the only thing local couplings can store) are recorded by writing local gratings
via phase-gated Hebbian `Δc_ij ∝ f_i·f_j`. Recall clamps a *fragment* of a
pattern onto its nodes and lets the lattice physically re-propagate through the
recorded couplings ("a fragment reconstructs the whole"). Five arms, matched
representation, with a **no-memory floor** (decode the raw fragment directly):

1. raw-cue (no memory) — the floor
2. lattice-wave — local, recurrent, phase-bearing (momentum) ← the test
3. lattice-diffusive — local, recurrent, overdamped (phase-blind)
4. local-static — local couplings, one-shot (single diffraction pass)
5. global-static — full N×N outer-product memory, one-shot

### Result (grid 16×16, N=256, 40 smooth patterns k≤7, 80 trials)

Completion accuracy vs. fragment fraction:

| fragment | 20% | 10% | 5% |
|---|---|---|---|
| raw-cue (no memory) | 100% | 99% | 92% |
| lattice-wave (iterated) | 100% | 85% | 48% |
| lattice-diffusive | 100% | 86% | 52% |
| local-static (single pass) | 100% | 99% | 92% |
| global-static (ideal) | 100% | 94% | 78% |

Re-propagation steps sweep (fragment 10%): 1→100%, 3→100%, 10→75%, 100→86%,
300→89% — **more iteration is worse**, single pass is best. Verified robust
across {k≤7 M=60; k≤11 M=120; grid24 k≤11 M=200; k≤15 M=180}: in every regime
**raw-cue ≥ local ≥ global**.

### Verdict — the hypothesis is falsified, structurally

- **No memory arm beats the raw-cue floor.** There is no completion to do: the
  local couplings can only store *smooth* patterns, but smooth patterns are
  redundant enough that a tiny fragment already determines them. The regime where
  completion *would* matter — high-frequency, fragment-ambiguous patterns — is
  exactly what local couplings cannot store. This is a genuine structural tension,
  not a tuning miss.
- **Iterating the re-propagation hurts.** Linear iteration is power-iteration: it
  drifts toward the dominant stored mode and accumulates crosstalk. A hologram
  reconstructs in a *single* diffraction pass; multi-step re-propagation is not
  what a hologram does.
- **Wave ≈ diffusive.** Phase/momentum is irrelevant for linear re-propagation —
  the dynamical analog of #2a's "complex = real DOF."

### What #2a + #2b together establish

On the wave substrate, **neither static complex vectors (#2a) nor literal
multi-step re-propagation (#2b) earns the holographic framing.** The linear
wave-optical picture buys nothing beyond DOF count. A genuine memory advantage
would require **nonlinear attractor recall** (modern Hopfield / attention) — a
recurrent-*neural* mechanism, not a linear wave-optical one. This also sharpens
the Seed branch (#2-Seed): for the Seed's phase to matter for memory, it must act
through nonlinear coherence-gating / attractor dynamics, **not** linear wave
re-propagation. The honest takeaway from #2a still stands and is the actionable
one: `wim_brain` should read the full `(disp, vel)` state rather than discarding
`vel`, doubling capacity — but that is a DOF win, not a holographic one.

---

## #3 — Holo-Reservoir Recorder: the positive result

**File:** [`03_holo_reservoir_recorder.py`](03_holo_reservoir_recorder.py)

The original motivation: a reservoir is bad at long memory because its echo
*fades*. Holography is the trick of *freezing* a fleeting wave into a medium that
does **not** fade. So put them side by side and race them — this tests the part
of holography that genuinely works (permanence + content-addressed recall),
sidestepping the #2b wall (clean completion needs a nonlinear decider).

### Setup

The reservoir is a bank of N oscillators at the **DFT frequencies** — literally a
sliding Fourier transform. Drive it with a sequence of symbols (random
unit-magnitude phasor vectors); it rings and decays. The bank's own phases form a
**clock**: at each step the oscillators sit at a unique combination of angles — a
natural timestamp (the reference wave R). Two memories are built from identical
ingredients, differing in **one** thing:

- **Live echo (fading):** the reservoir's actual state. Symbol of age `a` survives
  only as `λ^a`.
- **Frozen film (permanent):** each step records `symbol ⊙ timestamp` into a medium
  with **no decay**. Symbol survives at full strength regardless of age.

Recall is holographic: multiply by the step's timestamp (shine the reference back),
decode nearest symbol.

### Result (N=512 DFT oscillators, 32-symbol codebook, sequence length 50, λ=0.85)

Recall accuracy vs. item age (steps since seen):

| age | 0 | 4 | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| echo (reservoir) | 100% | 100% | 100% | 13% | 0% | 7% | 0% |
| **film (hologram)** | 100% | 100% | 97% | 93% | 100% | 100% | 93% |

The reservoir echo dies by age ~16; the film holds **~96% flat across all 50**.

Cost — capacity (mean film recall vs. number of symbols stored):

| stored | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| film recall | 100% | 97% | 73% | 39% | 20% | 12% |

### Reading

- **GAIN:** the film recalls age-49 items at 100% where the reservoir echo is long
  dead. Same binding, same reference clock — the *only* difference is that the film
  does not decay. This is holography answering the reservoir memory-decay problem,
  exactly as the original intuition predicted.
- **COST:** with no decay to shield it, the film blurs once you superpose more than
  ~N symbols — a **capacity (crosstalk) limit, not a time limit**. The reservoir
  trades the opposite way: unlimited symbols, but only the last few survive.
- **Right architecture:** not "make the reservoir holographic" but **reservoir
  (fast, fading, nonlinear processing) + holographic film (slow, permanent recall)**
  — two subsystems, like working memory vs. long-term memory. This is the linear
  half of holography; clean recall from a *messy* cue still needs a nonlinear
  decider (#2b), which is the future Seed coherence-gate.

---

## #4 — Nonlinear Recall: the decider, delivered

**File:** [`04_nonlinear_recall.py`](04_nonlinear_recall.py)

#3 left two walls, both artifacts of *linear* recall: the capacity wall
(crosstalk past ~N superposed symbols) and the messy-cue wall (#2b: linear
dynamics cannot "decide"). Both are attacked with the same weapon — nonlinear
decisions fed back into the recall loop.

### A — Capacity: the film is a CDMA channel

The film is literally CDMA: symbols spread by near-orthogonal codes (the clock
phases), superposed in one medium. Linear recall = matched filter — and telecom
solved the matched filter's crosstalk wall decades ago with **successive
interference cancellation** (SIC): decode the most confident position
(top1−top2 margin), subtract its reconstruction from the film, re-decode the
rest against the cleaned residual. Then relax: coordinate-descent sweeps,
re-deciding each position against everything currently explained.

Result (N=512, codebook 32, 6 trials, greedy batch 5%):

| stored | 50 | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|---|---|
| linear (#3) | 97% | 72% | 53% | 42% | 26% | 21% | 13% |
| SIC | 100% | 100% | 97% | 51% | 22% | 16% | 10% |
| SIC+relax | 100% | 100% | **100%** | 27% | 14% | 11% | 8% |

- **The near-perfect region tripled** (~50 → ~150 stored symbols at ≥97%).
  Below the wall, decisions bootstrap: each correct subtraction cleans the
  medium for the next.
- **The wall itself is a phase transition, not a slope.** Between 150 and 200
  stored, everything collapses at once. This is the known SIC load threshold
  from CDMA/statistical physics: below it errors self-correct, above it they
  avalanche. Relaxation makes the signature crisp — it *helps* below the wall
  (97→100% at 150) and *hurts* above it (51→27% at 200), descending to a wrong
  fixed point. Rhymes with #2b's "iterating hurts": iteration is only as good
  as the basin it starts in.
- Batch schedule matters near the wall: greedy 25% batches lose to careful 3-5%
  batches (82% → 98% at 150) — but no schedule crosses the threshold.

### B — Messy cue: episodic recall through decisions

A first design failed structurally, and the failure rhymes with #2b: cueing
with a **masked timestamp** is ill-posed — the clock manifold has one degree of
freedom, so a small fragment of reference phases pins down k with no memory
needed (the fragment "already determines the whole", the same trap that
flattened #2b's smooth patterns). Worse, in a symbol+time resonator loop every
symbol that occurs *anywhere* in the sequence is a stable attractor with full
coherence — the loop forgets the cue. Documented dead end.

The well-posed messy cue is **content**, and the well-posed question is
**episodic**: given a phase-noised version of a stored symbol, the cue alone
can at best re-identify the symbol (codebook prior) — it can never say *when*
it happened or *what came next*. Only the film can. Mechanism: a recognition
scan over the clock manifold with the decider inside the loop — for each
position, read the film, **project onto the codebook** (nonlinear cleanup),
score the clean candidate against the noisy cue; best match gives k, read k+1
for the episodic answer.

Result (N=512, codebook 64, 50 unique stored symbols, cue phase noise σ):

| cue noise σ | 0.5 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|
| identity, direct (floor, no film) | 100% | 100% | 88% | 6% | 4% | 4% |
| when — k found (film only) | 86% | 86% | 82% | 6% | 6% | 4% |
| **what came next (film only, chance 2%)** | **82%** | **76%** | **78%** | 6% | 6% | 4% |

- **Episodic retrieval works**: "what came next" is unanswerable from the cue
  (chance = 2%) and the film delivers it at ~80% from cues noisy up to σ=2.0
  (cue-to-symbol correlation e^{−σ²/2} ≈ 0.14 — the cue is barely recognizable).
- **Noise tolerance tracks the identity floor**: both collapse together at
  σ≈2.5. The bottleneck is cue recognition, not film readout — the film's
  internal decode at 50 stored is ~97% clean.
- This is completion-through-**decisions** where #2b proved
  completion-through-**dynamics** impossible: the win appears exactly when
  recall passes through nonlinear projections onto known structure (codebook
  attractors, 1-dof clock manifold). The medium alone earns nothing (#2a/#2b);
  the medium plus a decider earns episodic memory.

### What #3 + #4 establish together

The architecture is now complete in miniature and every part is earned:

1. **Reservoir** — fast, fading, nonlinear processing (working memory).
2. **Holographic film** — slow, permanent, content-addressable store (#3),
   with a hard but *known* load threshold (~0.3N symbols for clean SIC recall).
3. **Nonlinear decider** — projection onto structured priors turns the film
   from a lookup table into an associative episodic memory (#4): a corrupted
   *content* cue retrieves *time* and *temporal context*.

This is the Seed coherence-gate story with numbers attached, and it is the
spec for the embodied next step: record the agent's (reservoir state ⊙ clock)
into a film during exploration, and let a familiar-smelling sensory state
retrieve *what happened next last time* — a prediction signal the ESN's fading
echo structurally cannot provide. Independent of all this, the #2a actionable
fix still stands open: `wim_brain`'s readout discards `vel`, halving capacity.

---

## #5 — The reel meets the world: embodied episodic memory

**Files:** [`05_embodied_episodic.py`](05_embodied_episodic.py),
[`../embodied/holo_memory.py`](../embodied/holo_memory.py)

### The slide architecture

One film has a hard load threshold (~0.3N clean bindings, #4-A) and a
*periodic* clock — timestamps alias after ~N steps, so an unbounded single
film fails on time alone, independent of crosstalk. Instead of fighting
either limit, long-term memory is a **reel of slides**: each slide is one
film recording one bounded episode; the budget (or an event boundary — death,
surprise) closes it and a fresh slide starts. The capacity wall becomes the
episode length; the clock's finite horizon becomes the episode's internal
timeline. Within-slide time is fine-grained (clock phases), across-slide time
is ordinal (slide index) — matching the human signature of sharp
within-episode order and fuzzy cross-episode dating. Episodes never
crosstalk; the reel grows linearly. That linear growth is the honest price of
permanence, and it is also the consolidation hook (merge/compress slides
offline — the OCA circadian cycle).

Pipeline: obs → PCA features → VQ codebook (64 prototypes = the symbol
attractors) → phasor code ⊙ clock → current slide. Recording is event-based:
a record happens only when the quantized symbol *changes*, so slide time
advances with the world, not the frame rate (20k steps → ~3.5k records → 30
slides). Valence rides along as a side tag per record. The phasor encoder is
a fixed random projection into phase space, which makes #4-B's noise model
exact: a cue's effective σ is `s·‖obs − prototype‖` — quantization error IS
cue noise, and #4-B's tolerance (σ ≲ 2) becomes a concrete quantization
budget (median σ_eff set to 1.2).

### Two structural lessons the world taught (v1/v2 dead ends, documented)

1. **A natural codebook is not a random codebook.** On a continuous state
   manifold, nearest prototypes sit ~1 quantization-distance apart (σ-equiv
   1.2-1.4) — neighboring symbols look like noisy copies of each other, where
   #4's random symbols were near-orthogonal. SIC fidelity drops from ~100%
   (toy) to ~90% (world), with confusions landing on neighbor prototypes.
   PCA whitening made it *worse* (amplifies jitter directions; fidelity 82%):
   denoise the metric, don't distort it.
2. **Episodic addressing is trajectory addressing.** All occurrences of a
   symbol have *identical* clean phasors, so a single-moment cue can only
   find *a* time the symbol occurred — never *which* time (exact-moment
   retrieval: 0.7%). Cueing with a short trajectory stub (last L transition
   features, window-aligned scoring) disambiguates: exact-moment 0.7% → 20%
   (L=3) → 34% (L=5), and effective separation grows with L. "When" is not a
   property of a state; it is a property of a path through states.

### Result (20k-step recording run; 300 queries; K=64, N=512, budget 120)

Baselines: **marginal** (most common symbol) and **Markov-1** — a
purpose-built global transition table over the same data, i.e. *semantic*
memory competing against the reel's *episodic* retrieval.

| | REPLAY (cue from recorded run) | TRANSFER (new world layout, never recorded) |
|---|---|---|
| retrieval precision (top-1 symbol) | 71% | 38% |
| exact recorded moment found | 34% | — |
| next-symbol: **film** | **52%** | **16%** |
| next-symbol: markov | 38% | 15% |
| next-symbol: marginal | 6% | 5% |
| Δvalence forecast r | **+0.57** | +0.20 |

- **Episodic beats semantic on its own past** (52% vs 38%): five retrieved
  instances, voted, out-predict the full transition table — from one
  superposed phasor vector per episode plus a decider.
- **Transfer ties semantic** (16% vs 15%, both 3-4× marginal) in a world the
  reel never saw: "what happened next last time" carries across layouts
  because the symbols are egocentric.
- **The valence tags ride back with recall** (Δvalence r=+0.57 replay, +0.20
  transfer, vs 0 for persistence): the memory answers "did this kind of
  moment get better or worse last time?" — precisely the RPE-gating signal
  an embodied brain needs, and one the fading echo structurally cannot
  provide (it forgets in ~16 steps; the reel never does).

### Next

- **Live brain integration**: mount `HoloEpisodicMemory` in a running brain
  (seed/wim/ESN), recording reservoir state instead of raw observations, and
  bias action selection by recalled Δvalence ("this smelled good/bad last
  time"). → done in #6.
- **Consolidation**: SIC-clean and re-record old slides (crosstalk laundering),
  merge near-duplicate episodes — sleep for the reel.
- Still open from #2a: `wim_brain` readout discards `vel`, halving capacity.

---

## #6 — The reel mounted in a live brain

**Files:** [`../embodied/holo_mount.py`](../embodied/holo_mount.py),
hooks in [`../embodied/brain.py`](../embodied/brain.py) (config-gated,
default off)

`HoloMount` wraps the #5 reel as an online component for any reservoir brain:
feed it the hidden state (or observation) and the reward each step, and it
returns **dv̂** — the recalled Δvalence of "the last time the recent
trajectory looked like this". This is episodic control in the film substrate
(cf. Blundell's Model-Free Episodic Control): instead of a k-NN table of
(state, value) pairs, the store is one superposed phasor vector per episode,
addressed by trajectory-stub recognition, with the recalled value riding back
as side tags. Everything is online: it calibrates itself (random-projection
sketch for reservoirs > 256 dims → PCA → VQ → phasor scale), then records,
recalls, and keeps an honest scorecard — every dv̂ is held as a pending
prediction and resolved against what valence actually did (**foresight r**),
with matches inside the query's own recent tail excluded (recalling one
second ago is not foresight).

In `brain.py` the mount is opt-in via `holo_*` config keys; `learn()` shapes
`reward + β·dv̂` (β=0 default = pure observer), episode resets close slides,
and the reel persists beside the checkpoint (`.npz.holo.npz`).

### Engineering lessons (each one earned the hard way)

1. **The active slide is working memory.** Re-running SIC on the active slide
   every query is O(count²) per step — and pointless: the writer knows what
   it just wrote. Closed slides are read holographically (SIC once, cached
   forever — the past exists only in the medium); the present episode is read
   from its session-known symbols. Past = film, present = echo: the #3
   architecture, now forced on us by a profiler.
2. **Track the situation, not the state.** A noisy reservoir
   (noise_scale 90) flips its VQ symbol every frame — the reel recorded
   framerate noise (~1 record/step, dv̂≡0). EMA-smoothing the sketched state
   (τ≈12 steps) plus a 2-step dwell gate restored #5's natural transition
   rate (~0.17/step). Event-based memory needs an event-rate signal.
3. `OMP_NUM_THREADS` matters when numpy recall shares a process with a torch
   reservoir (36 min of sys-time thrash → 388% CPU clean).

### Live results

Mounted on **Bob-16k** (trained 16384-unit ESN checkpoint, RTX 4090, frozen
readout, 35k steps, observe-only): runs at ~240 steps/s. Source matters in
texture but not in conclusion (an earlier config-passthrough bug made both
"sources" the reservoir state; fixed and re-run): **state**-sourced — ~0.16
transitions/step, precision ~37%, foresight r ≈ 0 with almost no valence
events (10-70); **obs**-sourced — ~0.06 transitions/step (EMA'd observations
are calmer), precision ~32%, far more events (358) but foresight mildly
*negative* (−0.06 overall, −0.11 events). Either way: no usable signal.

Self-test (`python holo_mount.py`): the *same mount* under the structured #5
explorer policy (approach food / flee danger / wander):

| resolved predictions | foresight r (all) | event r (\|Δv\|>0.05) | events |
|---|---|---|---|
| ~1800 | **+0.09 → +0.14** | **+0.15 → +0.21** | ~35% of resolutions |

The event-level r ≈ +0.2 matches #5's transfer r = +0.20 almost exactly —
live foresight *is* the transfer regime, as it should be (the future is
never in the reel).

### The finding

**The reel is only as prophetic as the life it records.** Bob's actions are
exploration-noise-dominated (explore_noise 90 through tanh ≈ near-random
motor babble), so trajectory motifs don't repeat and valence events are rare
at the transition timescale; the structured explorer lives in repeating
approach→eat / drift→danger motifs, and the same memory extracts a real
predictive signal from them. Episodic memory and behavioral structure are
co-dependent: the memory can only pay off for an agent whose policy already
produces recurring situations — which is precisely the RL bootstrapping
argument for β > 0 (memory-shaped reward → more structured behavior → more
prophetic memory), and the right next experiment once a brain with
structured behavior is learning online.

### Next

- β > 0 A/B on a brain with structured behavior (teleop-taught or Seed-Hemi):
  does memory-shaped RPE close the loop? → done in #7.
- Consolidation: decode, deduplicate, re-record — sleep for the reel.
- Confidence gating: dv̂ should only speak when retrieval score is high.
- Still open from #2a: `wim_brain` readout discards `vel`, halving capacity.
  → closed: `readout_vel` config in `wim_brain.py` reads the full phasor
  `[tanh(disp), tanh(vel)]`, with auto-migration for disp-only checkpoints.

---

## #7 — β > 0 A/B on Seed-Hemi: flat, as predicted, and why

**File:** [`06_seed_beta_ab.py`](06_seed_beta_ab.py); `holo_*` hooks in
[`../embodied/seed_brain.py`](../embodied/seed_brain.py)

The reel mounted on the hemispheric Seed brain (reward-modulated Hebbian
plasticity; dv̂ enters the RPE that gates it). Paired 15k-step headless runs
from the trained Seed-Hemi-64_af checkpoint, matched world seeds, β ∈
{0, 0.6}, reel live after 5k. Nothing saved.

| world | β | valence (post-cal) | eats | deaths | fore r | event r |
|---|---|---|---|---|---|---|
| 7 | 0.0 | −0.117 | 1 | 1 | −0.07 | −0.04 (n=16) |
| 7 | 0.6 | −0.129 | 1 | 1 | +0.11 | **+0.64 (n=27)** |
| 21 | 0.0 | **+0.420** | 0 | 0 | +0.26 | — (n=0) |
| 21 | 0.6 | −0.037 | 1 | 1 | −0.26 | −0.01 (n=17) |

Reading (n=2 pairs — no statistical claims, only mechanics and diagnosis):

- **The loop closes mechanically.** dv̂ flows through the RPE into Hebbian
  plasticity for 10k shaped steps without instability; mount overhead ~5%
  (59 → 55 steps/s on CPU).
- **The result is flat-to-negative, for the reason #6 predicted.** Seed-Hemi
  eats 0-1 times per 15k steps — its life contains almost no valence events,
  so the reel has nothing recallable to bootstrap (event-n: 0-27 per run).
  β amplifies episodic structure; it cannot create it.
- **World 21's big negative delta is attractor divergence, not a β effect.**
  The β=0 run drifted into a *food-staring attractor*: valence climbs to
  +0.48 with zero eats, sustained purely by `food_vision_reward`
  (+0.002/step at max proximity). The shaped run's perturbed trajectory left
  that basin and found danger instead. Two lessons: (i) single paired runs
  in a chaotic plant measure trajectory divergence, not treatment effect;
  (ii) **the reward landscape has a degenerate no-op optimum** — an agent
  can farm vision-valence by parking in front of food it never eats, which
  may be quietly shaping every embodied result in this folder.
- One tantalizing crumb: the world-7 shaped run's reel reached event
  foresight r = +0.64 (n=27) — when events exist, the recall is predictive.

### Next: teleop as the structure injector

The loop needs behavioral structure to amplify, and teleoperation is the
direct way to inject it: teach approach→eat by demonstration, let the reel
record it, then let dv̂ keep re-evoking the demonstrations after the teacher
lets go — episodic memory as a demonstration amplifier. Planned mechanics:
tag slides recorded under `teacher_forced` and weight demonstration episodes
above self-play in recall. Separately: consider capping or saturating
`food_vision_reward` to close the food-staring exploit.

---

## #8 — Scripted teacher: the reel recalls values, not actions

**File:** [`07_scripted_teacher.py`](07_scripted_teacher.py); demo-hit
counter added to [`../embodied/holo_mount.py`](../embodied/holo_mount.py)

### Post-mortem of the human teleop session first

A ~1-hour live teleop session (2026-07-05, `Wim-Teleop-64`, run right after
the demo-weighting commit) showed no visible behavioral change. The saved
reel explains why: the machinery worked perfectly — 1,820 taught records
across 19 slides with mean tag **+0.82**, against 15,651 self-play records
at **−0.04**; the demonstrations are the highest-valence content in the
reel — but teaching covered <1% of a ~200k-step session on a fresh brain,
and the session log shows mean valence *declining* by quarter (+0.33 →
−0.09) while the demos sat unrecalled. The dose was homeopathic.

### Protocol

The #5 Braitenberg policy replaces the human as teacher, at a controlled
dose: fresh WimBrain per run (matched brain seed 42), alternating 1.5k-step
teach/free blocks over steps 4k–18k (~25% coverage), then a 12k-step
retention window. Teacher actions enter through the exact live-teleop path
(`set_executed_action(a, teacher_forced=True)`). Worlds 7/21/33, three
arms: **baseline** (no teaching, reel observing), **teach** (β=0 — pure
supervised nudge, reel observing), **teach+dv** (β=0.4, demo_weight=3 —
the full demonstration-amplifier stack). New metrics: **imitation** =
Pearson r between teacher turn command and the brain's deterministic
proposal on the same obs, free-play steps only; **demo-hit** = fraction of
recalls whose top match is a taught record. ~200 steps/s on GPU; nothing
saved.

### Results (retention-late deltas vs baseline, same world)

| world | arm | Δimit (turn r) | Δval | Δdeaths | demo-hit (ret) |
|---|---|---|---|---|---|
| 7  | teach    | **+0.08** | +0.031 | −1 | 100% |
| 7  | teach+dv | −0.00 | −0.016 | −1 | 100% |
| 21 | teach    | **+0.18** | +0.001 | −1 | 100% |
| 21 | teach+dv | +0.12 | −0.000 | −1 | 100% |
| 33 | teach    | **+0.07** | +0.151 | −2 | 100% |
| 33 | teach+dv | +0.04 | +0.169 | −2 | 100% |

- **The supervised nudge leaves a real but weak trace.** Δimit positive in
  3/3 seeds for the teach arm (mean +0.11), eat-agreement up in 3/3
  (+17/+11/+46 pts), and **0 deaths in all 6 taught runs** vs 4 across the
  3 baselines. Direction right, magnitude a whisper — absolute imitation r
  stays ≈0.05; nobody watching the game would see it (consistent with the
  human session's "no visible change").
- **Recall is saturated by demonstrations, and it does not matter.** After
  teaching ends, demo-hit is 100% — at demo_weight=3 with taught records
  ~15% of the reel, essentially *every* recall's top match is a taught
  record. The amplifier mechanism is maximally engaged. Yet teach+dv is
  *weaker* than teach alone on imitation in 3/3 seeds. dv̂ flowing through
  the RPE adds nothing the nudge didn't already do.

### The finding

**The reel is a value memory, not an action memory — so demonstration
recall has no path to the motor readout.** Slides store side *tags* =
valence; the teacher's actions are not in the reel at all. "What did the
teacher do here" is unanswerable by construction; the only recallable
content is "how did valence move when the teacher was here," and that
enters as reward shaping → RPE → eligibility trace, which credits *the
agent's own executed actions*. dv̂ can sculpt the value landscape but
cannot inject a policy; for it to amplify demonstrations, exploration
would have to spontaneously re-enact demo-like trajectories so the boost
lands on the right actions — which a fresh noisy brain almost never does.
This is #7's lesson one level up: β amplifies episodic structure it cannot
create, and value-only recall amplifies *values* it cannot turn into
*actions*.

### Next

- **Action side-tags:** record the executed action alongside valence
  (actions ride the film exactly as tags do), so recall returns
  (situation → what-was-done, what-happened) instead of value alone.
- **Recollection as suggestion:** on high-score taught recall, blend the
  recalled action into the motor proposal (episodic control proper, NEC
  Pritzel et al. 2017-style) — giving demonstrations a direct,
  supervised-strength path to behavior that persists after the teacher
  lets go.
- Confidence gating (open since #6) becomes load-bearing here: recalled
  actions should only speak when retrieval score is high.
- Still open: consolidation (decode, dedup, re-record — sleep for the
  reel).

## #9a — The reel reads sheet music: episodic advantage is real, and the foresight currency was saturated

**Setup** (`08_music_reel.py`, `melody.py`; plan in `09_music_plan.md`): songbook of P phrases
(16 notes, 8 motif incipits, cadence rule) from a first-order Markov listener model at
temperature τ; one phrase per slide, one-hot codebook (VQ bypassed — exact quantization,
clean cues); record pass then re-performance test pass, top-1 query on 3-note stubs.
3 seeds × τ ∈ {0.4, 1.0, 2.5} × P ∈ {16, 64, 256}. Full sweep: 170 s, CPU.

```
 tau    P |   id%   pos% | next_reel next_gram | r_reel r_markov |    adv
 0.4   16 | 64.6%  61.9% |     0.753     0.444 |  0.925    0.940 | -0.015
 0.4   64 | 30.6%  29.2% |     0.535     0.463 |  0.836    0.941 | -0.106
 0.4  256 | 12.7%  12.6% |     0.446     0.467 |  0.836    0.938 | -0.102
 1.0   16 | 81.6%  80.0% |     0.864     0.319 |  0.931    0.942 | -0.011
 1.0   64 | 50.9%  50.6% |     0.646     0.327 |  0.819    0.939 | -0.121
 1.0  256 | 27.3%  27.1% |     0.466     0.329 |  0.761    0.940 | -0.179
 2.5   16 | 84.1%  83.8% |     0.888     0.228 |  0.960    0.963 | -0.003
 2.5   64 | 65.9%  65.8% |     0.752     0.261 |  0.862    0.964 | -0.102
 2.5  256 | 40.9%  40.8% |     0.543     0.257 |  0.708    0.964 | -0.256
```

**Findings:**
1. **The thesis confirmed on the prediction axis.** Episodic advantage in next-note accuracy
   widens with entropy exactly as designed: at P=16, reel−grammar = +0.31 (τ=0.4) → +0.55
   (τ=1.0) → +0.66 (τ=2.5). Atonal-but-memorized is where episodic memory earns its keep;
   grammar-predictable streams leave it little to add. First clean measurement of the
   parametric↔episodic tradeoff in this stack.
2. **Temperature decorrelates the songbook.** Identity accuracy RISES with τ at fixed P
   (64.6→84.1% at P=16; 12.7→40.9% at P=256): low-τ phrases are all the same scale-runs,
   so 3-note stubs are ambiguous across phrases; high-τ phrases are distinctive. Interference
   with reel size is the dominant cost (id% roughly halves per 4× P). id≈pos everywhere:
   when the right phrase is found, alignment is essentially always right — identity, not
   position, is the bottleneck.
3. **Methods discovery — the foresight currency was parametrically saturated.** r_markov ≈
   0.94–0.96 across the board: surprisal-derived valence is almost fully predictable from
   the grammar (cadence/position structure dominates Δvalence), so dv̂ cannot beat the
   parametric oracle by construction — recall errors can only subtract (adv ≤ 0 everywhere,
   −0.26 worst at 2.5/256, while r_reel stays 0.71–0.96 absolutely). This retroactively
   sharpens #6's lesson: "the reel is only as prophetic as the life it records" needs an
   addendum — **and only as valuable as the part of the life that memory alone can know.**
   Surprisal-valence is knowable-by-grammar; episodic memory can only show its worth on
   value the grammar cannot see.
4. **One honest negative:** at τ=0.4 with P ≥ 64, wrong-phrase recalls actively mislead
   next-note prediction (reel 0.446 < grammar 0.467 at P=256). No confidence gating was
   used; the score-margin gate (#6) is the obvious fix and is now measurably motivated.

**Next:** (a) 9b transposition (ABS vs REL encoding) on this same harness; (b) redesign
valence for 9c/#10: give each phrase an idiosyncratic affect offset the grammar cannot
predict (memory-only-knowable value) — then dv̂ foresight becomes a fair fight instead of
a saturated one; (c) wire in score-margin confidence gating before 9c.

## #9b — Transposition: values vs differences. The derived-DoF claim, measured

**Setup** (`09_transposition.py`): same songbook machinery as #9a, melodies confined to
degrees 0..9, transposition = +4 degrees. Two reels record the same phrases: ABS (one-hot
scale degree, cue = last 4 notes) vs REL (one-hot successive interval, cue = the 3
intervals of the same 4-note window). Test in-key and transposed. 3 seeds; foresight
deliberately unscored (#9a: currency saturated). Full sweep 28 s.

```
 tau    P |  in-key id%   | transposed id% | transposed next
          |   ABS    REL  |   ABS     REL  |   ABS     REL
 1.0   16 |  91.3   78.6  |   6.1    78.6  |  0.064   0.797   (chance 6.2%)
 1.0   64 |  75.5   46.7  |   0.8    46.7  |  0.109   0.550   (chance 1.6%)
 2.5   16 |  93.4   87.2  |   5.2    87.2  |  0.073   0.882   (chance 6.2%)
 2.5   64 |  83.9   64.6  |   1.4    64.6  |  0.093   0.678   (chance 1.6%)
```

**Findings:**
1. **Transfer is exact, collapse is total.** REL's transposed columns equal its in-key
   columns to the decimal — interval streams are bit-identical under transposition, so
   invariance is exact by construction, and the experiment proves the encoding actually
   delivers it through the full record/recall pipeline. ABS drops to statistical chance
   in every cell (6.1% vs 6.2% chance, 0.8% vs 1.6%, ...). Musical identity lives on the
   derived DoF; the reel confirms the framework's oldest structural claim.
2. **The invariance tax is real, grows with load, shrinks with entropy.** In-key, ABS
   beats REL everywhere: +13 pts (τ1.0/P16) to +29 pts (τ1.0/P64). Mechanism: interval
   streams are self-similar — scale runs (+1,+1,...) recur across phrases, an ambiguity
   absolute encoding never pays. Higher τ decorrelates intervals and shrinks the tax
   (P64: 29 → 19 pts). This is the invariance ↔ absolute-information complementarity
   pair (ro_framework §7.2), now with numbers: at τ1.0/P64 you pay 29 points of in-key
   identity to keep 45 points over chance under transposition.
3. **Neither encoding dominates → the structural answer is both.** A dual-reel ensemble
   (ABS + REL side by side, answer taken from whichever recall clears the higher score
   margin) would inherit ABS's in-key sharpness and REL's transfer — multiple mappings
   with different reference frames, complementarity managed by carrying both. Cheap
   micro-experiment; queued behind 9c.
4. Methods note: one real bug caught by the smoke test — the REL cue window initially
   included iv[t], the interval leading into the yet-unheard note (answer leakage +
   one-position misalignment). Causal windows matter; worth a regression check in 9c.

**Next:** score-margin confidence gating (#9a finding 4), phrase-level affect offsets
(memory-only-knowable valence) — then 9c, the mounted listener.

## Gate — confidence gating closes the inversion, and falsifies its own blanket version

**Setup** (`10_confidence_gate.py`; `margin` + `margin_slide` added to
`HoloEpisodicMemory.query()`, 443 tests green): 9a harness, policy = reel prediction iff
confidence ≥ θ else grammar argmax; one query pass, θ-curves post-hoc; both margin signals.
3 seeds × τ ∈ {0.4, 1.0, 2.5} × P ∈ {64, 256}.

**Findings:**
1. **The #9a inversion cell is closed.** τ0.4/P256: gated 0.484 > reel-only 0.446 and
   grammar-only 0.467, at θ=0.01 with 4% coverage and 0.99 precision-when-used — when the
   reel is unambiguous it is essentially always right; everything else goes to the grammar.
2. **The blanket gate is falsified everywhere else** (pre-registered prediction 2 was
   wrong): in every healthy cell, any θ > 0 *hurts* (τ2.5/P64: 0.752 → 0.629 at θ=0.01).
   Mechanism: margins are near-binary — either exact ties or clear wins, almost nothing
   between — because recurring trigrams (motifs, scale runs) produce IDENTICAL clean
   phasors across slides. And tied recalls are benign for continuation: the wrong phrase
   with the same local pattern usually has the same next note. Identity errors ≠
   prediction errors.
3. **margin vs margin_slide: no difference** (prediction 1 also failed) — exact ties
   dominate both signals for the same reason.
4. **The real lesson, for 9c:** the reel serves two distinct functions with opposite
   gating needs. As a *pattern-completer* (next-symbol continuation), identity mistakes
   are harmless and gating should be ~off. As an *episode-indexer* (importing value dv̂ or
   suggesting whole continuations from a specific past episode), identity is everything
   and gating should be strict (θ small but nonzero: 0.99 precision at 4% coverage is
   exactly the profile value-import wants). One memory, two read policies, gated
   differently — this goes into 9c's design as a requirement, not an option.

**Next:** phrase-affect valence (memory-only-knowable value), then 9c with dual read
policies: ungated continuation, strictly-gated value/suggestion import.

## Affect — episodic value pays only through the gate

**Setup** (`11_phrase_affect.py`; `affect_amp` spikes in `melody.Songbook`): each phrase
gets ONE valence spike at a phrase-specific position (random sign, amplitude A) — value
the grammar cannot see. Design note that mattered: a phrase-CONSTANT affect offset cancels
out of Δvalence entirely; episodic value must be an *event*. Forecasters: markov oracle
(spike-blind), ungated reel dv̂, and the dual-read hybrid (reel iff margin_slide ≥ 0.01,
else markov). τ=1.0, P ∈ {16,64,256}, A ∈ {0,0.5,1,2}, 3 seeds.

```
   P    A |  r_reel r_markov r_gated |    adv   adv_g |   id%  cov%
  16  0.0 |   0.931    0.942   0.972 | -0.011  +0.030 | 81.6% 68.3%
  16  2.0 |   0.918    0.934   0.971 | -0.016  +0.037 | 81.7% 68.3%
  64  0.0 |   0.819    0.939   0.958 | -0.121  +0.018 | 50.9% 31.2%
  64  2.0 |   0.746    0.925   0.950 | -0.180  +0.025 | 50.5% 30.6%
 256  0.0 |   0.761    0.940   0.947 | -0.179  +0.006 | 27.3% 11.8%
 256  2.0 |   0.582    0.927   0.935 | -0.345  +0.008 | 25.5%  9.7%
```

**Findings:**
1. **The ungated crossover never comes — at any identity level.** Both pre-registered
   ungated predictions failed, and the mechanism is exact: on CORRECT recalls the reel's
   tags already include the spike, so dv̂ was already a perfect forecaster — affect adds
   zero headroom. On WRONG recalls the reel imports the wrong phrase's spike (wrong moment,
   possibly wrong sign) — error that scales with A. Ungated episodic foresight therefore
   has no affect upside, only downside: adv worsens monotonically with A in every P
   (P=256: −0.179 → −0.345).
2. **The gated hybrid wins every cell and profits from affect.** adv_g > 0 in all 12 cells,
   growing with A where coverage is decent (P=16: +0.030 → +0.037; P=64: +0.018 → +0.025).
   At P=16 the hybrid reaches r = 0.972, beating the saturated parametric ceiling even at
   A=0 — gate-kept recalls carry exact tags, better than the Monte-Carlo oracle everywhere
   they fire.
3. **The design law for 9c and #10, in one line: episodic value must be imported through
   an identity gate or not at all.** This is #8's lesson completed: what the loop carries
   matters (#8), and carrying it back to the WRONG moment is worse than carrying nothing —
   closure pays in proportion to addressing reliability. For the v2 framework's closure
   story this adds a measurable rider: Closed(O) with unreliable d_meta addressing is a
   net-negative loop; the gate is what makes consumption safe.
4. Bottleneck at scale: P=256 coverage ~10% caps adv_g at +0.008. Coverage, not precision,
   is now the limiting resource — longer cue windows (4-5 notes) are the obvious knob;
   test in 9c.

**Currency decision going forward: r_gated is the foresight metric for 9c and #10.**

## #9c — The mounted listener: recollection-as-suggestion works, and confidence means consensus

**Setup** (`12_mounted_listener.py`): numpy ESN (256 leaky units, online softmax readout)
predicting next symbols; reel mounted alongside, recording the same stream (valence =
surprisal + affect spikes A=1). Exposure schedule: 8 COMMON songs × 12 performances vs
8 RARE songs × 2. Test: every song re-performed, learning frozen. Suggestion = recalled
continuation OVERRIDES the readout when the identity gate fires; dv arm = gated dv̂
modulating learning rate (β=0.5), agent-side only. τ ∈ {0.8, 2.0}, 3 seeds.

```
 tau  train | common: base  sugg | rare: base  sugg | fire%   hit   r_dv
 0.8  plain |        0.561 0.822 |      0.283 0.639 | 55.8% 0.988  0.996
 0.8     dv |        0.589 0.822 |      0.283 0.642 | 55.8% 0.988  0.996
 2.0  plain |        0.589 0.850 |      0.242 0.753 | 65.4% 0.998  0.998
 2.0     dv |        0.603 0.861 |      0.233 0.753 | 65.4% 0.998  0.998
```

**Findings:**
1. **The discovery came from the smoke test: agreement is not ambiguity.** With repeated
   exposures the reel holds duplicate slides of the same song, so margin_slide reads exact
   ties everywhere and the gate NEVER fires (0.0%). Duplicates compete on uniqueness while
   agreeing on content. The correct confidence signal for value/policy import is
   **margin_agree**: lead over the nearest candidate that predicts something *different*
   (continuation symbol or conflicting tags). Re-experiencing strengthens memory; a
   uniqueness-based gate reads that strength as confusion, a consensus-based gate reads it
   as corroboration. Implemented as `query_gated()` in the experiment; promote to
   `holo_memory.query()` next session.
2. **Recollection-as-suggestion — queued since #8 — works.** With the consensus gate:
   fires on 56-65% of test positions at 0.99+ precision, lifting rare-song accuracy
   +0.36 to +0.51 (0.242 → 0.753 at τ=2.0) — 2-exposure material the parametric learner
   cannot own. The #8 diagnosis is fully repaired: when the film carries continuations
   (policy-relevant content) and imports them through an identity gate, recall steers.
3. **Honest deviation from prediction 1: common songs also gained (+0.26).** The ESN
   readout is a weaker parametric learner than assumed (0.56-0.60 on 12-exposure songs),
   so the override helps everywhere, not just on rare material. The dissociation exists
   (rare gains more at τ=2.0: +0.51 vs +0.26) but is partly masked by parametric
   weakness. Stronger readout or longer training would sharpen it; not rerun tonight.
4. **The value channel, closed fairly.** dv-modulated plasticity changes almost nothing
   (≤ +0.03 anywhere) — completing the #7→#8→tonight arc: value tags can't inject policy
   (#8), value import without identity gating is net-negative (Affect), and even gated
   value consumed as learning-rate modulation barely moves behavior when the policy
   channel already exists. The loop's cargo should be content; value is seasoning.
5. Gated foresight in the mounted setting: r_dv = 0.996-0.998 — with consensus gating and
   a small reel, recalled Δvalence is essentially a perfect forecaster. The #6 scorecard
   metric ("foresight r") finally has a configuration that saturates it honestly.

**Next:** promote margin_agree into holo_memory.query(); #10 closure sweep / hysteresis
(the ro_framework v2 §5.5 falsification experiment) — the melodic stream now has
everything it needed: dense valence, working suggestion channel, trustworthy gate.

## #10 — The closure sweep: no hysteresis. Gradualism wins on the loop we swept

**Setup** (`13_closure_sweep.py`): performer loop — frozen-readout ESN samples notes;
gate-cleared recalls override with probability g; played notes are re-recorded (recall →
override → action → memory of own action). Songbook seeded as demonstrations
(demo_weight=3, taught=True — see finding 3). Protocols: g swept 0→1→0 (updown), 1→0→1
(downup, time-confound control), flat0 (drift baseline). τ=2.0, 3 seeds, 11 steps × 12
phrases. Order parameters: in-language 4-gram rate (gram), gate fire rate; plus strict
prompted-song fidelity and full-phrase lock.

**Result — unambiguous within this design:** all curves smooth and monotone in g
(gram 0.162 → 0.618, fire 0.43 → 0.65, no jumps); hysteresis areas −0.021..+0.008,
the same order as flat-drift noise (±0.003); both sweep directions agree; lock appears
only near g=1 (0.111) exactly as the compounding probability (fire·g)^12 predicts —
tail of a smooth function, not a threshold. **No discontinuity. No path dependence.
As pre-registered, this reads gradualist.** The loop was demonstrably alive (fire 0.65,
suggestion machinery proven in #9c), so the null is meaningful, not a dead-loop artifact.

**What exactly was falsified — the attribution, stated carefully:**
1. The swept gain routes EPISODIC-MEMORY CONTENT (world/action cargo) back into behavior.
   Nothing in this loop represents its own representing — by ro_framework v2's own
   taxonomy (§5.4) it is a closed but UNTWISTED loop, a class for which v2 makes no
   threshold claim. The §5.5 clause specifically concerns consumption of d_meta
   (self-model outputs). Read strictly: tonight retires the bridge-doc-era
   operationalization (holo_beta-as-closure-gain) — it was always a memory-consumption
   gain, not a self-model-consumption gain — and establishes empirically that memory
   loops scale gradually. The binary clause survives untested at its actual referent;
   testing it requires the v2-migration A–B machinery (a real d_meta with routable
   consumption), which the core-lib audit already scheduled.
2. HOWEVER: the honest bracket on even this narrower null — the demo_weight=3 crutch
   that prevented self-poisoning also CLAMPS path-dependence: with the songbook boosted
   3×, self-generated replays barely tilt recall consensus, so the medium's memory of
   its own trajectory (the substrate hysteresis needs) was partially suppressed. The two
   failure modes bracket the interesting regime: demo_weight=1 → self-poisoning kills
   ignition; demo_weight=3 → teacher prior kills path-dependence.

**Banked side-findings (both from smoke tests):**
3. **Self-poisoning:** an agent that indiscriminately records its own low-competence
   behavior buries the consensus its memory-guidance depends on — with demo_weight=1 the
   loop can NEVER ignite on ascent. Consolidation ("sleep for the reel", parked since #6)
   is hereby promoted from wishlist to structural necessity.
4. **The medley attractor:** when the loop does engage, it closes onto the songbook's
   LANGUAGE, not onto faithful episodes — replay hops songs at shared motifs (gram 0.62
   while prompted-fidelity stays 0.24). Closure targets the corpus-attractor, not the
   episode.

**Next (the fork):**
(a) **Self-taught variant:** record the performer's own gate-cleared overrides as
    taught=True — replays then gain recall priority and the compounding mechanism
    (consensus → firing → consensus) is restored WITHOUT the teacher clamp. This is
    confidence-begets-confidence by construction — delusion dynamics — which is exactly
    why it is the right next probe for path dependence.
(b) **The real §5.5 test** waits on v2-migration phases A–B (core-lib d_meta + gain g in
    observe()): sweep the SELF-MODEL consumption gain, not the memory gain. Tonight's
    design transfers wholesale; only the cargo changes.

## #10b — Self-taught variant: the ratchet, not the lock. Consensus confidence fails safe

**Setup** (`13_closure_sweep.py --self-taught`): identical to #10 except gate-cleared
overrides are recorded taught=True — the performer's confident replays gain the same 3×
recall priority as the teacher's demonstrations. Same protocols, seeds, grid.

```
                          plain #10        self-taught
fire at g=1 (fresh)         0.648             0.488
fire at g=1 (updown peak)   0.650             0.269   ← falls WITH g
gram peak                   0.618             0.323
fire hysteresis area     -0.004/-0.007     -0.088/-0.081  (flat drift +0.002)
```

**Findings:**
1. **Path dependence found — and it is a ratchet, not a memory.** Fire-rate areas are
   −0.088/−0.081 in BOTH sweep directions, 40× the flat-drift baseline. Same sign under
   updown and downup means it is not retention-at-matched-g (bistability); it is monotone,
   cumulative, irreversible degradation: fire falls with total overrides-so-far, never
   recovers when g returns (downup second visit to g=1: 0.275 vs 0.488), and flat0 shows
   zero decay without the self-teaching channel. The loop consumed its own confidence.
2. **Mechanism:** overridden notes include medley-hops and post-deviation continuations —
   locally song-like, globally inconsistent. Taught, they outrank the songbook and then
   DISAGREE WITH EACH OTHER at query time; margin_agree collapses; the gate stops firing;
   the junk stays. Each confident act, re-recorded as authoritative, eroded the authority
   structure that made confidence possible.
3. **No delusion-lock — the consensus gate fails SAFE.** The feared signature (fire stays
   high while content degrades) did not occur: fire fell WITH gram. Consensus-based
   confidence cannot be captured by diverse self-generated error, because diverse error
   disagrees with itself — the gate detects the contradictions it caused and shuts down.
   Silence, not delusion. Corollary and open probe: capture should require SELF-CONSISTENT
   false content (correlated error). The earworm experiment: inject one repeated false
   phrase as taught and test whether it captures the gate. Untested.
4. **The pair of runs, one statement:** for this untwisted memory loop, the teacher-clamped
   regime is stable but cannot compound (#10), and the self-authorized regime compounds
   only destructively (#10b). No self-reinforcing constructive regime exists in this
   architecture — self-authorization without external anchoring is a poison ratchet, and
   consolidation (selective, consistency-checked promotion of self-records) is the missing
   organ, now indicated by two independent failure modes.

**Next candidates:** the earworm probe (cheap, sharp); consolidation-as-organ (promote
self-records only when consistent with existing consensus); the real §5.5 d_meta sweep
(v2-migration A–B).

## Earworm (#14) — capture by flooding the consensus horizon: the day's one true discontinuity

**Setup** (`14_earworm.py`): one internally-consistent false phrase opening with a motif
UNIQUE to a single true song (fork unambiguous pre-infection: gate fires, plays truth),
diverging at position 4 into songbook-absent content. Injected at dose E (taught,
demo_weight parity). Performances at g=1.0 prompted with the true host song — every
prompt ends at the fork. Static vs self-taught modes. 3 seeds.

```
dose  mode  | forkfire agree | fork_true fork_worm | ride  | wormgram
  0  static |   0.667  0.137 |    0.733     0.017  | 0.08  |  0.083
  1  static |   0.000  0.000 |    0.208     0.008  | 0.06  |  0.092
  4  static |   0.000  0.000 |    0.208     0.008  | 0.06  |  0.092
  6  taught |   0.000  0.000 |    0.236     0.028  | 0.39  |  0.099
  8  static |   1.000  1.000 |    0.000     1.000  | 0.74  |  0.843   ← step
 16  static |   1.000  1.000 |    0.000     1.000  | 0.74  |  0.844
```

**Findings:**
1. **Below the horizon: jamming, never capture.** One worm copy suffices to close a
   previously-confident fork (fire 1→0, truth collapses to ESN-chance) and additional
   copies change nothing (1 ≡ 4 ≡ 6). The gate will not choose between two consistent
   stories — the fail-safe of #10b extends to consistent false content. The attack
   available to a minority story is denial-of-service: it turns confidence into doubt.
2. **At the horizon: total, unanimous capture — a true step function.** margin_agree
   scans an 8-candidate window (holo_memory's decode window), tie-broken by recency.
   At dose ≥ 8 the worm's copies monopolize the window; the true song's 3 copies are
   never seen; the gate finds UNANIMITY (margin 1.0) and fires maximally; the worm is
   played at 100% of the true song's own prompts. Transition from fully-jammed to
   fully-captured in one dose step at exactly the window size. **Consensus confidence is
   only as safe as the diversity of the sample it is computed over: repetition past the
   horizon does not make a story more credible — it makes the alternative invisible.**
3. **Self-teaching corrodes even the invader.** In taught mode at capture, rides collapse
   (0.74 → 0.11) and wormgram falls (0.84 → 0.27): re-taught ride-deviations disagree
   with the worm's originals and jam its own interior — the #10b ratchet is
   content-agnostic. No epidemic below threshold within tested horizons (growth linear
   in chance entries).
4. **The day's irony, recorded:** we swept closure-space for a discontinuity all day and
   found smooth gradualism; the one genuine step function of the day lives in adversarial
   memory dynamics — a sharp threshold in epistemic security, not in consciousness. It
   also darkly mirrors the record-redundancy predicate from the theory side (definiteness
   = proliferation of copies): in a consensus-read memory, whoever writes the most copies
   past the horizon owns the fact.

**Defense implications (for the architecture, next sessions):** widen or stratify the
disagreement scan (sample distinct blocs across the FULL ranking, not top-8); weigh
consistency by distinct-source count rather than raw copies; the recency tie-break is an
attack surface; generalize `taught` into provenance (self/teacher/world) so consensus can
be diversity-weighted. Consolidation remains the missing organ — now with a security
requirement attached.
