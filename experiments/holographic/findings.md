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
