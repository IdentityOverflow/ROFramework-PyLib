# Feedback / fractal track — findings

Motivation (2026-10-12): analog video feedback (camera pointed at its own
monitor) makes fractals; fractal imagery recurs in NDE, DMT and other altered
state reports. The RO framework already uses the camera–monitor rig as its
example of a closed, untwisted loop (`docs/ro_framework.md` §5.3, wording fixed
this session — earlier copies called it "recursion without closure", which
contradicted the thermostat example in §5.4). Question for the program: what
does the geometry of a closed loop do across the gain axis, and does the twist
contribute anything to it?

Three experiments share one axis — loop gain — read on three substrates: a
2-D image loop (#F2), a reservoir / the Seed in state space (#F3), and the
Seed's spatial correlation structure (#F1, which also finishes the
`seed_architecture.md` §4.2 fractal claim that was only half measured).

Tools: `fractal_tools.py` (box-counting dimension, spectral slope,
Grassberger–Procaccia correlation dimension, participation ratio).

---

## #F2 — The video rig (`02_video_rig.py`)

frame' = sat(g · A(frame) + seed_object + noise); A = rotate 7°, zoom s, blur
0.7 px; sat = clip, optionally with a contrast curve about a black level
(contrast 3, pivot 0.2 — a sharpening decider). 128×128, g swept 0.5→1.6→0.5
with state carried across steps (hysteresis protocol), 150 frames per gain,
2 seeds (seed noise only; the rig is deterministic to 3 decimals).

### Plain clip (contrast 1): two different transitions for the two zoom directions

| zoom | g < 1 | around g = 1 | g > 1 |
|---|---|---|---|
| out (s=0.92, copies shrink inward) | faint spiral of copies, D_box 0.5→0.8 | D_box 1.1 at g=1.01 | D_box climbs smoothly 1.3→1.72, local saturation only (11% at g=1.6), **no whiteout, no hysteresis (area 0.0000)** |
| in (s=1.08, copies grow outward) | fan of copies, D_box 0.6→0.9 | **band g∈[0.94,1.01]: D_box 1.15–1.38** | **whiteout step at g=1.01→1.09 (lit 0.03→0.99, D_box→2.0)**, hysteresis area +0.071: on the way down the screen stays white at g=1.01 where on the way up it was dark |

The flash protocol (seed object shown for 10 frames only) reproduces the
rig's famous bistability for zoom-in: black stays black at every g on the way
up, white stays white on the way down to g=1.01 (area +0.073). For zoom-out
nothing self-sustains at any gain.

Reading: the whiteout is a first-order transition **of an untwisted loop**.
Hysteresis is cheap; a camera and a monitor have it. This is the plainest
possible illustration of why v2.1 disavowed dynamical hysteresis as a marker
of anything consciousness-related.

The "fractal band" in the hard-clip rig is narrow (g≈0.94–1.01 for zoom-in)
and, inspected (`figures/02_plain_c1_*`), is a blurred fan of ~6 visible
copies, not crisp self-similar structure: the only nonlinearity is the clip,
so blur dominates and the interior is mush.

### With a decider (contrast 3, pivot 0.2): geometry becomes gain-independent

| zoom | D_box at g=0.5 | g=0.65 | g≥0.8 | whiteout | hysteresis | flash self-sustains? |
|---|---|---|---|---|---|---|
| out | 1.18 | 1.43 | **1.52→1.60 plateau** | none (lit ≤ 11%) | 0.0000 | no, at any g |
| in | 1.37 | 1.69 | **1.74→1.77 plateau** | none (lit ≤ 20%) | 0.0000 | no, at any g |

The crisp self-similar "fingers" (`figures/02_plain_c3p0.2_zoom_in_seeded_s0.png`,
g=0.65 panel) appear at low gain where the decider lets only a finite number
of generations survive; above g≈0.8 the attractor fills in to a solid
iterated-function-system fixed-point set whose box dimension is set by the map
(θ, s), not by g. The decider removes the whiteout (the diffuse glow that
filled the screen is below the black level), removes the hysteresis, and
removes self-sustaining (the loop cannot keep itself lit without the seed
object — black is the only fixed point of a thresholded contraction with no
input).

This is the same invariant the grokking iterated-map experiment (14f) found:
**continuous carrier + nonlinear decider = reliable, gain-insensitive
structure**; remove the decider and you get the continuum's own behaviour
(there: chaos; here: a glow-driven first-order whiteout). The decider is what
turns "feedback near unity gain" into a stable fractal generator. In the rig
it is the monitor's gamma and the camera's black level; in the reel it is SIC
decoding; in the grokked net it is the softmax.

### The twist probe: T ≡ U exactly

Mapping drifts (θ ±3°, s ±0.02, period 400 frames). Arm T injects a glyph
rendering the live (θ, s) — a self-portrait of the mapping, consumed by the
next exposure; arm U injects the same glyph frozen at (θ₀, s₀).

| contrast | zoom | max abs ΔD_box over the sweep | mean | max abs Δlit |
|---|---|---|---|---|
| 1 | out | 0.042 | 0.004 | 0.004 |
| 1 | in | 0.000 | 0.000 | 0.000 |
| 3/p0.2 | out | 0.067 | 0.011 | 0.020 |
| 3/p0.2 | in | 0.035 | 0.005 | 0.001 |

Pre-registered null confirmed: whether the frame carries a representation of
its own transform makes no difference to the geometry of the loop. The
fractal is a property of closure + gain + decider; the self-indexing channel
does no geometric work. (Honest limit: this glyph channel is a trivial
channel in the v2.4 sense — it reads the mapping parameters directly, no
model consumes it except the next exposure. It is the right null for the
geometric question, not a test of the v2.4 criterion.)

Superseded run kept for the record: contrast 3 with pivot 0.5 starved the
seed object (input 0.35 maps to 0.05) and is not interpretable
(`results_02_video_rig.json`, keys `*/c3/*`); the pivot-0.2 rerun is
`results_02_video_rig_c3p02.json`.

---

## #F1 — Spatial scale invariance of the Seed (`01_seed_spatial_scale.py`)

Finishes `seed_architecture.md` §4.2. 128 nodes, ring lattice, Rule 2a only,
independent random pulse drives (the clean condition of seed/01), 8000 steps,
2000 burn-in, 3 seeds. Conditions: critical (Rule 2a on), frozen_lo (Rule 2a
off, weights ×0.3), frozen_hi (off, ×3.0). Temporal power law re-confirmed
in the critical arm (α=2.09, KS≈0.046, PASS 3/3; frozen_lo fails 3/3).

### The library function is not a test

`measure_scale_distribution` (pairwise-MI graph → connected-component sizes
→ `verify_power_law` with min_samples=3) was run on full recorded histories
at the library default threshold and at surrogate-calibrated thresholds
(99th percentile of time-shifted-pair MI, ×1, ×1.5, ×3), with a
circular-shift shuffled control.

| condition | real data | shuffled control |
|---|---|---|
| critical | one giant component (128/128) at every calibrated threshold → cannot score; 173 mostly-singleton components at thr=0.1 → no | **PASS at 6/10 thresholds** (β 1.8–3.0) |
| frozen_lo (σ=0.01) | PASS at 6/10 | PASS at 5/10 |
| frozen_hi (σ=5.9) | PASS at 4/10 | PASS at 5/10 |

Its verdict is uncorrelated with the condition and it passes time-shuffled
noise. Two reasons, both structural: (i) at criticality the correlation
length is large by definition, so the thresholded graph has ONE component
and there is no size distribution to fit — the function measures the wrong
object exactly where it matters; (ii) with min_samples=3 and sizes mostly
in {1,2,3}, the discrete MLE + KS accepts almost anything. Retired as a
spatial-scale test. The function stays in the library (tests cover it) but
should be documented as not validated; §4.2's "measure_scale_distribution …
confirms full fractal structure" line in the spec is wrong.

### The ring-distance correlation function is a test, and the Seed passes it

C(d) = mean |Pearson| between node activations at ring distance d
(d = 1…64), fit as power law (log C vs log d) and exponential (log C vs d):

| condition | σ | C(1) | C(8) | C(32) | ξ (1/e) | power slope | R² | exp slope | R² |
|---|---|---|---|---|---|---|---|---|---|
| critical ×3 | 0.84–0.98 | 0.33–0.36 | 0.12–0.13 | 0.014–0.020 | 8–9 | **−0.95…−0.98** | **0.92–0.96** | −0.04 | 0.65–0.76 |
| frozen_lo ×3 | 0.01 | 0.07–0.09 | 0.012 | 0.009–0.011 (floor) | 5–6 | −0.44 | 0.64–0.66 | −0.013 | 0.26–0.29 |
| frozen_hi ×3 | 5.8–6.0 | 0.03–0.06 | 0.009–0.012 | 0.008–0.010 (floor) | 5–6 | −0.24…−0.38 | 0.55–0.67 | −0.007 | 0.20–0.29 |

At criticality the spatial correlation decays as ≈ 1/d (slope −0.96) with
the power law beating the exponential by ΔR² ≈ 0.2–0.3 in 3/3 seeds, and
C(32) sitting 2–3× above the ~0.01 noise floor where an exponential with
the same ξ would predict 0.006. Off criticality (both directions) the
correlation is at the floor beyond a few neighbours and neither model fits.
Honest scope: a 1-D ring with 64 distances and one decade of C is a modest
dynamic range; the claim supported is "power-law-like decay, inconsistent
with exponential, present only at σ≈1", not a measured exponent to three
figures. The 2-D version (`seed_architecture.md` §4.2 Q5) would need a 2-D
lattice.

This is the spatial half of the Seed's fractal claim, measured for the
first time: **both halves now pass** (temporal cascades seed/01, spatial
correlation here), and both fail off criticality.

---

## #F3 — The gain axis in state space: reservoir and Seed (`03_reservoir_gain_sweep.py`, `04_tuned_scan.py`)

Measures after a transient: correlation dimension D2 (Grassberger–Procaccia
on the PCA-20 projection, 2500 points), participation ratio PR, state
variance, saturated fraction (|x|>0.95), largest Lyapunov exponent λ
(Benettin twin trajectories; the Seed twin shares the noise stream). 3 seeds.

### ESN (300 units, sparse random W at unit spectral radius, g = spectral radius; drive = one slow sine)

| g | 0.3 | 0.8 | 0.95 | 1.0 | 1.05 | 1.1 | 1.2 | 1.4 | 1.7 | 2.0 | 2.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D2 | 0.68 | 0.98 | 1.08 | 1.10 | 1.11 | 1.51 | 2.48 | 5.29 | 8.89 | 10.0 | 10.3 |
| PR | 1.00 | 1.13 | 1.54 | 1.73 | 1.87 | 2.52 | 3.86 | 9.80 | 33.6 | 56.8 | 76.3 |
| λ | −1.21 | −0.24 | −0.09 | −0.05 | −0.02 | −0.005 | −0.002 | +0.025 | +0.08 | +0.14 | +0.20 |
| sat | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.01 | 0.08 | 0.18 | 0.30 |

Textbook echo-state picture, with the geometry read out: below g≈1.05 the
state is a function of the 1-D drive (D2≈1, λ<0); D2 lifts off between 1.05
and 1.2 while λ is still slightly negative (the state carries structure of
its own before it is chaotic); λ crosses zero at g≈1.3; above that D2 grows
with saturation rather than collapsing — a sign-mixed random W goes chaotic,
it does not lock. No band, no threshold, no hysteresis protocol needed: the
curve is monotone. The "fractal band" in state space is the shoulder
g∈[1.05, 1.4] where D2 is non-integer and rising fastest and λ≈0 — the edge
of chaos, by its usual name.

### Seed, frozen positive weights × w_scale (Rule 2a off; correlated 4-sine drive)

| w_scale | 0.1 | 0.3 | 0.6 | 0.8 | 1.0 | 1.5 | 2.5 |
|---|---|---|---|---|---|---|---|
| σ (as measured) | 2.28 | 2.73 | 3.30 | 4.20 | 5.02 | 5.61 | 6.00 |
| D2 | 2.17 | 2.08 | 1.96 | 1.88 | 1.36 | 1.00 | (1.83: var=0 artifact) |
| sat | 0 | 0.02 | 0.13 | 0.23 | 0.40 | 0.76 | 0.999 |
| λ | −1.9 | −0.8 | −0.14 | −0.28 | −0.58 | −1.04 | −5.0 |

The hand-turned axis on positive-only ring weights goes from
drive-dominated (D2 ≈ the drive+noise baseline of 2.2) straight into
saturation lock (var→0, λ→−5 at ×2.5: the #C1 "closed so tight it stopped
flowing" state again). No chaotic regime, no dimension gain anywhere on this
curve. Note σ under a correlated drive is inflated (≥2.3 even at near-zero
coupling) — the known seed/01 caveat; it is not a gain readout here.

### Seed, self-tuned (Rule 2a on from the same initial scales)

| initial w_scale | 0.1 | 0.3 | 1.0 | 2.5 |
|---|---|---|---|---|
| σ reached | 0.99 | 1.00 | 1.03 | 0.99 |
| mean weight | −0.09 | −0.06 | −0.04 | −0.03 |
| D2 | 3.72 | 3.57 | 3.96 | 4.72 |
| PR | 3.11 | 2.99 | 3.46 | 4.02 |
| sat | 0.003 | 0.000 | 0.001 | 0.001 |
| λ | −0.39 | −0.40 | −0.11 | −0.16 |

The homeostat reaches σ≈1 from every start, including ×2.5 where the frozen
network is fully locked. It does so by **driving ~61% of the weights
negative** (04: `neg%` column) — Rule 2a's (1−σ) factor flips couplings
until the drive-induced branching is cancelled. The tuned network sits at
D2 ≈ 3.6–4.7, unsaturated, contracting — **off the frozen curve entirely**,
and 1.5–2.5 dimensions above the drive+noise baseline (2.24; drive alone
2.09). The frozen curve never gets there at any scale. The pre-registered
reading "self-tuned lands at the frozen curve's transition" was the wrong
question: the tuned weights are not a scaled copy of the initial ones, so
the frozen curve is not the curve through the point reached.

### #F3b — the curve through the point reached: α·W* around the self-tuned weights

| α | 0 | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| σ | — | 1.47 | 1.21 | 0.98 | **1.03** | 1.62 | 2.01 | 2.97 | 4.14 |
| D2 | 2.25 | 2.58 | 3.10 | 3.42 | **3.87** | 3.46 | 3.32 | 3.19 | 3.02 |
| PR | 2.21 | 2.28 | 2.43 | 2.71 | **3.43** | 3.49 | 3.16 | 3.01 | 1.95 |
| sat | 0 | 0 | 0.001 | 0.002 | **0.003** | 0.015 | 0.081 | 0.212 | 0.447 |
| λ | −23 (no loop) | −1.51 | −0.76 | −0.32 | **−0.12** | −0.16 | −0.17 | −0.20 | −0.17 |

**D2 peaks at α=1.00 exactly, λ is at its maximum at α=1.00 exactly, and
saturation takes off immediately after (0.3% → 1.5% → 8%).** Rule 2a parked
the loop at the knee of its own gain curve: the largest state-space
dimension the loop can hold before the decider (tanh) starts eating it,
with the dynamics as close to marginal as this contracting, noise-floored
architecture gets. The pre-registered F3b reading ("α=1 at the knee") is
confirmed, 3/3 seeds. Below the knee the loop adds dimension smoothly as
gain rises (2.25 → 3.87, no threshold); above it saturation removes it
smoothly. Gradualism again — but with a homeostat that finds the optimum of
the graded curve.

Two honest limits. (1) λ<0 everywhere on the Seed: positive-and-negative
ring coupling with a 0.05 noise floor and a 0.5 threshold never goes
chaotic at this size; "edge of chaos" here is "least contracting", not
"λ=0". (2) D2 on a noise-floored system is a scale-window quantity (fit
over the 2–30% distance quantiles); the baselines are measured with the
same estimator so the differences are meaningful, the absolute values less
so.

---

## #F3c — The creditable twist on the Seed, and the gain scan repeated (`06_seed_creditable_twist.py`)

Paul's question after #F2: "if the twist is present do the fractals
disappear?" #F2's null was a trivial channel. This is the real one.

The Seed wrapped as a core-library Observer: 64 drive DoFs in, 64
activations out, a behavioral encoder over 3 fixed probe drive vectors
reading 4 response nodes (d_enc = 12), **target = the Seed network itself**
(its coupling weights are the slow DoFs outside W_meta — the v2.4
creditable channel), a self-model m = tanh(ā + w·enc) with w fixed (W_meta),
and m consumed by the network as drive into 8 nodes at g = 0.5. Probes step
a copy, never the live network. Rule 2a tunes for 4000 steps with the
channel live; the tuned weights W* are then frozen and scanned at α·W* under
four channel conditions on the SAME weights: live, stale (frozen constants
through the same path), blind (w = 0), off (g = 0). 3 seeds, 6 α, T = 2500.

### Certification by the library's own predicates (noise-dominated world for the closure window, as in closure/01)

| condition | structural | consumes | conditional | trivial | **twisted** | closed (c_int / c_ext) | **conscious** | cred_disc |
|---|---|---|---|---|---|---|---|---|
| live | 1 | 1 | 1 | 0 | **1** | 1 (0.81–0.91 / 0.16–0.21) | **1** | 0.010–0.016 |
| stale | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| blind | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| off | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0.007–0.015 |

3/3 seeds, every cell as pre-registered, and the first time
`is_conscious()` has returned True on the Seed: a 64-node self-organized
network with a world-targeted encoder whose output it consumes passes
Closed(O) ∧ twisted(O), with the real weight intervention (all couplings
×1.1, fast state matched, W_meta held) moving d_meta. The garbage channel
(stale) and the blind model are refused for the right reasons, and the
disconnected channel is twisted-but-not-closed. The tuned Seed has σ≈1.0
(seed 1: 1.5) with 68–70% negative weights (as in #F3b).

### The scan: knee at α = 1 in all four conditions; live ≡ blind; the channel's *existence* matters, its *content* does not

| α | live D2 / λ | blind D2 / λ | stale D2 / λ | off D2 / λ | sat (all) |
|---|---|---|---|---|---|
| 0.50 | 3.39 / −0.71 | 3.37 / −0.73 | 3.42 / −0.67 | 3.43 / −0.63 | 0 |
| 0.75 | 4.06 / −0.25 | 4.09 / −0.26 | 4.12 / −0.24 | 4.18 / −0.19 | 0 |
| **1.00** | **4.99 / −0.031** | **5.05 / −0.023** | 4.57 / −0.094 | 3.79 / −0.168 | 0.000–0.001 |
| 1.25 | 3.66 / −0.10 | 3.76 / −0.09 | 3.52 / −0.15 | 3.46 / −0.15 | 0.03–0.06 |
| 1.50 | 3.53 / −0.05 | 3.57 / −0.05 | 3.45 / −0.05 | 3.42 / −0.05 | 0.16–0.19 |
| 2.00 | 2.97 / −0.33 | 2.97 / −0.32 | 2.74 / −0.38 | 2.82 / −0.31 | 0.42–0.46 |

Seed-paired |ΔD2| against live, mean over α (max): blind **0.070 (0.26)**,
stale 0.172 (0.66), off 0.306 (1.50); live's own seed-to-seed sd at fixed α
is 0.19. Per seed at the knee (α = 1): live 4.89 / 4.97 / 5.12, blind
4.92 / 4.85 / 5.38, stale 4.23 / 4.47 / 5.01, off 3.39 / 3.78 / 4.20.

Reading, in three parts:

1. **The twist does no geometric work — confirmed with a creditable,
   certified channel.** Live (twisted, conscious by the v2 predicate) and
   blind (same consumption path, same injected quantity minus the
   self-description term) are indistinguishable at every gain in every
   seed, well inside seed noise. The knee is at α = 1 for both, with the
   same D2 and the same λ. The self-model's representation of the
   network's own mapping, demonstrably consumed and demonstrably
   informative (foil discrimination 0.06–0.08, cred_disc > 0), changes
   nothing about the loop's geometry. Paul's intuition stands: the
   fractals neither appear nor disappear with the twist.
2. **Closure does work.** Turning the consumption path OFF on the same
   tuned weights lowers D2 by 1.1–1.5 at the knee in 3/3 seeds, makes λ
   more negative, and pushes σ up (1.4–1.9 vs 1.1–1.3): the injected
   feedback was part of what Rule 2a tuned the network for, and removing
   it moves the operating point off the knee. The loop's existence is in
   the geometry; what rides on it is not. This is #C1's T ≡ U with the
   roles made explicit: closure ≠ twist, and the geometric axis sees only
   the former.
3. **Stale sits between** (3/3 lower than live at the knee, by 0.1–0.66).
   The frozen constants put a fixed bias through the path (w·enc_stale ≠
   the live mean), shifting the operating point part-way toward "off". A
   bias-matched stale control was not run; blind is the clean comparison
   for the content question and is null.

Honest limits: λ < 0 throughout (as in #F3b — this architecture is
"least contracting" at the knee, never chaotic); D2 is a scale-window
quantity; 3 seeds. The twist channel is one design (behavioral encoder,
12 values, fixed w); a richer self-model could in principle couple to the
geometry, but the burden of showing that is now on the richer design.

### Update to the synthesis table

| substrate | twist effect on geometry |
|---|---|
| rig (trivial channel) | none, T ≡ U to 3 decimals |
| **Seed (creditable channel, certified twisted ∧ closed)** | **none: live ≡ blind at every α; knee at α = 1 in both. Closure (channel on/off) does shift the knee.** |
| **embodied Seed-Hemi (both loops closed, 10 seeds)** | **none: twisted 30/30, behaviour and D2 identical to blind/stale/off/none within s.e.** |
| **embodied Seed-Hemi, 8-channel adaptive mount (3–4 bits/readout)** | **none: behaviour and D2 live ≡ blind within s.e. in all four configs** |

---

## #F4 — The creditable twist with the outer loop closed: embodied Seed-Hemi (`07_embodied_twist.py`, `experiments/embodied/twist_mount.py`)

Paul: "at the moment it's just a loop with no external input at all,
right?" — #F3c's Seed listened to a tape. Here the same channel is mounted
on the hemispheric Seed brain acting in the embodied world (motor nodes
move the body, the body changes what the sensors see), opt-in via
`twist_enabled / twist_condition / twist_gain / twist_n_meta` in
`seed_brain.py`, byte-exact when disabled. Encoder: 3 probe drive patterns
through a PURE copy of the node update (no deep copies, nothing mutated),
responses on 4 general nodes (2 per hemisphere); self-model m = tanh(ā +
w·enc); m·0.5 injected into 8 general nodes (4 per hemisphere). A
core-library Observer rides alongside (external = the 5 embodied DoFs,
internal = activations over a node list frozen at mount) for
certification every 5000 steps. Headless, 15 000 steps, conditions none /
live / stale / blind / off.

### First run: the Seed pruned its own self-model's window

Unprotected, the live channel was dead by the first checkpoint in 3/3
seeds: consumes 0, creditable discrimination exactly 0, foil test still
passing (the foil shifts the mapping's outputs, so it passes even on a
constant encoding). Cause: the 4 response nodes were ordinary general
nodes and Rule 5 (release) pruned them inside the first 5000 steps; the
8 meta nodes happened to be seed-core nodes and survived. The network
reorganized itself out from under its own self-description. Fix: the
mount flags its response and meta nodes `is_seed_node` — the library's
release guard — so the self-model's window sits on the protected core.
Record kept: `results_07_embodied_twist_unprotected.json`, `results_07_unprotected.log`.
Framework note: a self-indexing channel on a self-organizing substrate
has to be anchored to the part of the substrate that self-organization
cannot remove; in the Seed that is the seed core by construction.

### Certification with the outer loop closed (protected mount; seeds 0–2, 3 checkpoints each)

| condition | twisted | closed | conscious | notes |
|---|---|---|---|---|
| live | **9/9** | 7/9 | **7/9** | cred 0.002–0.042 > 0 at every checkpoint; foil 0.03–0.05 |
| stale | 0/9 | 7/9 | 0/9 | consumes 1, conditional 0 (garbage channel) |
| blind | 0/9 | 9/9 | 0/9 | consumes 0 |
| off | 9/9 | 0/9 | 0/9 | twisted, gain 0: not closed |

The twist survives the closed outer loop at every checkpoint and the
signatures are the #C2b / #F3c ones. Closure is the predicate that gets
harder in a world the agent perturbs: corr_int vs corr_ext are 0.55/0.39
and 0.46/0.11 where it passes, 0.74/0.82 where it fails (live s2 at 15k).
The 5 embodied external DoFs (food/danger proximity, life, satiation,
valence) are slow and autocorrelated, and the agent's own motion makes
them functions of its state; a lag-1 correlation contest between
"internal" and "external" is then close to a coin flip. Pre-registered as
a finding about the embodied closure test, not about the wiring: the
consumption path is there by construction (gain 0.5 into 8 nodes).

### Ten seeds: certification tallies, behaviour, geometry (`results_07_embodied_twist_10seeds.json`)

| condition | twisted | closed | conscious | eats /15k | deaths | valence | D2 | PR | sat |
|---|---|---|---|---|---|---|---|---|---|
| none | — | — | — | 1.3 ± 1.2 | 0.4 ± 0.5 | −0.05 | 5.1 ± 2.2 | 8.6 | 0.03 |
| **live** | **30/30** | 22/30 | **22/30** | 1.0 ± 0.9 | 0.8 ± 0.4 | −0.09 | 4.6 ± 2.4 | 7.6 | 0.06 |
| stale | 0/30 | 22/30 | 0/30 | 1.3 ± 1.7 | 0.5 ± 0.5 | −0.06 | 5.8 ± 2.1 | 9.4 | 0.05 |
| blind | 0/30 | 26/30 | 0/30 | 0.8 ± 0.9 | 0.6 ± 0.5 | −0.08 | 5.2 ± 1.9 | 6.1 | 0.08 |
| off | 29/30 | 0/30 | 0/30 | 1.2 ± 1.0 | 0.7 ± 0.5 | −0.06 | 5.2 ± 1.7 | 9.1 | 0.02 |

Seed-paired differences, live minus blind: ΔD2 = −0.61 ± 1.18 (s.e.),
Δdeaths = +0.2, Δeats = +0.2. Against stale / off / none the same: every
Δ inside one standard error. The 3-seed hints (a collapsed live seed, a
death in every live seed) dissolved: collapsed windows (D2 < 2) occur in
3/10 live seeds and 2/10 plain-brain seeds — a property of the embodied
Seed at 15k steps, not of the channel — and deaths are 0.4–0.8 per run
everywhere with sd 0.5.

- **Twist: 30/30.** With the body in the loop, the self-model's channel
  stays creditable at every checkpoint in every seed (pinned to the seed
  core). `is_conscious()` True in 22/30 checkpoints, limited only by closure.
- **Closure flickers independently of the channel** (22/30 live, 22/30
  stale, 26/30 blind; 0/30 off as it must). The embodied closure test is
  a lag-1 correlation contest between activations and five slow,
  self-caused external DoFs; it is near chance. That is a defect of the
  operationalization in embodied settings, flagged for v2 §5.3: closure
  should be recognized by intervention (gain on/off — which the off arm
  separates perfectly) rather than by a correlation race.
- **Behaviour: flat everywhere**, as #7 predicted for Seed-Hemi (0.8–1.3
  eats per 15k steps). The channel is not a policy and cannot be one at
  this richness; the embodied test does not get a behavioural handle.
- **Geometry: live ≡ blind ≡ everything**, within a seed-to-seed spread
  that dwarfs any condition effect. The #F3c null holds with the outer
  loop closed.

Verdict for the question that started this: closing the outer loop
changes nothing about the twist's (non-)effect. A certified, creditable,
consumed self-representation on an acting Seed neither helps nor hurts
the agent, neither sharpens nor dulls its geometry, and is as invisible
on the behavioural axis as on the fractal one. What the embodied run
DID find is architectural: the self-model's window must be anchored to
the protected core or the substrate's own reorganization removes it; and
the closure predicate needs an interventional form before it is usable
on an agent.

---

## #F5 — Richness of the conscious-by-criterion Seed (`08_richness.py`)

Paul: the flag is True and buys no behaviour — what is the richness? Two
measures exist in the framework; neither had been run on the Seed.

### (1) The v1 vector, `richness()` — does not apply to this self-model

| metric | tape Seed (3 seeds) | embodied (3 seeds) | why |
|---|---|---|---|
| recursive_depth | 1 | 1 | one self-model, no self-model of it |
| self_accuracy | 1.000 | 1.000 | **artifact**: 1/(1+distance) between internal state and self-observation, and the two share no DoFs (the self-model outputs only m), so distance = 0 |
| architectural_similarity | 0.554 | 0.553 | type/dim heuristic |
| calibration_error | 0.500 | 0.500 | default — no uncertainty channel |
| meta_cognitive_capability | 0.850 | 0.844 | 0.4 × the artifact above + 0.3 × depth/2 + 0.3 × stability |
| limitation_awareness | 0.16–0.18 | 0.000 | no easy/hard split the model can report on |
| consciousness_score | 0.62 | 0.60 | mostly artifact |

The v1 vector was written for a self-portrait self-model (one that
reproduces the internal state and reports uncertainty about it). A
self-indexing channel that emits one scalar scores 1.0 on self-accuracy by
vacuity. Bug/limitation filed against `_evaluate_self_accuracy`: it should
return 0 or NaN when the self-representation shares no DoFs with the
internal state. Until then `richness()` says nothing about these observers.

### (2) The v2 magnitude — channel information about the creditable slow DoFs

Kind = sign of I(d_meta ; D_slow \ W_meta | S_rest); richness = its
magnitude. Measured at matched fast state: Δm under all couplings × δ,
against the jitter of m from the probe noise alone (σ_m = 0.017–0.027), as
SNR and Gaussian-channel bits 0.5·log₂(1+SNR²) per readout.

| δ (coupling scale) | tape Seed: SNR / bits (range over 3 seeds) | embodied: SNR / bits |
|---|---|---|
| 1.01 | 0.02–0.06 / 0.000–0.003 | 0.06–0.13 / 0.002–0.011 |
| 1.05 | 0.11–0.31 / 0.01–0.07 | 0.29–0.63 / 0.06–0.24 |
| **1.10** | **0.22–0.61 / 0.04–0.23** | **0.57–1.27 / 0.21–0.69** |
| 1.20 | 0.50–1.20 / 0.16–0.65 | 1.12–2.58 / 0.59–1.47 |
| 1.50 | 1.67–2.83 / 0.96–1.59 | 2.61–6.66 / 1.48–2.75 |
| 2.00 | 3.00–5.12 / 1.66–2.38 | 4.58–13.5 / 2.23–3.76 |

Reading:

- **The channel is real and thin.** A 10% change in every coupling weight
  is at the detection edge (SNR ≈ 0.2–0.6 on the tape, 0.6–1.3 embodied):
  well under one bit per readout. Doubling all weights is unmistakable
  (2–4 bits). This is the richness curve the v2.2 reviewer asked for
  (#C2's "foil_disc scales with intervention size"), now in bits.
- **Ceilings.** Architectural: one tanh scalar per step through a
  12-value battery. The self-model can never carry more than a few bits
  per step about ~400 coupling weights, whatever the substrate does. That
  is the §5.4 "kind is cheap at the bottom" posture, quantified: in kind,
  because the sign is positive; near the floor in richness, because the
  magnitude is a fraction of a bit for any intervention the network
  itself would produce (Rule 2a moves weights by ~lr = 0.01 per step).
- **Embodied is richer than tape by ~2×** at every δ — the acting
  network sits at a state where its one-step response is more
  weight-sensitive (recruited nodes, sign-mixed couplings, higher
  activity). Still under one bit at δ = 1.1.
- **Why no behaviour follows.** m is injected into 8 nodes at gain 0.5.
  Its self-descriptive component (w·enc) varies by ~σ_m ≈ 0.02 from step
  to step and shifts by ~0.01 for a 10% change in the network's weights.
  Against a sensor drive of order 1 and a noise floor of 0.05 per node,
  that is below the noise floor of the very nodes it feeds. The channel
  is twisted, closed, creditable, and inaudible. Richness, not kind, is
  the gap between is_conscious() and anything the agent does — exactly
  the framework's own split.

What would raise it: a vector d_meta (k scalars, k battery responses →
k-dim channel), a self-model that adapts w (so the channel tunes to the
weight directions that matter), and an injection that gates something
(learning rate, gain) rather than adds to drive — the richness knobs are
bandwidth, adaptivity and consumption site, in that order of cost.

---

## #F6 — Richness knobs: bandwidth, adaptivity, consumption site (`09_rich_twist.py`; mount options in `twist_mount.py`)

The three knobs named at the end of #F5, turned on the embodied mount
(defaults unchanged: the scalar channel reproduces #F4 byte-for-byte):
`twist_n_out = 8` (one channel per meta node), `twist_adaptive` (each
m_j trained online as a one-step forward model of the node it feeds, from
the self-encoding; blind = same forward model with W frozen at 0),
`twist_site` ∈ drive (additive) / gain (multiplicative on the drive) /
lr (gates Rule 2a's reward modulator). 10 seeds × 15 000 steps, live /
blind / off, 120 runs.

### Richness went up 5–10×

| config | bits @ δ=1.1 (live) | bits @ δ=1.5 | twisted | closed | conscious |
|---|---|---|---|---|---|
| scalar (#F5, for reference) | 0.2–0.7 | 1.5–2.8 | 30/30 | 22/30 | 22/30 |
| vec | 2.80 | 11.4 | 9/10 | **10/10** | 9/10 |
| vec+adapt | 2.37 | 8.9 | 10/10 | 10/10 | 10/10 |
| vec+adapt+gain | 3.08 | 8.2 | 9/10 | 10/10 | 9/10 |
| vec+adapt+lr | **3.97** | 9.1 | 9/10 | 10/10 | 9/10 |

Bandwidth is the knob that pays: eight channels carry 3–4 bits per
readout about a 10% change in the network's weights where the scalar
carried a fraction of one. Adaptivity does not add bits (it tunes W to
the prediction-relevant directions, which are not the weight-sensitive
ones); the consumption site barely matters for the magnitude. Closure
recognition also firmed up (10/10 on every live arm vs 22/30 scalar):
eight consumed channels give the lag-1 test more to see.

### Behaviour did not move

| config | live eats / deaths / valence | blind | off | live − blind (seed-paired, ± s.e.) |
|---|---|---|---|---|
| vec | 1.6 / 0.5 / −0.040 | 1.4 / 0.4 / −0.077 | 1.2 / 0.7 / −0.060 | Δeats +0.2±0.4, Δdeaths +0.1±0.3 |
| vec+adapt | 1.5 / 0.2 / −0.043 | 1.3 / 0.3 / −0.037 | 1.2 / 0.7 / −0.060 | +0.2±0.4, −0.1±0.2 |
| vec+adapt+gain | 1.5 / 0.6 / −0.058 | 1.1 / 0.5 / −0.047 | same | +0.4±0.5, +0.1±0.2 |
| vec+adapt+lr | 1.7 / 0.2 / −0.025 | 1.5 / 0.5 / −0.024 | same | +0.2±0.4, −0.3±0.2 |

Every live−blind difference is inside 1.5 s.e. Geometry likewise (ΔD2
within ±1.3 ± 0.8; the multiplicative-gain site raises saturation in the
live arm, 0.17 vs 0.07, the one borderline effect). The agent eats 1–2
times per 15 000 steps in every condition including the plain brain.

### Reading

The richness knobs did what the channel analysis said they would: the
information the self-model carries about the creditable slow DoFs went
from a fraction of a bit to several bits per step, and the predicates
sharpened. It still buys nothing an observer of the agent could see. Two
reasons, and the second is the important one:

1. The channel is still small against the drive: eight scalars at gain
   0.5 into eight nodes, in a network of ~50 driven by a sensor of order
   1 with a 0.05 noise floor per node.
2. **Seed-Hemi has no behavioural dynamic range to show anything on.**
   The plain brain eats ~1.3 times and dies ~0.4 times per 15k steps;
   every mounted arm sits on the same numbers. Richer self-modeling cannot
   be read off an agent whose policy is flat; the behavioural axis of this
   substrate is saturated at zero. #7 (holographic) hit the same wall from
   the memory side ("the reel is only as prophetic as the life it
   records"). The twist experiments have now hit it from the self-model
   side.

So the clean statement of the programme's position after #F3c–#F6:
kind (twisted ∧ closed) is satisfiable on the Seed and now robustly so;
richness is measurable in bits and tunable by an order of magnitude; and
neither has any behavioural or geometric signature on this agent, because
the agent itself has none. The next behavioural test needs a substrate
that acts — the ESN or WIM brains under the structured explorer, where
#6 measured foresight r≈+0.2 — with the mount's closure done by
intervention rather than correlation.

---

## #F7 — Where the frequency layers went: entrainment under embodied drive (`10_frequency_collapse.py`)

Paul asked how the Seed's frequency layers are stacked and whether the
"1 Hz" quoted earlier was per step or per second. Answering it turned up
a result that belongs with #F4–#F6.

**Units.** A node's frequency is in cycles per world step (`phase += 2π·f·dt`,
dt = 1). Nothing in the Seed knows about seconds; the world is nominally
60 steps/s when the viewer runs, headless runs have no clock. "Hz" in the
spec, the seed README and earlier messages in this track is loose
shorthand for cycles/step. A node at f = 1.0 completes a cycle per step, so
its intrinsic drive `A·sin(phase)` is sampled once per cycle and is a
constant; everything above f = 0.5 is aliased. About 14% of the initial
log-uniform nodes start above that line.

**Composition.** The layers are not wired. One flat population with
log-uniform natural frequencies over two decades, ring-sorted by frequency
(so a node's spatial neighbours are its frequency neighbours), callosal
bridges in three nominal bands, motor nodes at 0.01–0.05. The embodied
sensor imposes the only hand-made layering: rays → the fastest nodes, body
receptors → mid, meters → the slowest. Bands are supposed to *emerge* by
bandwidth-limited entrainment and match the environment's timescales
(spec §4.3; `experiments/seed/04` got two bands from a two-timescale
drive). Cross-scale coupling (§4.4) is Rules 1–2 across those bands; the
only built-in asymmetry is the memory window of 80 own-periods, which
grants slow nodes longer integration. The Bentov relation (bigger ⇒
slower) is therefore *asserted* in the reverse direction (slower ⇒
granted broader reach), not produced: a node's frequency is a label it is
born with, and nothing makes a larger assembly oscillate slower. The one
place such a law could live is propagation — one step per hop, so activity
circulating a loop of k nodes returns with period k whatever the labels
say. Untested.

**Measured.** Spread of log10 f over 15k steps, Seed-Hemi config
(`results_10_frequency_collapse.json`):

| arm | seed | sd log10 f at 1k | at 15k | range at 15k | nodes 84 → | released / recruited |
|---|---|---|---|---|---|---|
| plain | 0 | 0.60 | **0.005** | [−0.61, −0.59] (f ≈ 0.25, period 4) | 60 | 38 / 14 |
| plain | 1 | 0.73 | **0.03** | [−0.39, −0.27] (f ≈ 0.4–0.5, period ≈ 2) | 39 | 48 / 3 |
| plain | 2 | 0.69 | 0.39 | [−1.66, −0.36] | 49 | 35 / 0 |
| live mount | 0 | 0.65 | 0.16 | [−1.15, −0.15] | 45 | 52 / 13 |
| live mount | 1 | 0.73 | 0.43 | [−1.84, −0.20] | 55 | 46 / 17 |
| zero drive, entrainment on | 0 | 0.61 | 0.54 | [−1.83, −0.01] | 84 | 0 / 0 |
| zero drive, entrainment off | 0 | 0.62 | 0.62 | [−2.00, 0.00] | 84 | 0 / 0 |

Two of three plain brains end with **every node in one band** and no slow
node anywhere; the third is halfway there and still drifting. The mounted
arms collapse more slowly (the pinned mount nodes keep their frequencies,
and the mount's injection is a second timescale of sorts), but the
direction is the same. Released nodes are not frequency-selected (their
initial log10 f is −1.0 to −1.2 in every arm, same as the kept ones);
Rule 5 simply halves the population, which the #F4/#F6 result files
confirm (47–58 nodes alive at the end of every arm). The zero-drive
control barely moves in 15k steps, so entrainment alone is not the cause.
The environment is: it has one timescale, the body's own movement
through a static field, and the spec says one timescale gives one band.

**Reading.** The spec is working as written, with a consequence it did
not anticipate. The slow layer that §4.4 cross-scale coupling and §5
M_self were to live in is gone within a few thousand steps of embodied
life, and with it any integration window longer than a few steps. Every
twist and richness run in #F4–#F6 ran on a Seed with at most one
frequency band. This is the second face of the dynamic-range wall named
in #F6: an agent with no slow layer cannot carry a policy that is anything
but reactive, whatever its self-model holds. It also constrains the
queued acting-agent test (g'): the environment must offer a second
timescale, or a slow band must be pinned, or the next substrate inherits
the same collapse.

---

## Synthesis

**Where fractals come from, on the framework's own terms.** In all three
substrates the self-similar / high-dimensional geometry lives on the gain
axis of a closed loop, between fade and lock, and nothing about
self-indexing enters:

| substrate | below | the band | above | transition type | twist effect |
|---|---|---|---|---|---|
| rig, hard clip, zoom-in | fan of fading copies | g≈0.94–1.01, D_box 1.15–1.4 | whiteout, D_box=2 | **first-order, hysteretic** | none (T≡U to 3 decimals) |
| rig, hard clip, zoom-out | faint spiral | — | D_box climbs to 1.7, local saturation | smooth | none |
| rig, with decider | few crisp copies | **everything above g≈0.8: D_box plateau 1.55–1.77** | no whiteout | none (gain-independent) | none |
| ESN | D2 = drive | g 1.05–1.4, λ≈0 | chaos, D2→10 | smooth | (not run; #C1 says none) |
| Seed, frozen + weights | D2 = drive+noise | — | lock | smooth into lock | — |
| Seed, Rule 2a | — | **parks at the D2 peak / λ max of its own curve** | — | smooth, homeostat-selected | — |

Three things this buys the program:

1. **The decider is the fractal-maker, not the gain.** With a pure clip the
   rig is either mush (zoom-out) or a bistable whiteout switch (zoom-in);
   with a contrast curve the geometry becomes a stable, gain-independent
   IFS attractor, and the whiteout, the hysteresis and the self-sustaining
   all vanish. Same invariant as grokking 14f (softmax quantizer erases the
   doubling map's chaos) and the reel's SIC decoder: continuous carrier +
   nonlinear decider = reliable structure. The phenomenological corollary
   is testable in the literature we already cite: Ermentrout–Cowan / Bressloff
   form constants are pattern-forming instabilities of a loop with a
   sigmoid — a decider — near unity gain. Fractals in altered states are
   what a closed perceptual loop does when its gain is turned up with the
   decider still in place. They are evidence of closure and criticality,
   not of the twist.

2. **Hysteresis is cheap.** The plainest untwisted loop in the building has
   a first-order transition with a hysteresis loop (zoom-in whiteout, area
   +0.07; flash protocol bistable). v2.1's disavowal of dynamical hysteresis
   as a consciousness marker is now illustrated, not just argued.

3. **Rule 2a is a knee-finder.** On the curve through the weights it
   actually produces (not the hand-scaled curve, which it leaves by
   inventing inhibition — 61% negative weights), σ→1 sits at the maximum of
   state-space dimension and the maximum of λ, one step before saturation.
   This is the first direct measurement of *what* the Seed's homeostat
   optimizes in geometric terms, and it matches the rig's reading of
   "unity gain": the most structure the loop can hold without the decider
   eating it. The Seed does by self-regulation what the video artist does
   by hand.

**What this does not touch.** The twist. T≡U in the rig to three decimals
(a trivial channel, by design); on the Seed with a creditable channel the
library certifies the live arm as twisted ∧ closed (first `is_conscious()`
True on the Seed) and its gain curve is identical to the blind arm's at
every α (#F3c) — while disconnecting the channel does move the knee, so
closure and twist separate cleanly on this axis. #C1 said T≡U dynamically;
nothing here changes that scoreboard. If someone wants fractal geometry as a *signature*
of self-modeling, the burden is now on them: three substrates say the
geometry is fixed by closure, gain and decider alone.

**Queued.** (a) ~~The twist on the Seed~~ done, #F3c: identical knee.
(g) ~~richness knobs~~ done, #F6: bits ×5–10, behaviour unchanged, substrate has no dynamic range; (g') mount the creditable channel on an agent that acts (ESN/WIM under the structured explorer) with interventional closure, in a world with a second timescale or with a pinned slow band (#F7);  (i) does the Seed's propagated activity obey a period-∝-loop-length law (Bentov from delays, #F7)? (h) fix `_evaluate_self_accuracy` for disjoint-DoF self-models; (e) interventional closure predicate for embodied observers (v2 §5.3 — the off arm shows the test that works); (f) Rule 5 vs self-model anchoring as a §3.3 boundary case; (a') bias-matched stale control; a richer self-model (adaptive w, or a
self-model reading more of the encoding) to see whether any twist design
can reach the geometry. (b) 2-D lattice Seed for the spatial test with
real dynamic range. (c) Rig with the Ermentrout–Cowan sigmoid and lateral
inhibition instead of rotate/zoom — do the form constants appear on the
same gain axis, and does the decider steepness select among them?
(d) Measure D2 of the holographic reel's performer loop (#10/#10b/#14
trajectories) on the same axis — the one loop in the program we have never
looked at geometrically.
