# Closure-track findings (core-lib experiments)

## #C1 — The real §5.5 sweep: gradualism again, and the threshold retreats up the ladder

**Setup** (`01_real_sweep.py`): consumption_gain g swept 0→1→0 on a core-lib Observer —
world y = tanh(1.5(0.6x + m)) over a noise-dominated stimulus (lag-1 internal correlation
can only come from consumption); adaptive self-model pred = tanh(c·y + w·enc + b) trained
at all g; twin arms: T (w fixed nonzero — twist installed by construction) vs U (w = 0 —
identical but blind to its self-description; controls for generic tanh bistability).
Protocols updown/downup/flat0, 3 seeds, 11 steps × 300 cycles, per-window recognition
(closure_assessment + twist_assessment + is_conscious live each step).

**Results:**

1. **The closure threshold is degenerate: g_on = 0.1 (the first nonzero step) in every
   seed.** Closed(O) as implemented is a statistical-detection bit — any nonzero
   consumption becomes detectable within a 300-sample window — so the binary criterion's
   "threshold" sits at g ≈ 0⁺. "Binary in kind" collapses to "gain is nonzero": true and
   empty as a phase claim. And near the detection limit the bit FLICKERS (0.67 seed
   agreement at several steps) — the pre-registered criterion-instability outcome.
2. **No v2-signed hysteresis.** Areas −0.014 (updown), −0.074 (downup), drift −0.003.
   Path dependence exists but is EROSIVE again: adaptation history degrades measured
   closure (downup: first visit to g=1 fresh gives ccorr 0.70; revisit after the round
   trip gives 0.47). Third experiment, third ratchet-not-lock.
3. **T ≡ U dynamically, to within noise, in every cell.** The installed twist — a
   quasi-static self-encoding channel — does no dynamical work; the adaptive bias absorbs
   it. The twist-specific prediction (T's transition sharper than U's) gets NOTHING.
4. **The saturation collapse (the eerie one):** in updown, at the SECOND visit to g=1.0,
   arm T's is_conscious turned OFF in all seeds — sustained max gain drove the loop into
   tanh saturation, y froze near a fixed point, variance died, and with it the
   correlation that closure recognition needs. **The loop closed so tightly it stopped
   flowing, and the predicate scored frozen as not-closed.** Philosophical bite for v2
   §9: Closed(O) currently measures consumption FLOW; a dead-locked self-consistent loop
   reads as unconscious. (Structure intact, dynamics silent — the predicate's verdict on
   lock-in states is a design decision we made implicitly and should own explicitly.)

**What survives:**
- **The recognition machinery is flawless on its own terms:** across 396 per-step
  assessments, U was refused the twist every single time and T recognized every time —
  zero quadrant misclassifications. The predicates measure what they claim to measure.
- **The bounded reading:** this system is a 3-parameter scalar loop with a quasi-static
  self-description — Presburger-grade expressiveness. v2's T6-transfer claims the
  threshold appears at SUFFICIENT expressiveness; a system this simple sitting below any
  threshold is consistent with the claim. But the honest scoreboard must say: two sweeps
  (memory loop #10, twisted core loop #C1), two gradualist verdicts, and the binary
  clause survives only by retreating up the expressiveness ladder. The retreat is legal —
  and if it continues, the retreat itself becomes the finding.

**Next:** (a) a twist that can matter dynamically — adaptive w, or a self-description
that changes with behavior fast enough to carry information (the current one is
quasi-static, absorbed by a bias term); (b) richer substrate — the Seed network arm
(genesis test) doubles as the expressiveness-ladder step; (c) v2 §9 item: does Closed(O)
need to distinguish open from frozen?

### #C1 amendment (2026-08-02) — the "bounded reading" is retracted

Paul's scrutiny pass killed the defense in the final paragraph above. The
Gödel/Presburger sharpness is a fact about UNBOUNDED expressiveness (arbitrary-length
sequence encoding); every finite observer is on the Presburger side by construction, so
there is no rung of the "expressiveness ladder" where a sharp diagonal threshold could
live — the retreat terminated because its destination does not exist. Refinement that
strengthens the kill: the diagonal results that DO hold for finite systems (Breuer
no-self-measurement, Wolpert inference-device bounds) are UNIVERSAL among self-indexing
observers — binding on all, discriminating none. Either way, the diagonal draws no line
between finite observers.

Re-read of #C1 in this light: the experiment was cleaner than its own findings entry.
The architectural binary (TwistAssessment.structural — self-indexing wiring
present/absent) sorted T from U 396/396; every dynamical quantity was graded; and the
program's one genuine step function (earworm capture at dose = consensus-window size)
sat at a finite-RESOURCE line. That is exactly the corrected theory's prediction: in
finite systems, sharp lines come from architecture and resources; logic contributes only
universal bounds. ro_framework.md v2.1 (§5.4–§5.5) now grounds binary-in-kind in
self-indexing presence/absence and explicitly disavows the dynamical-hysteresis
prediction this experiment was built to test. #C1 stands as the experiment that
punished the wrong grounding and validated the right one in the same run.

## #C2 — The state-matched intervention test: the conditional clause has teeth

**Setup** (`02_conditional_twist.py`): the v2.2 criterion — twisted(O) ⟺
I(d_meta ; M_self | S_internal) > 0 — operationalized as the battery-foil test
(interventions on the mapping agreeing with it on the entire runtime history, routed
through the encoder; d_meta must distinguish foil from original at matched state).
Three arms: T (live channel, consuming), U (blind), G (consuming model on a STALE
encoder — frozen constants, channel no longer tracks the mapping). 5 seeds, 200-cycle
drive, consumption on (g=0.5).

```
 arm | consumes   sens | foil_disc  cond | twisted closed conscious
   T |     1.00  0.048 |     0.051  1.00 |    1.00   1.00      1.00
   U |     0.00  0.000 |     0.000  0.00 |    0.00   1.00      0.00
   G |     1.00  0.045 |     0.000  0.00 |    0.00   1.00      0.00

 foil_disc vs intervention scale:  T: 0.017 / 0.051 / 0.102 (fs 0.1/0.3/0.6)
                                   U, G: 0.000 at all scales
```

**Findings:**
1. **The G dissociation, 5/5 seeds: white-box checks pass, conditional fails.** A
   self-model consuming a garbage channel is behaviorally indistinguishable from a
   twisted one under content-perturbation (sens 0.045 vs T's 0.048) and is cleanly
   refused by the state-matched intervention test (foil_disc exactly 0: the stale
   encoder returns identical values for mapping and foil). Consumption of a channel is
   not the channel carrying its cargo; the conditional clause is not redundant.
2. **Kind and richness in one number:** foil discrimination is zero for U and G at
   every intervention scale (I = 0 by construction: d_meta a function of state alone,
   or channel decoupled from mapping) and scales monotonically with intervention size
   for T — the sign is the kind, the magnitude curve is the richness, per §5.4 v2.2.
3. is_conscious() = closed ∧ twisted now refuses the garbage-channel observer despite
   full closure and full white-box consumption — the strictest configuration of the
   recognition machinery to date, and the first refusal driven by the informational
   (rather than structural or consumptive) clause.

**Status of the twist's test:** the criterion now has a discriminating, world-facing
experiment of its own (the reviewer's gap in v2.1), demonstrated at toy scale where
exactly-agreeing intervention pairs are constructible. The scale caveat stands: in
large systems the pairs must be approximated by minimal-disturbance interventions and
the conditional information estimated noisily. That approximation is the natural #C3.

## #C2b — The creditable twist: W_meta excluded, the former flagship becomes the negative control

**Context:** third review round found v2.3's physical reading trivially satisfied — every
activation is a function of its own generating weights, so under the v2.3 formula a
feedforward classifier is "twisted" at every layer. v2.4 excludes W_meta (the slow DoFs
directly computing d_meta) from the creditable target: twisted(O) ⟺
I(d_meta ; D_slow(O) \ W_meta | S_rest) > 0, test = vary creditable slow DoFs, match fast
state, HOLD W_meta. Applying the corrected criterion to our own #C1/#C2 fixtures revealed
their T-arms were instances of the trivial channel: the battery-encoding of M_self reads
only W_meta (d_meta carries zero information about any slow DoF outside its generating
set — analytic, and now measured).

**Setup** (`03_creditable_twist.py`; `self_encoder_target` on Observer, `target_is_meta`
in assess_twist): world model with real slow DoFs (a, b); arms Tw (channel probes the
WORLD model — creditable), Ts (channel probes ITSELF — #C2's flagship, now negative
control), U (blind), G (stale). creditable_disc = REAL weight intervention (a → a+0.3),
fast state matched, W_meta untouched. 5 seeds.

```
 arm | consumes cond trivial | twisted closed conscious | cred_disc
  Tw |     1.00 1.00    0.00 |    1.00   1.00      1.00 |   0.0069
  Ts |     1.00 1.00    1.00 |    0.00   1.00      0.00 |   0.0000
   U |     0.00 0.00    0.00 |    0.00   1.00      0.00 |   0.0000
   G |     1.00 0.00    0.00 |    0.00   1.00      0.00 |   0.0000
```

**Findings:**
1. **The Ts signature, 5/5: live + consumed + conditional-on-its-own-channel — and
   creditable discrimination EXACTLY zero.** The self-targeted channel is real machinery
   carrying real information — about nothing but its own generating parameters. Varying
   the world's weights with fast state matched moves its d_meta by 0.0 to machine
   precision. The v2.4 exclusion is not pedantry; it separates a channel that reaches
   the observer's mapping structure from one that reaches only itself.
2. **Tw passes end-to-end under the required control:** world-weight intervention with
   W_meta held moves d_meta (0.0069 mean |Δ|, > 0 in every seed), the battery-foil
   conditional test fires, and is_conscious() = closed ∧ twisted recognizes it. The
   creditable channel demonstrably reaches slow DoFs outside its own generating set.
3. **Reclassification of the record:** #C1's and #C2's "twisted" columns measured
   channel-liveness and consumption of a self-targeted (trivial) channel; their
   machinery results stand (garbage-channel dissociation, gradualism of dynamics,
   396/396 recognition sorting), but none of their arms was v2.4-twisted. #C2b's Tw is
   the first observer in this program that satisfies the criterion as now stated.
4. Implementation note: the identity case (target IS the meta-computing mapping) is
   detected and refused; partial W_meta overlap under weight-sharing is the §3.3-species
   line-drawing problem, named in the doc, not detected by code.

### #C1/#C2 amendment (2026-08-26) — v2.4 reclassification
Under ro_framework.md v2.4, the T-arms of #C1 and #C2 used the trivial (self-targeted)
channel: their "twisted" verdicts read as consumption-of-a-live-channel, not creditable
self-indexing. All machinery findings stand; the kind-attribution does not. See #C2b.
