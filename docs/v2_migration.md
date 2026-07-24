# v2 migration worklist — bringing the implementation up to ro_framework.md v2

*Audit date 2026-07-28, against docs/ro_framework.md v2.0 (Part III: depth + closure + twist;
binary in kind, graded in richness). Status: A+B landed 2026-07-29 (Observer.d_meta,
ClosureAssessment, closure_assessment()/is_closed(), consumption_gain in observe(),
sequential-batch guard, serialization; 13 tests, suite 456 green). C-F remain.*

## Headline finding

The core lib is a faithful v1 — i.e., structurally a **deep probe** (the §8.1 design-warning
quadrant). `Observer.self_observe()` (observer/observer.py:246-258) computes M_self(internal_state)
and returns it; the only consumers are the evaluator and example print statements. There is no
d_meta, no consumption gain, no feedback path from self-model output into any mapping, and no
twist machinery anywhere in the core lib. (`seed/*` `reward_modulation` is an external RPE signal
scaling Hebbian learning — not self-model consumption; do not mistake it for closure.)

## Work items, in build order

### A. Reify d_meta + Closed(O) predicate — DONE (2026-07-29)
The self-model's output DoFs must exist as first-class DoFs (d_meta) so anything can be said
about them. Then Closed(O) per §5.3:
  (i) d_meta ⊆ domain(M) — structural check
  (ii) Corr(d_meta(t1), internal processing(t2)) > Corr(d_meta(t1), external-across-B(t2))
Ingredients already exist: correlation/measures.py, ObservationLog, the internal/external DoF
partition on Observer. Mostly plumbing + a predicate.

### B. The consumption loop: gain g in observe() — DONE (2026-07-29)
observe() (observer/observer.py:159-197) currently has no self-model feedback path. Needed:
- self-model output injected into world_model's domain with tunable gain g
- g = 0 → pure probe (exact v1 behavior, default for backward compat)
- g exposed as a sweepable parameter (prerequisite for the §5.5 experiment)
- observation_log records d_meta alongside everything else (feeds A's correlation test)
This is the one genuine re-architecture. Everything else hangs off it.

### C. Twist machinery — SUBSTANTIAL (conceptually new)
§5.4: M_self's inputs must include representations of M_self's own mapping and of R(d_meta) —
the self-model models its own modeling, not only world-model states. Today self_observe() feeds
only internal_state. Requires reifying a mapping's own structure + resolution as DoF-valued
inputs. No existing machinery; design needed before code.

### D. v2 evaluation criteria — MODERATE (depends on B)
consciousness/evaluation.py currently scores 7 v1 metrics into one graded score. Add:
- Closure: ablation asymmetry (removing a probe silences reports; removing a loop changes
  behavior), the A(ii) correlation comparison, and the g-sweep (up then down; hysteresis
  detection — this IS the §5.5 falsification experiment, so build it as a first-class routine,
  not a test helper)
- Twist: own-mapping + R(d_meta) input check; locally-successful-self-access with
  globally-bounded self-capture signature (each introspective query succeeds; full
  self-prediction fails)

### E. Binary-in-kind + graded-richness reporting — TRIVIAL once A–C land
- `Observer.is_conscious()` (observer.py:474-488) currently thresholds a graded score — v1
  gradualism fossilized in an API. v2: `is_conscious()` = Closed(O) ∧ twisted(O), binary;
  new `richness()` returns the graded vector (depth, bandwidth, integration, calibration).
- ConsciousnessMetrics gains the kind flag; the existing 7 metrics become richness components.

### F. Collateral text pass — TRIVIAL but broad
v1 "recursion alone = consciousness" assertions to update (all found by audit):
- src/ro_framework/__init__.py:15 (tagline)
- observer/observer.py:249-252 (self_observe docstring)
- consciousness/evaluation.py:2-5, 119-125 (module + class docstrings)
- examples/02_pytorch_conscious_observer.py:98,246; examples/05_consciousness_evaluation.py:8-10,235
- examples/README.md:25; CLAUDE.md:8; README.md:15-18,195-203

## Relation to the music track (09_music_plan.md)

None of A–F blocks 9a or 9b (reel-only, pure numpy) — those can start immediately. 9c runs on
the experiments/embodied mount, which has its own loop (holo_beta) independent of the core lib.
The dependency runs the other way: experiment #10 (hysteresis) is best run TWICE — once on the
embodied mount (cheap, soon) and once through the core lib's D-phase g-sweep routine (the
canonical version). Suggested interleave: build 9a/9b while A+B land, then D's g-sweep and #10
converge.
