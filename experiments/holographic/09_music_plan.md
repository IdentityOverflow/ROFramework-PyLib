# #9 — Music: episodic memory under a knowable expectation structure (PLAN)

*Status: design agreed 2026-07-28, not yet built. Numbering note: this track absorbs the
action-side-tags + recollection-as-suggestion experiment queued in findings #8 — see 9c,
where "action" = next-symbol and the suggestion machinery gets its cleanest possible test.*

## Why music, stated once

Every finding since #6 has been throttled by one bottleneck: **"the reel is only as prophetic
as the life it records"** — and the agent's life is noise. When foresight r ≈ 0 we cannot
attribute the failure: bad reel, or unpredictable world? Music inverts this:

1. **The generative grammar is known exactly** → per-note surprisal is computable, and the
   *foresight ceiling* r_max is computable by rollout. We stop reporting absolute r and start
   reporting **reel efficiency r/r_max**.
2. **Prediction strength is a dial** (grammar temperature τ) → predictability becomes an axis,
   not an accident.
3. **Violations arrive on schedule** (cadences, deceptive cadences) → dense, structured valence
   events; fixes #7's dead-world problem (Seed-Hemi: 0–1 events/15k steps).
4. **Music is natively symbolic** (scale degrees) → no VQ codebook, no EMA/dwell gating, no
   ~1σ neighbor-separation problem (#5). A whole noise layer deleted.
5. Negative results become decisive: if the reel can't prophesy *here*, the fault is finally
   the reel's.

## Shared infrastructure: `melody.py`

```
Grammar:   first-order Markov chain over scale degrees (alphabet: 7 degrees × 2 octaves
           + rest ≈ 15 symbols), transitions biased musically (steps > leaps,
           leading-tone → tonic strong). Temperature τ interpolates deterministic
           (τ→0) ↔ uniform (τ→∞). Exact p(x_t | x_{t-1}) always available.
Phrases:   fixed length (16 notes) = episodes = slides. Cadence schedule at phrase end:
           authentic (V→I, resolving) vs deceptive (V→vi) with set probabilities.
Motifs:    library of ~8 distinct 4-note incipits recurring across phrases — trajectory-stub
           cues (#5 lesson) with known identity, for episodic recall scoring.
Rare set:  2–3 motifs deliberately shown only 1–2× per run (for 9c's episodic-vs-parametric
           dissociation).
Valence:   v_t = z(−surprisal_t), z-scored online. Cadence resolutions and deceptive
           cadences produce ±spikes automatically — no hand-crafted schedule.
           (Huron-style tension = next-step entropy is a later refinement, noted not built.)
Truth:     r_max(τ, horizon) by Monte-Carlo rollout of the chain. Report everywhere.
```

## 9a — The reel reads sheet music (reel-only, no brain)

**Question:** with clean symbols and known statistics, what are the reel's real numbers
relative to ground-truth ceilings?

- Feed scale-degree symbols directly into films; slides = phrases; tags = v_t.
- **Tests:**
  1. *Episodic recall:* cue with a 4-note motif stub → correct phrase + position?
     (Exact chance rates now computable; compare #4's ~80%-from-corrupted-cues.)
  2. *What-came-next:* recalled continuation vs actual continuation (symbol accuracy).
  3. *Foresight:* dv̂ vs true future Δvalence → report r and **r/r_max**.
- **Sweep:** τ ∈ {low, mid, high} × load (phrases per reel) → capacity-vs-entropy surface.
- **Prediction:** r/r_max high at low τ under light load; collapses past the #4-style load
  threshold; τ shifts the threshold. Pure numpy, 5 seeds, no GPU.

## 9b — Transposition: values vs differences (the RO test)

**Question:** does relational encoding transfer where absolute encoding fails? This is the
framework's derived-DoF / reference-frame claim (ro_framework.md Part I–II) as a falsifiable
recall experiment.

- Same melodies, two encodings: **ABS** (pitch classes) vs **REL** (successive intervals,
  clipped to ±7). Control: match alphabet sizes/entropy as closely as possible.
- Record reel in C; test recall/foresight on the same melodies transposed +4 semitones.
- **Predictions:** REL retains recall/foresight under transposition; ABS collapses to chance.
  Secondary, worth having either way: in-key, ABS may slightly beat REL (absolute position
  information is real information) — if so, that's a complementarity pair (invariance ↔
  absolute information; ro_framework §7.2), measured.

## 9c — The mounted listener + recollection-as-suggestion (absorbs queued #9)

**Question:** in a predictable world, does the closed loop finally do work — and can recalled
continuations steer prediction? Episodic control (NEC-style), melodic version.

- **Brain:** ESN or wim; readout predicts next-symbol distribution; reward = −surprisal of
  its own prediction (prediction-quality reward; valence now dense and structured).
- **Side-tags (the #8 fix, one level cleaner):** record on the film, alongside valence,
  the symbol that came next. Recall then carries *policy-relevant content*, which #8 proved
  value-only tags cannot.
- **Suggestion:** on high-score recall, blend recalled continuation into the readout's
  prediction, gated by SIC score margin (the #6 confidence gate, now load-bearing).
- **Arms:** baseline / reel-observer (β=0, no suggestion) / reel+dv̂ (β>0) /
  reel+suggestion (β=0) / reel+both. 3 seeds × τ ∈ {low, high}.
- **Metrics:** next-note accuracy vs grammar optimum; r/r_max; suggestion hit-rate when
  fired; and the key dissociation — performance on **rare motifs** (1–2 exposures) vs
  common material.
- **Predictions (falsifiable both ways):** suggestions help specifically on rare-but-recurring
  motifs — one-shot patterns the slow parametric readout can't own; no effect on
  high-frequency material. If suggestion fails *here* — clean symbols, known grammar, dense
  valence — recollection-as-suggestion is dead in general, not throttled-by-noise. That is
  the point of running it in music first.

## #10 (follow-on, not this batch) — closure sweep / hysteresis on the melodic stream

The ro_framework v2 §5.5 falsification experiment needs dense valence events; the melodic
stream finally supplies them. Sweep holo_beta up then down; gradualism predicts smooth
path-independent scaling, the framework predicts discontinuity + hysteresis. Deliberately
deferred until 9a–9c establish signal, so a null can't be blamed on a dead loop.

## Build order

9a first (pure numpy, fastest, calibrates everything), 9b same harness + one encoding swap,
9c only after 9a shows nonzero r/r_max somewhere. Findings go to findings.md as usual.
