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
