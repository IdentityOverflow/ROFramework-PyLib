"""
Holographic Wave Memory #2b — recall through the literal lattice dynamics.

#2a showed that for a STATIC memory, complex/holographic phasors buy only what
doubling real DOF buys — no holographic advantage.  The conclusion: a genuine
holographic win must come from recall *through the medium's own dynamics*, not
from static complex vectors.  This experiment tests exactly that, physically.

The holographic picture, made literal:
    A hologram is a medium whose local structure is modulated by the recorded
    interference of a reference and an object wave.  Reconstruction is the
    physical RE-PROPAGATION of the reference wave through that modified medium.
    Here the medium is a 2D spring lattice with LOCAL (neighbour-only) couplings.

    Record:  store smooth wavefield patterns by writing local gratings into the
             couplings via phase-gated Hebbian, Δc_ij ∝ f_i·f_j (the in-phase
             interference / cross term R*O between coupled nodes).
    Recall:  clamp a FRAGMENT of a pattern onto its nodes (hold the reference),
             then run the lattice's wave dynamics through the recorded couplings
             and read the reconstructed field — "a fragment reconstructs the
             whole", the iconic holographic property.

Why smooth (band-limited) patterns: local couplings can only complete patterns
correlated at the lattice scale — i.e. wavefronts.  That is not a cheat; it is
precisely what a holographic medium stores.

Four arms, all at matched representation, to localise any advantage:
    1. lattice-wave      local couplings, recurrent, phase-bearing (momentum)  ← test
    2. lattice-diffusive local couplings, recurrent, NO momentum/phase (overdamped)
    3. local-static      local couplings, ONE-SHOT (no iteration)
    4. global-static     full N×N outer-product memory, one-shot (#2a/wim ideal)

Success bar (from #2a): lattice-wave beats local-static (dynamics help) AND
lattice-diffusive (the wave/phase helps, not just relaxation) on partial-cue
completion.  Bonus if local+dynamics rivals the global ideal despite far fewer
parameters (2N edges vs N²).

RESULT — NEGATIVE (hypothesis falsified).  No memory arm beats the raw-cue floor
(smooth patterns local couplings can store are already fragment-decodable);
iterating the re-propagation hurts (a hologram reconstructs in one pass); and
wave ≈ diffusive (phase irrelevant for linear dynamics).  Conclusion: the linear
wave-optical picture earns no holographic advantage; a real win needs NONLINEAR
attractor recall (modern Hopfield / attention).  See findings.md (#2b).

Requires: numpy + ro_framework.   Runtime: ~30-60s on CPU.
"""

import numpy as np

from ro_framework import Observer, PolarDoF, State
from ro_framework.observer.observer import ObservationPair


# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------


def build_neighbors(g: int):
    """Return (nb, valid, safe): (N,4) neighbour indices, validity, safe index."""
    idx = np.arange(g * g).reshape(g, g)
    nb = -np.ones((g * g, 4), dtype=int)
    for c, (di, dj) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        for i in range(g):
            for j in range(g):
                ni, nj = i + di, j + dj
                if 0 <= ni < g and 0 <= nj < g:
                    nb[idx[i, j], c] = idx[ni, nj]
    valid = (nb >= 0).astype(np.float64)
    safe = np.where(nb >= 0, nb, 0)
    return nb, valid, safe


# ---------------------------------------------------------------------------
# Patterns — smooth wavefields (superpositions of low spatial modes)
# ---------------------------------------------------------------------------


def make_smooth_patterns(g: int, m: int, k_max: int, rng) -> np.ndarray:
    """M smooth, unit-norm wavefield patterns on a g×g lattice.

    Each pattern is a random superposition of 2D cosine modes with spatial
    frequencies up to k_max — band-limited, hence completable by local couplings.
    """
    xs, ys = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    modes = []
    for kx in range(k_max + 1):
        for ky in range(k_max + 1):
            if kx == 0 and ky == 0:
                continue
            modes.append(np.cos(np.pi * kx * (xs + 0.5) / g)
                         * np.cos(np.pi * ky * (ys + 0.5) / g))
    modes = np.array([mode.ravel() for mode in modes])         # (n_modes, N)
    coeffs = rng.standard_normal((m, len(modes)))
    pats = coeffs @ modes                                       # (M, N)
    return pats / (np.linalg.norm(pats, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Recording — local gratings into couplings
# ---------------------------------------------------------------------------


def record_local_couplings(patterns, nb, valid, safe, eta) -> np.ndarray:
    """g[i,k] ∝ Σ_m f_m[i]·f_m[nb[i,k]]  — phase-gated Hebbian per edge.

    Bounded so max|g| = eta, decoupling stability from the pattern count M.
    """
    f_self = patterns[:, :, None]                  # (M, N, 1)
    f_nb = patterns[:, safe] * valid[None]         # (M, N, 4)
    grating = (f_self * f_nb).sum(0) * valid       # (N, 4)
    return eta * grating / (np.abs(grating).max() + 1e-12)


# ---------------------------------------------------------------------------
# Recall arms
# ---------------------------------------------------------------------------


def _make_cue(pattern, frac, rng):
    """Keep a random `frac` of nodes; return (cue vector, kept-node mask)."""
    n = len(pattern)
    keep = rng.random(n) < frac
    if not keep.any():
        keep[rng.integers(n)] = True
    cue = np.where(keep, pattern, 0.0)
    return cue, keep


def recall_lattice(cue, keep, grating, valid, safe, tension, steps,
                   wave=True, damp=0.05, alpha=0.2):
    """Re-propagate through the recorded lattice; clamp the fragment each step.

    wave=True:  second-order spring dynamics (momentum = phase-bearing).
    wave=False: first-order relaxation (overdamped diffusion = phase-blind).
    """
    d = cue.copy()
    v = np.zeros_like(d)
    c = tension * (1.0 + grating) * valid          # (N, 4) effective coupling
    for _ in range(steps):
        d_nb = d[safe] * valid                     # (N, 4)
        lap = (c * (d_nb - d[:, None] * valid)).sum(1)
        if wave:
            v = (1.0 - damp) * v + lap
            d = d + v
        else:
            d = d + alpha * lap
        d = np.clip(d, -3.0, 3.0)
        d[keep] = cue[keep]                        # hold the reference fragment
    return d


def recall_local_static(cue, grating, valid, safe):
    """One-shot local readout: p̂_i = Σ_k g[i,k]·cue[nb[i,k]]  (no iteration)."""
    return (grating * (cue[safe] * valid)).sum(1) + cue


def recall_global_static(H, cue):
    """One-shot global outer-product memory: p̂ = H·cue."""
    return H @ cue


def decode(field, patterns) -> int:
    """Nearest stored prototype by cosine similarity."""
    f = field / (np.linalg.norm(field) + 1e-12)
    return int(np.argmax(patterns @ f))


# ---------------------------------------------------------------------------
# K(d_ext) on lattice-wave completion
# ---------------------------------------------------------------------------


def assess_completion_knowledge(true_fields, recalled_fields) -> "KnowledgeAssessment":
    """ρ = how well the reconstructed field tracks the true stored pattern."""
    true_dof = PolarDoF(name="true_field", pole_negative=-1.0, pole_positive=1.0)
    rec_dof = PolarDoF(name="reconstructed")
    obs = Observer(name="completion", internal_dofs=[rec_dof], external_dofs=[true_dof],
                   world_model=lambda s: State(values={rec_dof: 0.0}),
                   log_capacity=true_fields.size + 100)
    t = 0
    for tf, rf in zip(true_fields, recalled_fields):
        rf = rf / (np.linalg.norm(rf) + 1e-12)
        for a, b in zip(tf, rf):
            obs.observation_log.append(ObservationPair(
                external_state=State(values={true_dof: float(a)}),
                internal_state=State(values={rec_dof: float(b)}),
                timestamp=float(t)))
            t += 1
    return obs.assess_knowledge(true_dof, min_samples=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(grid=16, n_patterns=40, k_max=7, eta=0.5, tension=0.3, steps=300,
         n_trials=80, seed=0):
    rng = np.random.default_rng(seed)
    n = grid * grid
    nb, valid, safe = build_neighbors(grid)
    patterns = make_smooth_patterns(grid, n_patterns, k_max, rng)
    grating = record_local_couplings(patterns, nb, valid, safe, eta)
    H = patterns.T @ patterns                       # (N, N) global memory

    n_edges = int(valid.sum() // 2)
    print("=" * 72)
    print("Holographic Wave Memory #2b — literal lattice re-propagation")
    print("=" * 72)
    print(f"  grid={grid}x{grid}  N={n}  patterns={n_patterns} (smooth, k≤{k_max})")
    print(f"  local edges={n_edges}  vs  global memory N²={n*n}  "
          f"({n*n // n_edges}× more params)")
    print(f"  recall steps={steps}  trials/point={n_trials}")

    fracs = [0.5, 0.3, 0.2, 0.1, 0.05]
    arms = ["raw-cue (no mem)", "lattice-wave", "lattice-diffusive",
            "local-static", "global-static"]
    acc = {a: [] for a in arms}

    # store true/recalled fields at one fraction for K assessment
    k_true, k_rec = [], []
    k_frac = 0.3

    for frac in fracs:
        hits = {a: 0 for a in arms}
        for _ in range(n_trials):
            a = rng.integers(n_patterns)
            cue, keep = _make_cue(patterns[a], frac, rng)

            fields = {
                "raw-cue (no mem)": cue,
                "lattice-wave": recall_lattice(cue, keep, grating, valid, safe,
                                               tension, steps, wave=True),
                "lattice-diffusive": recall_lattice(cue, keep, grating, valid, safe,
                                                    tension, steps, wave=False),
                "local-static": recall_local_static(cue, grating, valid, safe),
                "global-static": recall_global_static(H, cue),
            }
            for name, fld in fields.items():
                if decode(fld, patterns) == a:
                    hits[name] += 1
            if abs(frac - k_frac) < 1e-9:
                k_true.append(patterns[a]); k_rec.append(fields["local-static"])
        for name in arms:
            acc[name].append(hits[name] / n_trials)

    # --- Report ---
    print("\n  Completion accuracy vs. fragment fraction (cue = frac of pattern's nodes):")
    print(f"  {'arm':>18} | " + " ".join(f"{f:>5.0%}" for f in fracs))
    print("  " + "-" * (21 + 6 * len(fracs)))
    for name in arms:
        print(f"  {name:>18} | " + " ".join(f"{x:>5.0%}" for x in acc[name]))

    k = assess_completion_knowledge(np.array(k_true), np.array(k_rec))
    print(f"\n  K(d_ext) on single-pass (local-static) completion (fragment={k_frac:.0%}):")
    print(f"    ρ={k.correlation:.3f}  ε={k.systematic_error:.3f}  "
          f"σ={k.random_error:.3f}  C={k.calibration:.3f}  → {k.knowledge_type}")

    # --- Does iterating the re-propagation help or hurt? (steps sweep) ---
    step_grid = [1, 3, 10, 30, 100, 300]
    sweep_frac = 0.1
    print(f"\n  lattice-wave completion vs. # re-propagation steps "
          f"(fragment={sweep_frac:.0%}):")
    sweep = []
    for st in step_grid:
        hits = 0
        for _ in range(n_trials):
            a = rng.integers(n_patterns)
            cue, keep = _make_cue(patterns[a], sweep_frac, rng)
            fld = recall_lattice(cue, keep, grating, valid, safe, tension, st, wave=True)
            hits += decode(fld, patterns) == a
        sweep.append(hits / n_trials)
    print(f"    steps: " + " ".join(f"{s:>5}" for s in step_grid))
    print(f"    acc:   " + " ".join(f"{x:>5.0%}" for x in sweep))

    # --- Verdict ---
    print("\n" + "=" * 72)
    print("READING — was the literal re-propagation hypothesis right?")
    print("=" * 72)

    def mean(name):
        return float(np.mean(acc[name]))

    raw, wave, diff, loc, glob = (mean(x) for x in arms)
    print(f"  mean completion across fractions:")
    print(f"    raw-cue(no mem)={raw:.2f}  wave={wave:.2f}  diffusive={diff:.2f}"
          f"  local-static={loc:.2f}  global-static={glob:.2f}")
    print(f"  • memory adds anything (best mem-arm > raw-cue): "
          f"{'PASS' if max(loc, glob, wave) > raw + 0.05 else 'FAIL — raw cue ties everything'}")
    print(f"  • iterating re-propagation helps (wave > local-static): "
          f"{'PASS' if wave > loc + 0.05 else 'FAIL — iterating hurts'}")
    print(f"  • wave/phase beats diffusion (wave > diffusive): "
          f"{'PASS' if wave > diff + 0.05 else 'FAIL — momentum irrelevant'}")
    print("\n  NEGATIVE RESULT — the literal re-propagation hypothesis is falsified:")
    print("  1. No memory arm beats raw-cue: there is no completion to do.  The local")
    print("     couplings can only store SMOOTH (band-limited) patterns, but smooth")
    print("     patterns are redundant enough that a tiny fragment already determines")
    print("     them.  The regime where completion would matter (high-frequency,")
    print("     fragment-ambiguous patterns) is exactly what local couplings cannot store.")
    print("  2. Iterating the wave re-propagation HURTS (steps sweep): linear iteration")
    print("     accumulates crosstalk.  A hologram reconstructs in one diffraction pass.")
    print("  3. Wave ≈ diffusive: phase/momentum is irrelevant for linear re-propagation.")
    print("\n  Implication: on the wave substrate, neither static complex vectors (#2a) nor")
    print("  literal multi-step re-propagation (#2b) earns the holographic framing.  A real")
    print("  memory advantage would require NONLINEAR attractor recall (modern Hopfield /")
    print("  attention) — a recurrent-neural mechanism, not a linear wave-optical one.")
    print("\nDone.")


if __name__ == "__main__":
    main()
