"""
Holographic Wave Memory #2 (wave substrate) — does the phase the ESN throws
away buy us binding capacity?

Context (the holography ↔ HRR thread):
    A hologram stores a scene by recording the *interference* between a
    reference wave and an object wave — it keeps phase, where a photograph
    keeps only amplitude.  Our wave reservoir (experiments/embodied/wim_brain.py)
    is a 2D spring lattice whose state is (displacement, velocity).  For a
    harmonic medium those two are the in-phase and quadrature components of a
    complex phasor z = disp + i·vel/ω — so the medium already holds the full
    complex wave.  But the standard readout is `W_out @ tanh(disp)`: it keeps
    disp and throws vel (the phase quadrature) away.  It reads the hologram's
    amplitude and never uses the phase.

    This experiment asks the falsifiable question behind #2:
    does using the full complex phasor (holographic) store and retrieve more
    associations than the amplitude-only readout — and is the advantage from
    the *phase structure* or merely from recovering the extra degrees of
    freedom?

Design — classic holographic / correlation-matrix associative memory
    (Gabor 1969; Kohonen 1972; the linear core of HRR), built on phasors that
    the wave medium produces:

    1. A vocabulary of items, each injected into the spring lattice as a random
       initial displacement; after T steps of wave propagation the medium's
       (disp, vel) state is that item's phasor — a point spanning many standing
       modes.  Phase = relative timing of the excited modes (genuine info).
    2. Store M key→value associations one-shot: H = Σ vₘ kₘ^H  (outer products).
    3. Recall: r = H k_a ≈ v_a + crosstalk;  decode = nearest value prototype.

    Four representations of the same medium state are compared:
        (a) disp        — amplitude only, the current wim readout      [N dof]
        (b) |z|         — magnitude, phase-blind                        [N dof]
        (c) z=disp+i·vel— full complex phasor (HOLOGRAPHIC)            [2N dof]
        (d) [disp,vel]  — concatenated real, DOF-matched control       [2N dof]

    Predictions / success criteria:
        • Capacity:   (c) ≈ (d) > (a),(b).  Using the full wave state beats the
          amplitude-only readout.  If (c) > (d), the complex phase structure
          helps *beyond* doubling the DOF — the stronger holographic claim.
          If (c) ≈ (d), the gain is honestly just the recovered quadrature DOF.
        • Phase carries info:  (a),(c) [phase-bearing] beat (b) [phase-blind].
        • K(d_ext) on one-shot recall reaches "strong" for the holographic arm.

    Honest scope:  this isolates whether the wave medium's phasor representation
    gives a binding/capacity advantage and whether phase is informative.  The
    deeper holographic claim — recall by re-injecting the reference wave and
    reconstructing the object wave *through the medium's own adapted couplings*
    (phase-gated Hebbian) — is the follow-up #2b.  Here recall is a static
    correlation matrix, so both holographic and real arms are one-shot; the
    differentiator is capacity and the phase channel, not number of passes.

Requires: numpy + ro_framework.   Runtime: ~20-40s on CPU.
"""

import sys

import numpy as np

from ro_framework import Observer, PolarDoF, State
from ro_framework.observer.observer import ObservationPair


# ---------------------------------------------------------------------------
# Wave medium — minimal 2D spring lattice (same physics as wim_brain's
# WaveReservoir, decoupled from the embodied harness)
# ---------------------------------------------------------------------------


class WaveMedium:
    """2D spring lattice. State = (disp, vel). Linear harmonic dynamics."""

    def __init__(self, grid: int = 18, tension: float = 0.3,
                 damping: float = 0.002):
        self.grid = grid
        self.n = grid * grid
        self.tension = tension
        self.damping = damping
        self._nb = self._build_neighbors(grid)

    @staticmethod
    def _build_neighbors(g: int) -> np.ndarray:
        """(N, 4) neighbor index array, -1 = no neighbor (boundary)."""
        idx = np.arange(g * g).reshape(g, g)
        nb = -np.ones((g * g, 4), dtype=int)
        for (di, dj, c) in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
            for i in range(g):
                for j in range(g):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < g and 0 <= nj < g:
                        nb[idx[i, j], c] = idx[ni, nj]
        return nb

    def phasor(self, init_disp: np.ndarray, steps: int = 60) -> tuple:
        """Run the medium from an initial displacement; return (disp, vel)."""
        d = init_disp.astype(np.float64).copy()
        v = np.zeros(self.n)
        valid = self._nb >= 0
        safe = np.where(valid, self._nb, 0)
        for _ in range(steps):
            nb_d = d[safe] * valid                    # (N, 4)
            mean_nb = nb_d.sum(1) / np.maximum(valid.sum(1), 1)
            force = self.tension * (mean_nb - d)
            v = (1.0 - self.damping) * v + force
            d = d + v
        return d, v


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


def build_vocab(medium: WaveMedium, vocab_size: int, rng) -> dict:
    """Generate phasors for a vocabulary; return feature matrices per arm.

    Each item = random dense initial displacement → medium state (disp, vel).
    Rows are L2-normalized (complex norm for the holographic arm) so keys are
    unit vectors and inner products measure overlap directly.
    """
    disp = np.zeros((vocab_size, medium.n))
    vel = np.zeros((vocab_size, medium.n))
    for i in range(vocab_size):
        init = rng.standard_normal(medium.n)
        disp[i], vel[i] = medium.phasor(init)

    def norm_rows(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    return {
        "disp (amplitude only)":   norm_rows(disp),
        "|z| (phase-blind)":       norm_rows(np.sqrt(disp**2 + vel**2)),
        "z=disp+i*vel (HOLO)":     norm_rows(disp + 1j * vel),
        "[disp,vel] (DOF control)": norm_rows(np.concatenate([disp, vel], 1)),
    }


# ---------------------------------------------------------------------------
# Holographic / correlation-matrix associative memory
# ---------------------------------------------------------------------------


def recall_accuracy(feats: np.ndarray, m: int, n_trials: int, rng) -> float:
    """Store m random key→value pairs one-shot; return mean recall accuracy.

    H = Σ vₘ kₘ^H ;  recall r = H k_a ;  decode = argmax Re(vᵤ^H r).
    Keys and values are distinct vocab items with a random pairing.
    """
    v_size = feats.shape[0]
    is_complex = np.iscomplexobj(feats)
    hits = 0
    total = 0
    for _ in range(n_trials):
        perm = rng.permutation(v_size)
        keys = feats[perm[:m]]              # (m, D)
        vals = feats[perm[m:2 * m]] if 2 * m <= v_size else feats[rng.permutation(v_size)[:m]]

        # H = Σ_m v_m k_m^H   (D × D).  vals.T @ keys.conj() builds exactly this.
        H = vals.T @ (keys.conj() if is_complex else keys)
        for a in range(m):
            r = H @ keys[a]                 # ≈ v_a (k_a^H k_a) + crosstalk
            scores = np.real(vals.conj() @ r) if is_complex else vals @ r
            if int(np.argmax(scores)) == a:
                hits += 1
            total += 1
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# K(d_ext) on one-shot recall (holographic arm)
# ---------------------------------------------------------------------------


def assess_recall_knowledge(feats: np.ndarray, m: int, rng) -> "KnowledgeAssessment":
    """Build (true value component, retrieved component) pairs and assess K.

    ρ = how well the retrieved field tracks the true stored value across all
    dimensions and probes; ε,σ,C from the regression residuals.
    """
    v_size = feats.shape[0]
    is_complex = np.iscomplexobj(feats)
    perm = rng.permutation(v_size)
    keys = feats[perm[:m]]
    vals = feats[perm[m:2 * m]]
    H = vals.T @ (keys.conj() if is_complex else keys)

    true_dof = PolarDoF(name="true_value", pole_negative=-1.0, pole_positive=1.0)
    ret_dof = PolarDoF(name="retrieved")
    observer = Observer(
        name="wave_recall", internal_dofs=[ret_dof], external_dofs=[true_dof],
        world_model=lambda s: State(values={ret_dof: 0.0}),
        log_capacity=10 * m * feats.shape[1] + 100,
    )
    t = 0
    for a in range(m):
        r = H @ keys[a]
        true_v, ret_v = vals[a], r
        # normalize retrieved scale for fair component comparison
        ret_v = ret_v / (np.linalg.norm(ret_v) + 1e-12)
        if is_complex:
            true_comp = np.concatenate([true_v.real, true_v.imag])
            ret_comp = np.concatenate([ret_v.real, ret_v.imag])
        else:
            true_comp, ret_comp = true_v, ret_v
        for tc, rc in zip(true_comp, ret_comp):
            observer.observation_log.append(ObservationPair(
                external_state=State(values={true_dof: float(tc)}),
                internal_state=State(values={ret_dof: float(rc)}),
                timestamp=float(t),
            ))
            t += 1
    return observer.assess_knowledge(true_dof, min_samples=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _capacity(m_list, accs, thresh=0.5):
    """Interpolated M at which recall crosses `thresh` (memory capacity)."""
    prev_m, prev_a = m_list[0], accs[0]
    if prev_a < thresh:
        return 0.0
    for m, a in zip(m_list[1:], accs[1:]):
        if a < thresh:
            frac = (prev_a - thresh) / (prev_a - a + 1e-12)
            return prev_m + frac * (m - prev_m)
        prev_m, prev_a = m, a
    return float(m_list[-1])  # never crossed within range


def main(grid: int = 12, vocab_size: int = 3000, n_trials: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    medium = WaveMedium(grid=grid)
    n = medium.n

    print("=" * 70)
    print("Holographic Wave Memory — phase vs amplitude in a spring lattice")
    print("=" * 70)
    print(f"  grid={grid}x{grid}  N={n} nodes  vocab={vocab_size}  trials={n_trials}")
    print("  Each item: random init displacement → wave propagation → (disp,vel) phasor")

    vocab = build_vocab(medium, vocab_size, rng)
    m_list = [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1400]
    m_list = [m for m in m_list if 2 * m <= vocab_size]

    # --- Capacity sweep ---
    print("\n  Recall accuracy vs. number of stored associations M:")
    header = f"  {'representation':>26} | " + " ".join(f"{m:>5}" for m in m_list)
    print(header)
    print("  " + "-" * (len(header) - 2))
    results = {}
    for name, feats in vocab.items():
        accs = [recall_accuracy(feats, m, n_trials, rng) for m in m_list]
        results[name] = accs
        dof = feats.shape[1] * (2 if np.iscomplexobj(feats) else 1)
        print(f"  {name:>26} | " + " ".join(f"{a:>5.0%}" for a in accs)
              + f"   [{dof} dof]")

    # --- Capacity = interpolated M at 50% recall ---
    print("\n  Capacity (interpolated M at 50% recall):")
    for name, accs in results.items():
        print(f"    {name:>26}: {_capacity(m_list, accs):>6.0f}")

    # --- K(d_ext) on holographic one-shot recall ---
    holo = vocab["z=disp+i*vel (HOLO)"]
    m_probe = 40
    k = assess_recall_knowledge(holo, m_probe, rng)
    print(f"\n  K(d_ext) on holographic one-shot recall (M={m_probe}):")
    print(f"    ρ={k.correlation:.3f}  ε={k.systematic_error:.3f}  "
          f"σ={k.random_error:.3f}  C={k.calibration:.3f}  → {k.knowledge_type}")

    # --- Verdict against success criteria ---
    print("\n" + "=" * 70)
    print("READING vs success criteria")
    print("=" * 70)
    cap = {n_: _capacity(m_list, a) for n_, a in results.items()}
    holo_cap = cap["z=disp+i*vel (HOLO)"]
    disp_cap = cap["disp (amplitude only)"]
    ctrl_cap = cap["[disp,vel] (DOF control)"]
    blind_cap = cap["|z| (phase-blind)"]

    ratio = holo_cap / max(disp_cap, 1e-9)
    print(f"  • Capacity: HOLO={holo_cap:.0f} vs amplitude-only={disp_cap:.0f}"
          f" ({ratio:.1f}×)  → {'PASS' if holo_cap > 1.2 * disp_cap else 'FAIL'} "
          f"(full wave state beats the amplitude-only readout)")
    verdict = ("phase structure helps BEYOND dof" if holo_cap > 1.15 * ctrl_cap
               else "advantage is the recovered quadrature DOF (honest)")
    print(f"  • HOLO={holo_cap:.0f} vs DOF-matched real={ctrl_cap:.0f}  → {verdict}")
    print(f"  • Phase carries info: phase-bearing(disp={disp_cap:.0f}) vs "
          f"phase-blind(|z|={blind_cap:.0f})  → "
          f"{'PASS' if disp_cap > 2 * max(blind_cap, 1) else 'inconclusive'}")
    print(f"  • One-shot recall K = {k.knowledge_type} "
          f"({'PASS' if k.knowledge_type == 'strong' else 'below strong'})")
    print("\nDone.")


if __name__ == "__main__":
    main()
