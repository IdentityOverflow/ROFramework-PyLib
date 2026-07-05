"""
Nonlinear Recall — the decider the film was missing.

Where #3 left off: the holo-reservoir film stores permanently but has two walls,
both linear-recall artifacts:

    1. CAPACITY WALL: past ~N superposed symbols, matched-filter recall drowns
       in crosstalk (12% at 800 stored, N=512).
    2. MESSY-CUE WALL (#2b's ghost): a corrupted cue cannot cleanly retrieve —
       linear recall has no mechanism to "decide".

Both walls are attacked with the same weapon: nonlinear decisions fed back into
the recall loop.

EXPERIMENT A — Successive Interference Cancellation + relaxation (capacity).
    The film IS a CDMA channel: symbols spread by near-orthogonal codes (the
    clock phases), superposed in one medium.  Linear recall = matched filter.
    Telecom's fix for the matched filter's crosstalk wall:
      • SIC: decode the most CONFIDENT position (top1-top2 margin), SUBTRACT
        its reconstruction from the film, re-decode the rest on the residual.
      • Relaxation: then sweep positions repeatedly — put each symbol back,
        re-decode it against everything currently explained, until a fixed
        point (coordinate descent on the film's explanation; Hopfield-style).
    Success criterion: nonlinear recall clearly beats linear in the breakdown
    region (stored 100-400 at N=512) — i.e. the capacity wall is a property of
    LINEAR recall, not of the film.

EXPERIMENT B — Episodic recall from a corrupted content cue.
    A first design failed structurally, and the failure rhymes with #2b: cueing
    with a MASKED TIMESTAMP is ill-posed, because the clock manifold has one
    degree of freedom — a small fragment of reference phases pins down k with
    no memory needed at all (the fragment "already determines the whole", the
    same trap that flattened #2b's smooth patterns).
    The well-posed messy cue is CONTENT, and the well-posed question is
    EPISODIC: given a phase-noised version of a symbol that was stored, the cue
    alone can at best identify the symbol (codebook prior) — it can NEVER say
    WHEN it happened or WHAT CAME NEXT.  Only the film can.  So the floors are:
        identity: direct nearest-codebook decode of the noisy cue (strong floor)
        context (what came next): chance (no film, no answer)
    The nonlinear mechanism is a recognition scan over the clock manifold with
    the decider inside the loop: for every position k, read the film, PROJECT
    onto the codebook (the nonlinear step — reconstructing a clean candidate),
    and score that clean candidate against the noisy cue.  Best match wins;
    read position k+1 for the episodic answer.
    Success criterion: context retrieval far above chance, tracking the
    identity floor's noise tolerance — completion-through-decisions where #2b
    showed completion-through-dynamics is impossible.

Honest scope: the nonlinear recall's power comes from structured priors — the
codebook (discrete attractors) and the clock family (1-dof manifold).  That is
the claim, not a caveat: #2a/#2b showed the medium alone earns nothing; the
holographic win appears only when recall passes through nonlinear projections
onto known structure.  Linear film (permanent store) + nonlinear decider
(projection onto attractors) = the Seed coherence-gate in miniature.

Requires: numpy.   Runtime: ~1-2 min on CPU.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Film machinery (identical ingredients to #3)
# ---------------------------------------------------------------------------


def make_codebook(k_symbols: int, n: int, rng) -> np.ndarray:
    """K symbols, each a unit-magnitude phasor vector (random phases)."""
    return np.exp(1j * 2 * np.pi * rng.random((k_symbols, n)))


def timestamp(omega, dt, k):
    return np.exp(-1j * omega * dt * k)


def record_film(codebook, omega, dt, seq):
    """Record symbol ⊙ timestamp per step into a no-decay film (as in #3)."""
    film = np.zeros(len(omega), dtype=complex)
    for t, s in enumerate(seq):
        film += codebook[s] * timestamp(omega, dt, t)
    return film


def recall_film(film, omega, dt, k):
    """Linear recall of position k: shine the reference back."""
    return film * np.exp(1j * omega * dt * k)


def decode(x, codebook) -> int:
    return int(np.argmax(np.abs(codebook.conj() @ x)))


# ---------------------------------------------------------------------------
# Experiment A — SIC + relaxation (capacity)
# ---------------------------------------------------------------------------


def recall_all_linear(film, codebook, omega, dt, t_steps):
    """Baseline: independent matched-filter decode of every position."""
    return [decode(recall_film(film, omega, dt, k), codebook)
            for k in range(t_steps)]


def recall_all_sic(film, codebook, omega, dt, t_steps, batch_frac=0.05):
    """Successive interference cancellation: commit confident, subtract, repeat."""
    residual = film.copy()
    decoded = [-1] * t_steps
    remaining = list(range(t_steps))
    while remaining:
        scored = []
        for k in remaining:
            m = np.abs(codebook.conj() @ recall_film(residual, omega, dt, k))
            top2 = np.argpartition(m, -2)[-2:]
            best = top2[np.argmax(m[top2])]
            margin = m[best] - min(m[top2])
            scored.append((margin, k, int(best)))
        scored.sort(reverse=True)
        for _, k, sym in scored[:max(1, int(len(scored) * batch_frac))]:
            decoded[k] = sym
            residual -= codebook[sym] * timestamp(omega, dt, k)
            remaining.remove(k)
    return decoded, residual


def recall_all_relax(film, codebook, omega, dt, t_steps, max_rounds=8, rng=None):
    """SIC init, then coordinate-descent relaxation to a fixed point.

    Each sweep: put one position's current explanation back into the residual,
    re-decode it against everything else currently explained, subtract again.
    Stops when a full sweep changes nothing.
    """
    decoded, residual = recall_all_sic(film, codebook, omega, dt, t_steps)
    recon = [codebook[decoded[k]] * timestamp(omega, dt, k)
             for k in range(t_steps)]
    order = np.arange(t_steps)
    for _ in range(max_rounds):
        if rng is not None:
            rng.shuffle(order)
        changed = 0
        for k in order:
            residual += recon[k]
            sym = decode(recall_film(residual, omega, dt, k), codebook)
            if sym != decoded[k]:
                decoded[k] = sym
                recon[k] = codebook[sym] * timestamp(omega, dt, k)
                changed += 1
            residual -= recon[k]
        if changed == 0:
            break
    return decoded


def capacity_race(n, k_symbols, omega, dt, t_list, n_trials, rng):
    rows = []
    for t_steps in t_list:
        hits = np.zeros(3)
        total = 0
        for _ in range(n_trials):
            codebook = make_codebook(k_symbols, n, rng)
            seq = rng.integers(k_symbols, size=t_steps)
            film = record_film(codebook, omega, dt, seq)
            lin = recall_all_linear(film, codebook, omega, dt, t_steps)
            sic, _ = recall_all_sic(film, codebook, omega, dt, t_steps)
            rel = recall_all_relax(film, codebook, omega, dt, t_steps, rng=rng)
            for i, arm in enumerate((lin, sic, rel)):
                hits[i] += sum(a == b for a, b in zip(arm, seq))
            total += t_steps
        rows.append((t_steps, *(hits / total)))
    return rows


# ---------------------------------------------------------------------------
# Experiment B — episodic recall from a corrupted content cue
# ---------------------------------------------------------------------------


def episodic_recall(film, codebook, omega, dt, t_steps, cue):
    """Recognition scan over the clock manifold, decider inside the loop.

    For each position: read the film, project onto the codebook (nonlinear
    cleanup), score the CLEAN candidate against the noisy cue.  Returns
    (k_est, identity_est, next_est).
    """
    cleaned = [decode(recall_film(film, omega, dt, k), codebook)
               for k in range(t_steps)]
    match = np.abs(codebook[cleaned].conj() @ cue)      # (t_steps,)
    k = int(np.argmax(match))
    next_sym = (decode(recall_film(film, omega, dt, k + 1), codebook)
                if k + 1 < t_steps else -1)
    return k, cleaned[k], next_sym


def episodic_race(n, k_symbols, t_steps, omega, dt, sigmas, n_trials,
                  cues_per_trial, rng):
    rows = []
    for sigma in sigmas:
        direct = ident = timehit = context = total = 0
        for _ in range(n_trials):
            codebook = make_codebook(k_symbols, n, rng)
            # sample without replacement: every stored symbol is unique,
            # so "when did I see this?" has exactly one true answer
            seq = rng.permutation(k_symbols)[:t_steps]
            film = record_film(codebook, omega, dt, seq)
            for k_true in rng.integers(t_steps - 1, size=cues_per_trial):
                cue = (codebook[seq[k_true]]
                       * np.exp(1j * rng.normal(0.0, sigma, n)))
                direct += decode(cue, codebook) == seq[k_true]
                k_est, sym_est, next_est = episodic_recall(
                    film, codebook, omega, dt, t_steps, cue)
                ident += sym_est == seq[k_true]
                timehit += k_est == k_true
                context += next_est == seq[k_true + 1]
                total += 1
        rows.append((sigma, direct / total, ident / total,
                     timehit / total, context / total))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(n=512, dt=1.0, seed=0):
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi * np.arange(1, n + 1) / n

    print("=" * 70)
    print("Nonlinear Recall — the decider the film was missing")
    print("=" * 70)

    print(f"\nEXPERIMENT A — capacity: N={n}, codebook=32")
    print("  (the film is a CDMA channel; subtract what you have decided)")
    t_list = [50, 100, 150, 200, 300, 400, 600]
    rows = capacity_race(n, 32, omega, dt, t_list, n_trials=6, rng=rng)
    print("    stored:    " + " ".join(f"{int(t):>5}" for t, *_ in rows))
    print("    linear:    " + " ".join(f"{l:>5.0%}" for _, l, _, _ in rows)
          + "   ← #3 baseline")
    print("    SIC:       " + " ".join(f"{s:>5.0%}" for _, _, s, _ in rows)
          + "   ← subtract once")
    print("    SIC+relax: " + " ".join(f"{r:>5.0%}" for _, _, _, r in rows)
          + "   ← fixed point")

    print(f"\nEXPERIMENT B — episodic recall: codebook=64, seq=50 unique symbols")
    print("  cue = stored symbol with phase noise σ; ask WHEN and WHAT CAME NEXT")
    sigmas = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5]
    rows_b = episodic_race(n, 64, 50, omega, dt, sigmas,
                           n_trials=5, cues_per_trial=10, rng=rng)
    print("    cue noise σ:  " + " ".join(f"{s:>5.1f}" for s, *_ in rows_b))
    print("    identity/direct " + " ".join(f"{d:>5.0%}" for _, d, _, _, _ in rows_b)
          + "  ← floor (no film)")
    print("    identity/film   " + " ".join(f"{i:>5.0%}" for _, _, i, _, _ in rows_b))
    print("    when (k found)  " + " ".join(f"{t:>5.0%}" for _, _, _, t, _ in rows_b)
          + "  ← film only")
    print("    what came next  " + " ".join(f"{c:>5.0%}" for _, _, _, _, c in rows_b)
          + "  ← film only (chance 2%)")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  • A: wherever SIC+relax holds while linear collapses, the capacity")
    print("    wall was a property of LINEAR recall, not of the film — decisions")
    print("    fed back into the medium recover the superposed past.")
    print("  • B: 'when did I see this?' and 'what came next?' are unanswerable")
    print("    from the cue alone (chance floor) — if the film answers them from")
    print("    a corrupted cue, that is completion-through-decisions, where #2b")
    print("    proved completion-through-dynamics impossible.")
    print("  • Together: linear film (permanent store) + nonlinear decider")
    print("    (projection onto codebook attractors) = episodic memory for a")
    print("    reservoir agent — the coherence-gate architecture, earned.")
    print("\nDone.")


if __name__ == "__main__":
    main()
