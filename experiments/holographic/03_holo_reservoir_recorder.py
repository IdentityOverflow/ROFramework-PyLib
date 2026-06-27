"""
Holo-Reservoir Recorder — the reservoir forgets, the film remembers.

The idea you came in with: a reservoir is bad at long memory because its echo
fades.  Holography is the trick of *freezing* a fleeting wave into a medium that
does NOT fade.  So put them side by side and let them race.

The reservoir (Fourier-native):
    A bank of N damped oscillators at the DFT frequencies — i.e. the reservoir is
    literally a sliding Fourier transform.  Drive it with a sequence of symbols;
    it rings and decays.  Its state is a set of phasors (amplitude+phase) across
    frequencies.  Crucially, the bank's own phases form a "clock": at every step
    the N oscillators sit at a unique combination of angles — a natural timestamp
    (the reference wave R).  DFT frequencies make consecutive timestamps clean
    (near-orthogonal); messier, correlated frequencies just lower capacity.

Two memories built from the SAME ingredients, differing in ONE thing — decay:
    • Live echo (fading):  the reservoir's actual state after the sequence.
      Symbol t survives only as  λ^(age)  — older = fainter.  This is normal
      reservoir memory.
    • Frozen film (permanent):  each step we record  symbol ⊙ timestamp  into a
      medium with NO decay (the developed hologram).  Symbol t survives at full
      strength no matter how old.

Recall (holographic):  to read position k, multiply by that step's timestamp
(shine the reference back) and decode the nearest symbol in the codebook.
    film  → symbol_k + crosstalk            (flat with age)
    echo  → λ^(age)·symbol_k + crosstalk     (decays with age)

What this shows:
    • GAIN:  the film recalls items long after the reservoir's echo has died.
    • COST:  the film has no decay to protect it, so as you cram in more symbols
      they superpose and blur — a capacity limit (crosstalk), not a time limit.

Honest scope: this is the part of holography that works *linearly* — storage and
content-addressable recall.  It does NOT do clean completion from a messy cue
(that needs a nonlinear "decider"; see #2b).  Permanence is exactly what answers
the reservoir memory-decay problem.

Requires: numpy + ro_framework.   Runtime: ~5-15s on CPU.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Frequency reservoir + holographic film
# ---------------------------------------------------------------------------


def make_codebook(k_symbols: int, n: int, rng) -> np.ndarray:
    """K symbols, each a unit-magnitude phasor vector (random phases)."""
    return np.exp(1j * 2 * np.pi * rng.random((k_symbols, n)))


def run_recorder(codebook, omega, dt, lam, seq):
    """Drive the oscillator bank with the symbol sequence.

    Returns:
        echo:  live reservoir state after the sequence (fades with age).
        film:  permanent holographic record (no decay).
    Both are length-N complex phasor vectors.
    """
    n = len(omega)
    t_steps = len(seq)
    rot = np.exp(1j * omega * dt)                 # per-step phase advance (the clock)

    echo = np.zeros(n, dtype=complex)
    film = np.zeros(n, dtype=complex)
    for t in range(t_steps):
        sym = codebook[seq[t]]
        # Reservoir: rotate, decay, inject (the echo)
        echo = lam * (echo * rot) + sym
        # Film: record symbol bound to this timestamp (no decay)
        timestamp = np.exp(-1j * omega * dt * t)
        film += sym * timestamp
    return echo, film


def recall_film(film, omega, dt, k):
    """Read position k from the film: multiply by its timestamp."""
    return film * np.exp(1j * omega * dt * k)


def recall_echo(echo, omega, dt, age):
    """Read an item of given age from the live reservoir echo (de-rotate)."""
    return echo * np.exp(-1j * omega * dt * age)


def decode(x, codebook) -> int:
    """Nearest symbol by phasor matched filter."""
    return int(np.argmax(np.abs(codebook.conj() @ x)))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def recall_vs_age(n, k_symbols, t_steps, lam, omega, dt, n_trials, rng):
    """Recall accuracy as a function of how old the item is."""
    ages = list(range(t_steps))
    film_hits = np.zeros(t_steps)
    echo_hits = np.zeros(t_steps)
    for _ in range(n_trials):
        codebook = make_codebook(k_symbols, n, rng)
        seq = rng.integers(k_symbols, size=t_steps)
        echo, film = run_recorder(codebook, omega, dt, lam, seq)
        for k in range(t_steps):
            age = (t_steps - 1) - k
            if decode(recall_film(film, omega, dt, k), codebook) == seq[k]:
                film_hits[age] += 1
            if decode(recall_echo(echo, omega, dt, age), codebook) == seq[k]:
                echo_hits[age] += 1
    return ages, film_hits / n_trials, echo_hits / n_trials


def capacity_sweep(n, k_symbols, lam, omega, dt, t_list, n_trials, rng):
    """Mean film recall (over all positions) vs. number of stored symbols."""
    out = []
    for t_steps in t_list:
        hits = total = 0
        for _ in range(n_trials):
            codebook = make_codebook(k_symbols, n, rng)
            seq = rng.integers(k_symbols, size=t_steps)
            _, film = run_recorder(codebook, omega, dt, lam, seq)
            for k in range(t_steps):
                hits += decode(recall_film(film, omega, dt, k), codebook) == seq[k]
                total += 1
        out.append(hits / total)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(n=512, k_symbols=32, t_steps=50, lam=0.85, dt=1.0, n_trials=30, seed=0):
    rng = np.random.default_rng(seed)
    omega = 2 * np.pi * np.arange(1, n + 1) / n   # DFT oscillator bank
    half_life = np.log(0.5) / np.log(lam)

    print("=" * 70)
    print("Holo-Reservoir Recorder — the reservoir forgets, the film remembers")
    print("=" * 70)
    print(f"  {n} oscillators at the DFT frequencies (a sliding Fourier transform)")
    print(f"  decay λ={lam} → echo half-life ≈ {half_life:.1f} steps")
    print(f"  codebook={k_symbols} symbols, sequence length={t_steps}, trials={n_trials}")

    ages, film_acc, echo_acc = recall_vs_age(
        n, k_symbols, t_steps, lam, omega, dt, n_trials, rng)

    print("\n  Recall accuracy vs. item age (steps since it was seen):")
    show = [0, 1, 2, 4, 8, 16, 24, 32, 40, 48]
    show = [a for a in show if a < t_steps]
    print("    age:  " + " ".join(f"{a:>5}" for a in show))
    print("    echo: " + " ".join(f"{echo_acc[a]:>5.0%}" for a in show)
          + "   ← reservoir (fades)")
    print("    film: " + " ".join(f"{film_acc[a]:>5.0%}" for a in show)
          + "   ← hologram (permanent)")

    # where does the echo effectively die?
    echo_dead = next((a for a in ages if echo_acc[a] < 0.5), t_steps)
    print(f"\n  Reservoir echo drops below 50% by age ≈ {echo_dead} steps;"
          f"  film stays at {film_acc.mean():.0%} across all {t_steps}.")

    print("\n  COST — capacity: mean film recall vs. # symbols crammed in:")
    t_list = [t for t in [25, 50, 100, 200, 400, 800] ]
    caps = capacity_sweep(n, k_symbols, lam, omega, dt, t_list, max(8, n_trials // 3), rng)
    print("    stored: " + " ".join(f"{t:>5}" for t in t_list))
    print("    recall: " + " ".join(f"{c:>5.0%}" for c in caps)
          + f"   (blurs as superposition exceeds ~N={n})")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print(f"  • GAIN: the film recalls age-{t_steps-1} items at {film_acc[t_steps-1]:.0%} "
          f"where the reservoir echo is already dead ({echo_acc[t_steps-1]:.0%}).")
    print("    Same binding, same reference clock — the ONLY difference is the film")
    print("    does not decay.  That is holography answering reservoir memory-decay.")
    print(f"  • COST: the film has no decay to shield it, so memories blur once you")
    print(f"    superpose more than ~N symbols — a capacity limit, not a time limit.")
    print("  • This is the linear half of holography (store + content-addressed recall);")
    print("    clean recall from a messy cue still needs a nonlinear decider (see #2b).")
    print("\nDone.")


if __name__ == "__main__":
    main()
