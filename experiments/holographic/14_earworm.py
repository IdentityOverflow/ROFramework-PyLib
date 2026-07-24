"""Earworm — can self-CONSISTENT false content capture the consensus gate? (file 14)

#10b showed diverse self-generated error cannot capture margin_agree gating:
it disagrees with itself and the gate fails safe by silence. The remaining
attack is CORRELATED error — one false phrase, internally consistent,
repeated. The earworm: it opens with a motif shared with true songs (the
cue collision — its entry hook) and then diverges into a continuation that
exists nowhere in the songbook.

Pre-registered predictions (derived from the gate's mechanics before data):
  1. FORK-BLOCKING: at the collision cue, the earworm bloc and the true-song
     bloc both match with identical clean-phasor scores; margin_agree between
     two consistent-but-disagreeing blocs collapses to ~0; the gate CLOSES.
     Capture through the gate at the fork is structurally impossible — the
     consensus gate refuses to choose between stories, regardless of dose.
  2. INTERIOR RIDE: entry happens only by chance (ESN samples the divergent
     note, ~1/13); once inside, cues are earworm-unique, the gate fires, and
     the ride follows the earworm at high fidelity.
  3. NO EPIDEMIC THROUGH THE GATE: with self-teaching on, re-taught interior
     rides grow the earworm bloc, but fork ties stay ties (equal scores), so
     capture rate does NOT compound; dose growth is linear-in-chance-entries,
     not explosive.
If prediction 1 fails — the gate hands the fork to the earworm — consensus
confidence is weaker than #10b concluded, and that is the headline instead.

Setup: songbook seeded 3x taught (demo_weight=3) as in #10; earworm injected
E times (taught); performances at g=1.0 prompted with the TRUE songs sharing
the earworm's motif (every prompt ends at the fork). Modes: static (worm dose
fixed) and self-taught (overrides re-taught, worm can grow).

numpy only; CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embodied"))

from holo_memory import HoloEpisodicMemory, PhasorEncoder, VQCodebook  # noqa: E402
from melody import K, PHRASE_LEN, MelodyModel, Songbook  # noqa: E402

CUE_LEN = 3
THETA = 0.01
N_SONGS = 16
SEED_REPS = 3
PROMPT = 4
G = 1.0


def one_hot(sym: int) -> np.ndarray:
    v = np.zeros(K)
    v[sym] = 1.0
    return v


class ESN:
    def __init__(self, n_res: int, seed: int, leak: float = 0.3,
                 rho: float = 0.9, eta: float = 0.05) -> None:
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1, (n_res, n_res))
        W *= rho / max(abs(np.linalg.eigvals(W)).max(), 1e-9)
        self.W, self.W_in = W, rng.normal(0, 1, (n_res, K)) * 0.5
        self.W_out = np.zeros((K, n_res))
        self.leak, self.eta = leak, eta
        self.state = np.zeros(n_res)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, x: np.ndarray) -> None:
        pre = self.W @ self.state + self.W_in @ x
        self.state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)

    def predict(self) -> np.ndarray:
        z = self.W_out @ self.state
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def learn(self, target: int) -> None:
        p = self.predict()
        g = -p
        g[target] += 1.0
        self.W_out += self.eta * np.outer(g, self.state)


def build_earworm(book: Songbook, model: MelodyModel,
                  rng: np.random.Generator) -> tuple[np.ndarray, int, list[int]]:
    """A phrase opening with a shared motif, diverging at PROMPT into
    a continuation absent from the songbook."""
    counts = np.bincount(book.motif_of, minlength=len(book.motifs))
    # UNIQUE-host motif: the fork must be unambiguous pre-infection, so the
    # gate demonstrably fires at dose 0 and any change is the worm's doing.
    # (First smoke used the most-shared motif — a fork already ambiguous
    # among true songs, gate closed at dose 0, nothing to capture.)
    uniq = [m for m in range(len(book.motifs)) if counts[m] == 1]
    m_star = uniq[0] if uniq else int(np.argmin(
        np.where(counts > 0, counts, 99)))
    hosts = [i for i in range(N_SONGS) if book.motif_of[i] == m_star]
    true_next = {int(book.phrases[i][PROMPT]) for i in hosts}
    grams = set()
    for ph in book.phrases:
        for i in range(PHRASE_LEN - 3):
            grams.add(tuple(int(x) for x in ph[i:i + 4]))
    while True:
        worm = list(book.motifs[m_star])
        for pos in range(4, PHRASE_LEN):
            worm.append(model.sample_from(worm[-1], pos, rng))
        w = np.array(worm, dtype=int)
        novel = all(tuple(w[i:i + 4]) not in grams
                    for i in range(PROMPT, PHRASE_LEN - 3))
        if int(w[PROMPT]) not in true_next and novel:
            return w, m_star, hosts


def run(dose: int, self_taught: bool, tau: float, seed: int,
        n_perf: int) -> dict:
    rng = np.random.default_rng(seed)
    model = MelodyModel(tau=tau)
    book = Songbook(N_SONGS, 8, model, rng)
    worm, m_star, hosts = build_earworm(book, model, rng)

    esn = ESN(256, seed)
    encoder = PhasorEncoder(obs_dim=K, n=512, scale=1.0, seed=seed)
    mem = HoloEpisodicMemory(encoder, VQCodebook(np.eye(K), encoder),
                             budget=PHRASE_LEN + 4, demo_weight=3.0)

    for _ in range(SEED_REPS):
        for pid in rng.permutation(N_SONGS):
            phrase, val = book.phrases[pid], book.valence[pid]
            esn.reset()
            for pos in range(PHRASE_LEN):
                sym = int(phrase[pos])
                if pos > 0:
                    esn.learn(sym)
                mem.observe(one_hot(sym), tag=float(val[pos]), taught=True)
                esn.step(one_hot(sym))
            mem.boundary()

    worm_tags = -(model.surprisal(worm) - book.mu) / book.sd
    for _ in range(dose):
        for pos in range(PHRASE_LEN):
            mem.observe(one_hot(int(worm[pos])), tag=float(worm_tags[pos]),
                        taught=True)
        mem.boundary()
    worm_slides_0 = dose

    grams_worm = {tuple(int(x) for x in worm[i:i + 4])
                  for i in range(PHRASE_LEN - 3)}

    fork_fire = fork_true = fork_worm = fork_n = 0
    fork_agree = []
    rides = []          # fraction of worm followed after chance/any entry
    worm_gram = worm_gram_n = 0
    taught_worm_added = 0
    for _ in range(n_perf):
        pid = int(rng.choice(hosts))
        phrase = book.phrases[pid]
        esn.reset()
        played: list[int] = []
        in_worm = False
        worm_pos = -1
        ride_len = ride_start = 0
        for pos in range(PHRASE_LEN):
            was_ovr = False
            if pos < PROMPT:
                sym = int(phrase[pos])
            else:
                p = esn.predict().copy()
                p[played[-1]] = 0.0
                p /= p.sum()
                sym = int(rng.choice(K, p=p))
                cue = [one_hot(s) for s in played[-CUE_LEN:]]
                res = mem.query(cue, horizon=4, top_m=1)
                fired = False
                if res and res[0]["next_symbols"]:
                    if res[0]["margin_agree"] >= THETA:
                        fired = True
                        cand = int(res[0]["next_symbols"][0])
                        if cand != played[-1] and rng.random() < G:
                            sym = cand
                            was_ovr = True
                if pos == PROMPT:                     # the fork
                    fork_n += 1
                    fork_fire += int(fired)
                    if res:
                        fork_agree.append(res[0]["margin_agree"])
                    fork_true += int(sym == int(phrase[pos]))
                    fork_worm += int(sym == int(worm[pos]))
                    if sym == int(worm[pos]):
                        in_worm, worm_pos = True, pos
                        ride_start = pos
                elif in_worm and worm_pos + 1 < PHRASE_LEN:
                    worm_pos += 1
                    if sym == int(worm[worm_pos]):
                        ride_len += 1
                    else:
                        in_worm = False
            tag = float(-(model.surprisal(
                np.array(played[-1:] + [sym]))[-1] - book.mu) / book.sd) \
                if played else 0.0
            taught_now = self_taught and pos >= PROMPT and was_ovr
            mem.observe(one_hot(sym), tag=tag, taught=taught_now)
            if taught_now and tuple(played[-3:] + [sym]) in grams_worm:
                taught_worm_added += 1
            esn.step(one_hot(sym))
            played.append(sym)
        mem.boundary()
        if ride_start:
            rides.append(ride_len / max(PHRASE_LEN - 1 - ride_start, 1))
        for i in range(PHRASE_LEN - 3):
            worm_gram += int(tuple(played[i:i + 4]) in grams_worm)
            worm_gram_n += 1

    return {
        "fork_fire": fork_fire / max(fork_n, 1),
        "fork_true": fork_true / max(fork_n, 1),
        "fork_worm": fork_worm / max(fork_n, 1),
        "fork_agree": float(np.mean(fork_agree)) if fork_agree else 0.0,
        "ride": float(np.mean(rides)) if rides else 0.0,
        "n_rides": len(rides),
        "worm_gram": worm_gram / max(worm_gram_n, 1),
        "taught_added": taught_worm_added,
        "dose0": worm_slides_0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", type=int, nargs="+", default=[0, 1, 4, 16])
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-perf", type=int, default=40)
    args = ap.parse_args()

    print(f"earworm — tau={args.tau} seeds={args.seeds} g={G} theta={THETA} "
          f"n_perf={args.n_perf} (prompts = true songs sharing the worm's motif)")
    hdr = (f"{'dose':>4} {'mode':>6} | {'forkfire':>8} {'agree@fork':>10} | "
           f"{'fork_true':>9} {'fork_worm':>9} | {'ride':>5} {'#rides':>6} | "
           f"{'wormgram':>8} {'w_taught+':>9}")
    print(hdr)
    print("-" * len(hdr))
    for dose in args.doses:
        for st in (False, True):
            rows = [run(dose, st, args.tau, s, args.n_perf)
                    for s in range(args.seeds)]
            def agg(k):
                return float(np.mean([r[k] for r in rows]))
            mode = "taught" if st else "static"
            print(f"{dose:>4} {mode:>6} | {agg('fork_fire'):>8.3f} "
                  f"{agg('fork_agree'):>10.3f} | {agg('fork_true'):>9.3f} "
                  f"{agg('fork_worm'):>9.3f} | {agg('ride'):>5.3f} "
                  f"{agg('n_rides'):>6.1f} | {agg('worm_gram'):>8.3f} "
                  f"{agg('taught_added'):>9.1f}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
