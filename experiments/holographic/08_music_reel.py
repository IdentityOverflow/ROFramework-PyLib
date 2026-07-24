"""#9a — The reel reads sheet music (experiment file 08; see 09_music_plan.md).

Question: with a natively symbolic stream of exactly known statistics, what are
the reel's recall/foresight numbers relative to computable ceilings?

Protocol
--------
Record pass:  a songbook of P phrases (16 notes each, motif incipits, cadence
              rule) is performed once, one phrase per slide, valence tags
              v = z(-listener surprisal).
Test pass:    the same phrases are re-performed (up to --test-phrases of them).
              At each position t >= 2 the reel is queried with the last-3-note
              trajectory stub and its top match is scored:
                * identity   — did it recall the right phrase? right position?
                * next-note  — recalled continuation vs actual, against the
                               grammar-optimal predictor (argmax of the true
                               next distribution)
                * foresight  — dv̂ = mean(next_tags) - matched_tag vs realized
                               Δvalence, correlated over all query points;
                               reported next to r_markov, the same correlation
                               for the grammar-only Monte-Carlo oracle.
              Perfect episodic recall gives r = 1.0 by construction, so
              r_reel - r_markov is the episodic advantage over the parametric
              ceiling — the number this experiment exists to measure.

Sweep: tau (grammar temperature) x P (songbook size). Prediction: episodic
advantage grows with tau — grammar-predictable streams leave recall nothing to
add; atonal-but-memorized streams are where episodic memory earns its keep.

Music-specific reel setup: the VQ layer is bypassed with a one-hot codebook
(prototypes = I_K), so quantization is exact and cues are clean — the #5/#6
codebook-noise layer is deliberately absent.

numpy only; CPU; ~seconds per config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embodied"))

from holo_memory import HoloEpisodicMemory, PhasorEncoder, VQCodebook  # noqa: E402
from melody import (K, PHRASE_LEN, MelodyModel, Songbook)  # noqa: E402

CUE_LEN = 3
HORIZON = 4


def one_hot(sym: int) -> np.ndarray:
    v = np.zeros(K)
    v[sym] = 1.0
    return v


def run_config(tau: float, n_phrases: int, seed: int, n_motifs: int,
               test_phrases: int, n_roll: int) -> dict:
    rng = np.random.default_rng(seed)
    model = MelodyModel(tau=tau)
    book = Songbook(n_phrases, n_motifs, model, rng)

    encoder = PhasorEncoder(obs_dim=K, n=512, scale=1.0, seed=seed)
    codebook = VQCodebook(np.eye(K), encoder)
    mem = HoloEpisodicMemory(encoder, codebook, budget=PHRASE_LEN + 4)

    # ── record pass: one performance per phrase, one phrase per slide ──
    order = rng.permutation(n_phrases)
    slide_to_phrase: dict[int, int] = {}
    for slide_idx, pid in enumerate(order):
        for pos in range(PHRASE_LEN):
            mem.observe(one_hot(int(book.phrases[pid][pos])),
                        tag=float(book.valence[pid][pos]))
        slide_to_phrase[slide_idx] = int(pid)
        mem.boundary()

    # ── test pass: re-performances, query only ──
    test_ids = rng.permutation(n_phrases)[:min(test_phrases, n_phrases)]
    id_hit = pos_hit = 0
    next_hit_reel = next_hit_gram = 0
    n_next = 0
    dv_hat, dv_gram, dv_real = [], [], []

    for pid in test_ids:
        phrase = book.phrases[pid]
        for t in range(CUE_LEN - 1, PHRASE_LEN - 1):
            cue = [one_hot(int(phrase[t - j])) for j in range(CUE_LEN - 1, -1, -1)]
            res = mem.query(cue, horizon=HORIZON, top_m=1)
            if not res:
                continue
            m = res[0]
            id_hit += int(slide_to_phrase.get(m["slide"], -1) == pid)
            pos_hit += int(slide_to_phrase.get(m["slide"], -1) == pid
                           and m["position"] == t)
            # next-note accuracy
            if m["next_symbols"]:
                next_hit_reel += int(m["next_symbols"][0] == phrase[t + 1])
            gram_pred = int(np.argmax(model.next_dist(int(phrase[t]), t + 1)))
            next_hit_gram += int(gram_pred == phrase[t + 1])
            n_next += 1
            # foresight
            if m["next_tags"]:
                dv_hat.append(float(np.mean(m["next_tags"]) - m["matched_tag"]))
                dv_gram.append(book.markov_dv(pid, t, HORIZON, rng, n_roll))
                dv_real.append(book.realized_dv(pid, t, HORIZON))

    def corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n_queries": n_next,
        "id_acc": id_hit / max(n_next, 1),
        "pos_acc": pos_hit / max(n_next, 1),
        "next_reel": next_hit_reel / max(n_next, 1),
        "next_gram": next_hit_gram / max(n_next, 1),
        "r_reel": corr(dv_hat, dv_real),
        "r_markov": corr(dv_gram, dv_real),
        "records": mem.n_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.4, 1.0, 2.5])
    ap.add_argument("--phrases", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--motifs", type=int, default=8)
    ap.add_argument("--test-phrases", type=int, default=48)
    ap.add_argument("--rollouts", type=int, default=128)
    args = ap.parse_args()

    print(f"#9a music reel — taus={args.taus} phrases={args.phrases} "
          f"seeds={args.seeds} motifs={args.motifs}")
    hdr = (f"{'tau':>5} {'P':>4} | {'id%':>6} {'pos%':>6} | "
           f"{'next_reel':>9} {'next_gram':>9} | {'r_reel':>7} {'r_markov':>8} "
           f"| {'adv':>6}")
    print(hdr)
    print("-" * len(hdr))
    for tau in args.taus:
        for P in args.phrases:
            rows = [run_config(tau, P, s, args.motifs, args.test_phrases,
                               args.rollouts) for s in range(args.seeds)]
            def agg(k):
                v = np.array([r[k] for r in rows], dtype=float)
                return np.nanmean(v), np.nanstd(v)
            id_m, _ = agg("id_acc"); pos_m, _ = agg("pos_acc")
            nr_m, _ = agg("next_reel"); ng_m, _ = agg("next_gram")
            rr_m, rr_s = agg("r_reel"); rm_m, _ = agg("r_markov")
            print(f"{tau:>5.1f} {P:>4} | {100*id_m:>5.1f}% {100*pos_m:>5.1f}% | "
                  f"{nr_m:>9.3f} {ng_m:>9.3f} | {rr_m:>7.3f} {rm_m:>8.3f} "
                  f"| {rr_m - rm_m:>+6.3f}  (±{rr_s:.3f}, n={rows[0]['n_queries']})",
                  flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
