"""Affect — phrase-level valence events: making foresight a fair fight (file 11).

#9a's methods discovery: surprisal-derived valence is parametrically saturated
(r_markov ≈ 0.94-0.96) — the grammar predicts Δvalence almost perfectly, so
episodic recall can only lose on the foresight axis. The repair: give each
phrase ONE valence spike at a phrase-specific position with a random sign
("the moment in this song") — value the grammar cannot see, amplitude A.
A phrase-constant offset would cancel out of Δvalence; episodic value must
be an event.

Three forecasters of realized Δvalence, correlated over all query points:
  r_markov : grammar-only Monte-Carlo oracle (blind to spikes by construction)
  r_reel   : dv̂ = mean(next_tags) - matched_tag from top-1 recall, ungated
  r_gated  : the dual-read policy from the Gate experiment — reel dv̂ iff
             margin_slide >= theta (strict identity gate), else the markov
             forecast. First live test of "strictly gate value import."

Pre-registered predictions:
  1. adv = r_reel - r_markov rises with A and crosses zero: the crossover A*
     measures how much memory-only value the life must contain before
     episodic foresight pays. At A=0 this reproduces #9a (adv <= 0).
  2. r_gated >= max(r_reel, r_markov) across the whole A sweep — the hybrid
     never loses, because misidentified recalls import WRONG spikes (worse
     than no spike knowledge) and the strict gate filters exactly those.

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
HORIZON = 4
THETA = 0.01          # strict identity gate (Gate experiment: 0.99 precision)


def one_hot(sym: int) -> np.ndarray:
    v = np.zeros(K)
    v[sym] = 1.0
    return v


def run_config(tau: float, n_phrases: int, amp: float, seed: int,
               n_motifs: int, test_phrases: int, n_roll: int) -> dict:
    rng = np.random.default_rng(seed)
    model = MelodyModel(tau=tau)
    book = Songbook(n_phrases, n_motifs, model, rng, affect_amp=amp)

    encoder = PhasorEncoder(obs_dim=K, n=512, scale=1.0, seed=seed)
    mem = HoloEpisodicMemory(encoder, VQCodebook(np.eye(K), encoder),
                             budget=PHRASE_LEN + 4)
    order = rng.permutation(n_phrases)
    slide_to_phrase = {}
    for slide_idx, pid in enumerate(order):
        for pos in range(PHRASE_LEN):
            mem.observe(one_hot(int(book.phrases[pid][pos])),
                        tag=float(book.valence[pid][pos]))
        slide_to_phrase[slide_idx] = int(pid)
        mem.boundary()

    dv_reel, dv_markov, dv_gated, dv_real = [], [], [], []
    id_hits, gate_used = 0, 0
    n = 0
    for pid in rng.permutation(n_phrases)[:min(test_phrases, n_phrases)]:
        phrase = book.phrases[pid]
        for t in range(CUE_LEN - 1, PHRASE_LEN - 1):
            cue = [one_hot(int(phrase[t - j]))
                   for j in range(CUE_LEN - 1, -1, -1)]
            res = mem.query(cue, horizon=HORIZON, top_m=1)
            if not res or not res[0]["next_tags"]:
                continue
            m = res[0]
            n += 1
            id_hits += int(slide_to_phrase.get(m["slide"], -1) == pid)
            hat = float(np.mean(m["next_tags"]) - m["matched_tag"])
            oracle = book.markov_dv(pid, t, HORIZON, rng, n_roll)
            if m["margin_slide"] >= THETA:
                gated = hat
                gate_used += 1
            else:
                gated = oracle
            dv_reel.append(hat)
            dv_markov.append(oracle)
            dv_gated.append(gated)
            dv_real.append(book.realized_dv(pid, t, HORIZON))

    def corr(a, b):
        a, b = np.asarray(a), np.asarray(b)
        if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "r_reel": corr(dv_reel, dv_real),
        "r_markov": corr(dv_markov, dv_real),
        "r_gated": corr(dv_gated, dv_real),
        "id": id_hits / max(n, 1),
        "cov": gate_used / max(n, 1),
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0])
    ap.add_argument("--phrases", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--amps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--motifs", type=int, default=8)
    ap.add_argument("--test-phrases", type=int, default=48)
    ap.add_argument("--rollouts", type=int, default=128)
    args = ap.parse_args()

    print(f"affect — taus={args.taus} phrases={args.phrases} amps={args.amps} "
          f"seeds={args.seeds} theta={THETA} horizon={HORIZON}")
    hdr = (f"{'tau':>4} {'P':>4} {'A':>4} | {'r_reel':>7} {'r_markov':>8} "
           f"{'r_gated':>8} | {'adv':>7} {'adv_g':>7} | {'id%':>5} {'cov%':>5}")
    print(hdr)
    print("-" * len(hdr))
    for tau in args.taus:
        for P in args.phrases:
            for A in args.amps:
                rows = [run_config(tau, P, A, s, args.motifs,
                                   args.test_phrases, args.rollouts)
                        for s in range(args.seeds)]
                def agg(k):
                    return float(np.nanmean([r[k] for r in rows]))
                rr, rm, rg = agg("r_reel"), agg("r_markov"), agg("r_gated")
                print(f"{tau:>4.1f} {P:>4} {A:>4.1f} | {rr:>7.3f} {rm:>8.3f} "
                      f"{rg:>8.3f} | {rr-rm:>+7.3f} {rg-max(rr,rm):>+7.3f} | "
                      f"{100*agg('id'):>4.1f}% {100*agg('cov'):>4.1f}%",
                      flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
