"""Gate — score-margin confidence gating for episodic recall (9c prep; file 10).

#9a finding 4: at low grammar temperature with big songbooks, wrong-phrase
recalls actively mislead next-note prediction (reel 0.446 < grammar 0.467 at
tau=0.4, P=256). The #6-flagged fix: gate recall on its confidence and fall
back to the parametric predictor when the reel isn't sure.

query() now returns two confidence signals (holo_memory):
  margin       — relative lead over the runner-up in the full ranking
                 (usually a same-slide neighbor: position confidence)
  margin_slide — relative lead over the best candidate from a DIFFERENT
                 slide (identity confidence — the #9a failure was identity)

Policy tested:  predict with the reel iff confidence >= theta, else with the
grammar argmax. One query pass per config; theta curves are post-hoc.

Pre-registered predictions:
  1. margin_slide is the informative signal (identity errors are the failure
     mode; raw margin under-reads confident recalls via window overlap).
  2. A well-chosen theta makes gated accuracy >= max(reel-only, grammar-only)
     in every cell, closing #9a's inversion without hurting the high-tau
     cells where the reel dominates.

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
THETAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]


def one_hot(sym: int) -> np.ndarray:
    v = np.zeros(K)
    v[sym] = 1.0
    return v


def collect(tau: float, n_phrases: int, seed: int, n_motifs: int,
            test_phrases: int) -> list[dict]:
    """One record+query pass; returns per-query outcome rows."""
    rng = np.random.default_rng(seed)
    model = MelodyModel(tau=tau)
    book = Songbook(n_phrases, n_motifs, model, rng)

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

    rows = []
    for pid in rng.permutation(n_phrases)[:min(test_phrases, n_phrases)]:
        phrase = book.phrases[pid]
        for t in range(CUE_LEN - 1, PHRASE_LEN - 1):
            cue = [one_hot(int(phrase[t - j]))
                   for j in range(CUE_LEN - 1, -1, -1)]
            res = mem.query(cue, horizon=1, top_m=1)
            if not res or not res[0]["next_symbols"]:
                continue
            m = res[0]
            gram_pred = int(np.argmax(model.next_dist(int(phrase[t]), t + 1)))
            rows.append({
                "margin": m["margin"],
                "margin_slide": m["margin_slide"],
                "reel_ok": int(m["next_symbols"][0] == phrase[t + 1]),
                "gram_ok": int(gram_pred == phrase[t + 1]),
                "id_ok": int(slide_to_phrase.get(m["slide"], -1) == pid),
            })
    return rows


def gate_curve(rows: list[dict], signal: str) -> list[tuple]:
    out = []
    conf = np.array([r[signal] for r in rows])
    reel = np.array([r["reel_ok"] for r in rows])
    gram = np.array([r["gram_ok"] for r in rows])
    for th in THETAS:
        use = conf >= th
        gated = np.where(use, reel, gram)
        cov = use.mean()
        prec = reel[use].mean() if use.any() else float("nan")
        out.append((th, cov, prec, gated.mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.4, 1.0, 2.5])
    ap.add_argument("--phrases", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--motifs", type=int, default=8)
    ap.add_argument("--test-phrases", type=int, default=48)
    args = ap.parse_args()

    print(f"gate — taus={args.taus} phrases={args.phrases} seeds={args.seeds} "
          f"signals: margin_slide (primary) vs margin")
    for tau in args.taus:
        for P in args.phrases:
            rows = []
            for s in range(args.seeds):
                rows += collect(tau, P, s, args.motifs, args.test_phrases)
            reel_only = np.mean([r["reel_ok"] for r in rows])
            gram_only = np.mean([r["gram_ok"] for r in rows])
            print(f"\ntau={tau} P={P}  (n={len(rows)})  "
                  f"reel-only={reel_only:.3f}  grammar-only={gram_only:.3f}")
            for signal in ("margin_slide", "margin"):
                curve = gate_curve(rows, signal)
                best = max(curve, key=lambda c: c[3])
                line = "  ".join(f"θ{th:.2f}:{acc:.3f}" for th, _, _, acc in curve)
                print(f"  {signal:>12}: {line}")
                print(f"  {'':>12}  best θ={best[0]:.2f} → acc={best[3]:.3f} "
                      f"(coverage {best[1]:.0%}, precision-when-used {best[2]:.3f})",
                      flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
