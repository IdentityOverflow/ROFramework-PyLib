"""#F3b — Controls for the Seed arm of #F3.

(a) Baselines for the correlation dimension: the drive alone (the 4-sine
    signal mixture, per-node drive vectors) and the Seed with coupling
    weights set to ZERO (drive + noise floor, no loop). Any D2 above these
    is the loop's own contribution.
(b) Gain scan around the self-tuned weights: let Rule 2a run from w_scale=1,
    snapshot the weight matrix W*, then freeze and run alpha·W* for a range
    of alpha. The question "does the homeostat park the loop at the
    transition?" is only meaningful on the curve through the point it
    actually reached — not on the hand-scaled positive-weight curve, which
    the tuned weights (sign-mixed) are not on.

Pre-registered reading for (b): alpha=1 sits at the knee — the largest D2
that is not yet saturation-dominated, with lambda at its maximum — if
"sigma→1 parks at the fractal band" is the right description. If D2 keeps
rising past alpha=1 before saturation, Rule 2a is conservative (parks below
the band); if alpha=1 is already saturation-dominated it overshoots.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fractal_tools import correlation_dimension, pca_project  # noqa: E402

spec = importlib.util.spec_from_file_location("f3", HERE / "03_reservoir_gain_sweep.py")
F3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F3)


def drive_only_d2(seed: int, T: int = 2500, burn: int = 1500) -> float:
    cfg, nodes = F3.build_seed(64, seed, 0.0, 1.0)
    ids = sorted(nodes)
    D = np.array([[F3.seed_drive(t, nodes, cfg)[i] for i in ids] for t in range(burn, burn + T)])
    d2, _ = correlation_dimension(pca_project(D, 20), rng=np.random.default_rng(seed))
    return float(d2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save", default=str(HERE / "results_04_tuned_scan.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"drive_d2": [], "zero_w": [], "scan": []}

    print("(a) baselines")
    for s in range(args.seeds):
        res["drive_d2"].append(drive_only_d2(s))
    print(f"  drive alone: D2 = {np.mean(res['drive_d2']):.2f} ± {np.std(res['drive_d2']):.2f}")
    rows = [F3.seed_arm(0.0, 0.0, s) for s in range(args.seeds)]
    res["zero_w"] = rows
    m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
    print(f"  zero coupling (drive+noise): D2={m['d2']:.2f} PR={m['pr']:.2f} var={m['var']:.4f} λ={m['lyap']:+.3f}")
    print(f"  ({time.time()-t0:.0f}s)")

    print("\n(b) scan alpha·W* around the self-tuned weights (Rule 2a from w_scale=1, 4000 steps)")
    print(f"  {'alpha':>5} {'σ':>5} {'<w>':>6} {'neg%':>5} {'D2':>6} {'±':>5} {'PR':>6} {'var':>7} {'sat':>6} {'λ':>7}")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for s in range(args.seeds):
        cfg, nodes = F3.build_seed(64, s, 0.01, 1.0)
        rng = np.random.default_rng(s)
        for t in range(4000):
            F3.S01.minimal_step(nodes, F3.seed_drive(t, nodes, cfg), rng)
        for n in nodes.values():
            n._config = copy.copy(cfg)
            n._config.learning_rate = 0.0
        for a in alphas:
            nn = copy.deepcopy(nodes)
            for n in nn.values():
                for k in n.coupling_weights:
                    n.coupling_weights[k] *= a
            out = F3.measure_nodes(cfg, nn, 100 + s, T=2500, burn=500, t_offset=4000)
            out.update(alpha=a, seed=s)
            res["scan"].append(out)
    for a in alphas:
        rows = [r for r in res["scan"] if r["alpha"] == a]
        m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
        print(f"  {a:5.2f} {m['sigma']:5.2f} {m['mean_w']:6.3f} {100*m['frac_neg_w']:5.1f} {m['d2']:6.2f} "
              f"{m['d2_spread']:5.2f} {m['pr']:6.2f} {m['var']:7.4f} {m['sat']:6.3f} {m['lyap']:+7.3f}")
    print(f"  ({time.time()-t0:.0f}s)")
    Path(args.save).write_text(json.dumps(res, indent=1, default=float))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
