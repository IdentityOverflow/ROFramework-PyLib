"""#F1 — Spatial scale invariance of the Seed (finishing seed_architecture §4.2).

Experiment seed/01 verified the TEMPORAL half of the fractal claim (cascade
and avalanche sizes are power-law distributed under Clauset MLE + KS). The
SPATIAL half — pattern extent P(r) ~ r^-beta — has a library function,
`measure_scale_distribution`, that no experiment or test ever called. This
runs it.

Method
------
Same minimal Seed as seed/01 (ring lattice, Rule 2a only, sparse regime),
independent random pulse drives (the clean condition). Full activation
histories are recorded (the per-node deques keep only ~80 cycles, far too
short for pairwise MI on fast nodes) and handed to the library function,
which builds the pairwise-MI graph, thresholds it, and tests whether the
connected-component sizes follow a power law.

Controls:
  shuffled   each node's history circularly shifted by a random offset —
             destroys cross-node correlation, keeps marginals. Any power
             law surviving this is an artifact of the threshold graph.
  frozen_lo  Rule 2a off, initial weights x0.3 (subcritical by construction).
  frozen_hi  Rule 2a off, initial weights x3.0 (supercritical).

MI threshold: the library default (0.1) plus a data-driven one — the 99th
percentile of the shuffled-surrogate MI distribution — swept over a small
range, because the component-size distribution of a thresholded graph is
threshold-sensitive and an honest result has to show the dependence.

Pre-registered reading
----------------------
Spec §4.2 predicts: critical (Rule 2a on) → PASS at the surrogate-calibrated
threshold, with the pass disappearing in shuffled, frozen_lo (components all
size 1) and frozen_hi (one giant component). Anything else is a finding
against the spec's spatial claim or against the function as written.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from ro_framework.seed.criticality import (  # noqa: E402
    extract_cascades, fast_mi, measure_scale_distribution, verify_power_law,
)
from ro_framework.seed.node import SeedConfig  # noqa: E402


def _load_seed01():
    spec = importlib.util.spec_from_file_location(
        "seed01", HERE.parents[0] / "seed" / "01_criticality_validation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S01 = _load_seed01()


def run_seed(n_nodes: int, n_steps: int, seed: int, learning_rate: float,
             weight_scale: float):
    cfg = SeedConfig(n_init=n_nodes, k_neighbors=6, noise_floor=0.05,
                     drive_amplitude=0.2, activation_threshold=0.5,
                     learning_rate=learning_rate, w_max=2.0,
                     prune_weight_threshold=0.005, prune_weight_window=200)
    rng = np.random.default_rng(seed)
    nodes = S01.build_ring_lattice(cfg, seed=seed)
    if weight_scale != 1.0:
        for n in nodes.values():
            for k in n.coupling_weights:
                n.coupling_weights[k] = float(
                    min(cfg.w_max, n.coupling_weights[k] * weight_scale))
    hist = {nid: [] for nid in nodes}
    active = []
    for _ in range(n_steps):
        drives = S01.independent_random_drives(list(nodes.keys()), rng)
        S01.minimal_step(nodes, drives, rng)
        for nid, n in nodes.items():
            hist[nid].append(n.activation)
        active.append(sum(1 for n in nodes.values()
                          if abs(n.activation) > cfg.activation_threshold))
    sigma = float(np.mean([n.branching_ratio for n in nodes.values()]))
    return cfg, nodes, {k: np.array(v) for k, v in hist.items()}, np.array(active), sigma


def install_histories(nodes, hist, burn: int):
    """Replace each node's short deque with its full recorded history."""
    for nid, n in nodes.items():
        n.activation_history = deque(hist[nid][burn:])


def surrogate_threshold(hist, burn: int, rng, n_pairs: int = 400,
                        bins: int = 8, q: float = 0.99) -> float:
    ids = list(hist)
    T = len(hist[ids[0]]) - burn
    mis = []
    for _ in range(n_pairs):
        a, b = rng.choice(len(ids), 2, replace=False)
        x = hist[ids[a]][burn:]
        y = np.roll(hist[ids[b]][burn:], int(rng.integers(T // 4, 3 * T // 4)))
        mis.append(fast_mi(x, y, bins))
    return float(np.quantile(mis, q))


def shuffled(hist, burn, rng):
    out = {}
    T = len(next(iter(hist.values()))) - burn
    for nid, h in hist.items():
        out[nid] = np.concatenate([h[:burn], np.roll(h[burn:], int(rng.integers(T // 4, 3 * T // 4)))])
    return out


def component_sizes(nodes, threshold, bins=8):
    """Same graph the library builds, returned raw for reporting."""
    from scipy import sparse
    ids = sorted(nodes)
    H = np.array([np.array(nodes[i].activation_history) for i in ids])
    n = len(ids)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if fast_mi(H[i], H[j], bins) > threshold:
                adj[i, j] = adj[j, i] = True
    _, labels = sparse.csgraph.connected_components(sparse.csr_matrix(adj), directed=False)
    _, counts = np.unique(labels, return_counts=True)
    return sorted(counts.tolist(), reverse=True), int(adj.sum() // 2)


def correlation_vs_distance(hist, burn, n_dist=None):
    """Spatial correlation function along the ring.

    Node ids n{i} index the frequency-sorted ring; ring distance
    d(i,j) = min(|i-j|, n-|i-j|). Returns (d, C(d)) with C the mean |Pearson|
    over all pairs at that distance, plus power-law vs exponential fits of
    the decay (R² of log C vs log d, and of log C vs d), d >= 1.
    """
    ids = sorted(hist, key=lambda k: int(k[1:]))
    n = len(ids)
    H = np.array([hist[i][burn:] for i in ids])
    H = H - H.mean(axis=1, keepdims=True)
    sd = H.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = H / sd
    R = np.abs(Z @ Z.T / Z.shape[1])
    dmax = n // 2 if n_dist is None else n_dist
    C = np.zeros(dmax + 1)
    for d in range(1, dmax + 1):
        vals = [R[i, (i + d) % n] for i in range(n)]
        C[d] = float(np.mean(vals))
    d = np.arange(1, dmax + 1)
    c = C[1:]
    sel = c > 1e-6
    out = dict(d=d, C=c)
    if sel.sum() >= 4:
        for name, x in (("power", np.log(d[sel])), ("exp", d[sel].astype(float))):
            y = np.log(c[sel])
            slope, icpt = np.polyfit(x, y, 1)
            pred = slope * x + icpt
            r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
            out[name] = (float(slope), float(r2))
    # correlation length: first distance where C drops below C(1)/e
    thr = c[0] / np.e
    below = np.where(c < thr)[0]
    out["xi"] = float(d[below[0]]) if len(below) else float("inf")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    conditions = {
        "critical":  dict(learning_rate=0.01, weight_scale=1.0),
        "frozen_lo": dict(learning_rate=0.0, weight_scale=0.3),
        "frozen_hi": dict(learning_rate=0.0, weight_scale=3.0),
    }
    print(f"#F1 Seed spatial scale: {args.nodes} nodes, {args.steps} steps, "
          f"burn {args.burn}, seeds {args.seeds}")
    print("=" * 78)
    pooled = {}
    cvd_rows = []
    for cname, kw in conditions.items():
        for seed in range(args.seeds):
            t0 = time.time()
            cfg, nodes, hist, active, sigma = run_seed(args.nodes, args.steps, 42 + seed, **kw)
            rng = np.random.default_rng(1000 + seed)
            casc = []
            for h in hist.values():
                casc.extend(extract_cascades(h[args.burn:], threshold=cfg.activation_threshold))
            tpl = verify_power_law(casc) if len(casc) >= 50 else (False, 0.0, 1.0)
            cvd = correlation_vs_distance(hist, args.burn)
            cvd_rows.append((cname, seed, sigma, cvd))
            thr_sur = surrogate_threshold(hist, args.burn, rng)
            thresholds = sorted({0.1, round(thr_sur, 4), round(1.5 * thr_sur, 4), round(3 * thr_sur, 4)})
            print(f"\n[{cname} seed={seed}] σ={sigma:.3f} active={active[args.burn:].mean()/args.nodes:.1%} "
                  f"temporal PL: {'PASS' if tpl[0] else 'no'} α={tpl[1]:.2f} KS={tpl[2]:.3f} "
                  f"| surrogate MI q99={thr_sur:.4f}  ({time.time()-t0:.1f}s)")
            pw, ex = cvd.get("power", (np.nan, np.nan)), cvd.get("exp", (np.nan, np.nan))
            print(f"    C(d) along ring: C(1)={cvd['C'][0]:.3f} C(4)={cvd['C'][3]:.3f} "
                  f"C(16)={cvd['C'][15]:.3f} C(max)={cvd['C'][-1]:.3f} ξ={cvd['xi']:.0f} | "
                  f"power slope={pw[0]:.2f} R²={pw[1]:.3f}  exp slope={ex[0]:.3f} R²={ex[1]:.3f}")
            for label, H in (("real", hist), ("shuffled", shuffled(hist, args.burn, rng))):
                install_histories(nodes, H, args.burn)
                for thr in thresholds:
                    ok, beta, ks = measure_scale_distribution(nodes, threshold=thr)
                    sizes, n_edges = component_sizes(nodes, thr)
                    tag = "thr=lib0.1" if thr == 0.1 else f"thr={thr:.4f}"
                    key = (cname, label, tag)
                    pooled.setdefault(key, []).extend(sizes)
                    top = sizes[:5]
                    print(f"    {label:9s} {tag:12s} edges={n_edges:5d} comps={len(sizes):4d} "
                          f"top={top}  → {'PASS' if ok else 'no  '} β={beta:.2f} KS={ks:.3f}")

    print("\n" + "=" * 78)
    print("Pooled component sizes across seeds (verify_power_law, min_samples=3, x_min=1):")
    for key in sorted(pooled):
        sizes = pooled[key]
        ok, beta, ks = verify_power_law(sizes, min_samples=3, x_min=1)
        gt1 = sum(1 for s in sizes if s > 1)
        print(f"  {key[0]:9s} {key[1]:9s} {key[2]:12s} n={len(sizes):4d} (>1: {gt1:3d}) "
              f"max={max(sizes):4d} → {'PASS' if ok else 'no  '} β={beta:.2f} KS={ks:.3f}")

    print("\nSpatial correlation decay along the ring (the §4.2 pattern-extent test):")
    print(f"  {'cond':9s} {'seed':>4} {'σ':>6} {'C(1)':>6} {'C(8)':>6} {'C(32)':>6} {'ξ':>5} "
          f"{'pow slope':>9} {'R²':>6} {'exp slope':>9} {'R²':>6}")
    dump = []
    for cname, seed, sigma, cvd in cvd_rows:
        pw, ex = cvd.get("power", (np.nan, np.nan)), cvd.get("exp", (np.nan, np.nan))
        C = cvd["C"]
        print(f"  {cname:9s} {seed:4d} {sigma:6.2f} {C[0]:6.3f} {C[min(7,len(C)-1)]:6.3f} "
              f"{C[min(31,len(C)-1)]:6.3f} {cvd['xi']:5.0f} {pw[0]:9.2f} {pw[1]:6.3f} {ex[0]:9.3f} {ex[1]:6.3f}")
        dump.append(dict(cond=cname, seed=seed, sigma=sigma, d=cvd["d"].tolist(), C=C.tolist(),
                         power=pw, exp=ex, xi=cvd["xi"]))
    import json
    out = HERE / "results_01_seed_spatial.json"
    out.write_text(json.dumps(dict(cvd=dump, pooled={"/".join(k): v for k, v in pooled.items()}), default=float))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
