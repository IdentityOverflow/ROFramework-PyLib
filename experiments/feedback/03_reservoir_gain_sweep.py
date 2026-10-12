"""#F3 — The same gain axis, read in state space: reservoir and Seed.

The video rig (#F2) puts the fractal band at loop gain ≈ 1 between fade
(g<1) and whiteout (g≫1). Here the loop is a recurrent network and the
"frame" is its state vector; the geometry is measured in state space.

Arms
----
ESN    x_{t+1} = tanh(g · W x_t + W_in u_t), W random with unit spectral
       radius, so g IS the spectral radius. u_t = weak slow sine (the seed
       object). g swept 0.3 → 2.4.
Seed   the ring-lattice Seed of seed/01 under the deterministic correlated
       signal drive. Two sub-arms:
         frozen     Rule 2a OFF, initial coupling weights × w_scale, swept —
                    the gain knob turned by hand.
         self-tuned Rule 2a ON from the same initial scales — the Seed's
                    own homeostat (σ→1) turns the knob. Where does it park?

Measures per gain (after a transient): correlation dimension D2
(Grassberger-Procaccia on the PCA-20 projection of the trajectory),
participation ratio, state variance, saturated fraction (|x|>0.95), and the
largest Lyapunov exponent λ by the Benettin twin-trajectory method (the Seed
twin shares the noise stream, so λ measures the deterministic part).

Pre-registered readings
-----------------------
ESN: D2 ≈ dimension of the drive (≤2) for g<1 (echo state: the state is a
function of the input), rising through a band near g≈1 where λ crosses 0,
then either saturating (tanh lock, variance↑, D2 falls) or chaotic (D2↑).
Seed: the self-tuned arm lands at σ≈1 on the frozen curve's transition — the
homeostat parks the loop where the rig's fractal band is. If it parks
elsewhere, "criticality = the fractal regime" is not what Rule 2a buys.
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
sys.path.insert(0, str(HERE.parents[1] / "src"))
from fractal_tools import correlation_dimension, participation_ratio, pca_project  # noqa: E402
from ro_framework.seed.node import SeedConfig  # noqa: E402


def _load_seed01():
    spec = importlib.util.spec_from_file_location(
        "seed01", HERE.parents[0] / "seed" / "01_criticality_validation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S01 = _load_seed01()


def geometry(traj: np.ndarray, rng) -> dict:
    X = traj
    proj = pca_project(X, 20)
    d2, spread = correlation_dimension(proj, rng=rng)
    return dict(d2=d2, d2_spread=spread, pr=participation_ratio(X),
                var=float(X.var(axis=0).mean()),
                sat=float((np.abs(X) > 0.95).mean()))


# ---------------------------------------------------------------------------
# ESN arm
# ---------------------------------------------------------------------------

def esn_arm(g: float, seed: int, n: int = 300, T: int = 2500, burn: int = 500,
            drive_amp: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.1)
    W /= np.max(np.abs(np.linalg.eigvals(W)))
    w_in = rng.standard_normal(n) * drive_amp
    u = np.sin(2 * np.pi * np.arange(T + burn) / 50.0)
    x = np.zeros(n)
    traj = np.zeros((T, n))
    for t in range(T + burn):
        x = np.tanh(g * W @ x + w_in * u[t])
        if t >= burn:
            traj[t - burn] = x
    # Benettin largest Lyapunov on the driven system
    eps = 1e-8
    y = x + eps * rng.standard_normal(n) / np.sqrt(n)
    lam = 0.0
    for t in range(1000):
        ut = u[(T + burn + t) % len(u)]
        x = np.tanh(g * W @ x + w_in * ut)
        y = np.tanh(g * W @ y + w_in * ut)
        d = np.linalg.norm(y - x)
        if d == 0:
            lam += np.log(1e-16 / eps)
            y = x + eps * rng.standard_normal(n) / np.sqrt(n)
            continue
        lam += np.log(d / eps)
        y = x + (y - x) * (eps / d)
    out = geometry(traj, rng)
    out.update(g=g, lyap=lam / 1000)
    return out


# ---------------------------------------------------------------------------
# Seed arm
# ---------------------------------------------------------------------------

def build_seed(n_nodes, seed, lr, w_scale):
    cfg = SeedConfig(n_init=n_nodes, k_neighbors=6, noise_floor=0.05,
                     drive_amplitude=0.2, activation_threshold=0.5,
                     learning_rate=lr, w_max=2.0,
                     prune_weight_threshold=0.005, prune_weight_window=200)
    nodes = S01.build_ring_lattice(cfg, seed=seed)
    for n in nodes.values():
        for k in n.coupling_weights:
            n.coupling_weights[k] = float(min(cfg.w_max, n.coupling_weights[k] * w_scale))
    return cfg, nodes


def seed_drive(step, nodes, cfg):
    freqs = {nid: n.frequency for nid, n in nodes.items()}
    return S01.correlated_signal_drives(step * cfg.dt, freqs)


def seed_arm(w_scale: float, lr: float, seed: int, n_nodes: int = 64,
             T: int = 2500, burn: int = 1500) -> dict:
    cfg, nodes = build_seed(n_nodes, seed, lr, w_scale)
    out = measure_nodes(cfg, nodes, seed, T, burn)
    out.update(w_scale=w_scale, lr=lr)
    return out


def measure_nodes(cfg, nodes, seed: int, T: int = 2500, burn: int = 1500,
                  t_offset: int = 0) -> dict:
    """Run the Seed for burn+T steps (learning per cfg), record the trajectory
    after burn, then probe the largest Lyapunov exponent with weights frozen."""
    n_nodes = len(nodes)
    rng = np.random.default_rng(seed)
    ids = sorted(nodes)
    traj = np.zeros((T, n_nodes))
    sig_hist = []
    for t in range(T + burn):
        S01.minimal_step(nodes, seed_drive(t + t_offset, nodes, cfg), rng)
        if t >= burn:
            traj[t - burn] = [nodes[i].activation for i in ids]
            sig_hist.append(np.mean([nodes[i].branching_ratio for i in ids]))
    sigma = float(np.mean(sig_hist))
    mean_w = float(np.mean([w for n in nodes.values() for w in n.coupling_weights.values()]))
    # Benettin twin with a shared noise stream; weights frozen for the probe
    # so λ is about the activation dynamics at the reached operating point.
    for n in nodes.values():
        n._config = copy.copy(cfg)
        n._config.learning_rate = 0.0
    twin = copy.deepcopy(nodes)
    eps = 1e-6
    for i in ids:
        twin[i].activation += eps / np.sqrt(n_nodes) * rng.standard_normal()
    lam, steps = 0.0, 600
    for k in range(steps):
        t = T + burn + k + t_offset
        drives = seed_drive(t, nodes, cfg)
        r1 = np.random.default_rng(777_000 + t)
        r2 = np.random.default_rng(777_000 + t)
        S01.minimal_step(nodes, drives, r1)
        S01.minimal_step(twin, drives, r2)
        a = np.array([nodes[i].activation for i in ids])
        b = np.array([twin[i].activation for i in ids])
        d = np.linalg.norm(b - a)
        if d < 1e-300:
            lam += np.log(1e-16 / eps)
            d = eps
            b = a + eps / np.sqrt(n_nodes) * rng.standard_normal(n_nodes)
        else:
            lam += np.log(d / eps)
            b = a + (b - a) * (eps / d)
        for i, v in zip(ids, b):
            twin[i].activation = float(v)
    out = geometry(traj, rng)
    out.update(sigma=sigma, mean_w=mean_w, lyap=lam / steps,
               frac_neg_w=float(np.mean([w < 0 for n in nodes.values()
                                         for w in n.coupling_weights.values()])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save", default=str(HERE / "results_03_gain_sweep.json"))
    ap.add_argument("--skip-seed", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    results = {"esn": [], "seed_frozen": [], "seed_tuned": []}

    print("ESN arm (g = spectral radius)")
    print(f"  {'g':>5} {'D2':>6} {'±':>5} {'PR':>6} {'var':>7} {'sat':>6} {'λ':>7}")
    for g in [0.3, 0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 2.4]:
        rows = [esn_arm(g, s) for s in range(args.seeds)]
        results["esn"].extend(rows)
        m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
        print(f"  {g:5.2f} {m['d2']:6.2f} {m['d2_spread']:5.2f} {m['pr']:6.2f} {m['var']:7.4f} "
              f"{m['sat']:6.3f} {m['lyap']:+7.3f}")
    print(f"  ({time.time()-t0:.0f}s)")

    if not args.skip_seed:
        print("\nSeed arm, frozen weights (Rule 2a off), w_scale swept")
        print(f"  {'wscl':>5} {'σ':>5} {'<w>':>5} {'D2':>6} {'±':>5} {'PR':>6} {'var':>7} {'sat':>6} {'λ':>7}")
        for ws in [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.5]:
            rows = [seed_arm(ws, 0.0, s) for s in range(args.seeds)]
            results["seed_frozen"].extend(rows)
            m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
            print(f"  {ws:5.1f} {m['sigma']:5.2f} {m['mean_w']:5.2f} {m['d2']:6.2f} {m['d2_spread']:5.2f} "
                  f"{m['pr']:6.2f} {m['var']:7.4f} {m['sat']:6.3f} {m['lyap']:+7.3f}")
        print(f"  ({time.time()-t0:.0f}s)")

        print("\nSeed arm, self-tuned (Rule 2a on) from the same initial scales")
        print(f"  {'wscl':>5} {'σ':>5} {'<w>':>5} {'D2':>6} {'±':>5} {'PR':>6} {'var':>7} {'sat':>6} {'λ':>7}")
        for ws in [0.1, 0.3, 1.0, 2.5]:
            rows = [seed_arm(ws, 0.01, s) for s in range(args.seeds)]
            results["seed_tuned"].extend(rows)
            m = {k: np.nanmean([r[k] for r in rows]) for k in rows[0]}
            print(f"  {ws:5.1f} {m['sigma']:5.2f} {m['mean_w']:5.2f} {m['d2']:6.2f} {m['d2_spread']:5.2f} "
                  f"{m['pr']:6.2f} {m['var']:7.4f} {m['sat']:6.3f} {m['lyap']:+7.3f}")
        print(f"  ({time.time()-t0:.0f}s)")

    Path(args.save).write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
