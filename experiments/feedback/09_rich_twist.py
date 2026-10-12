"""#F6 — Richness knobs on the embodied twist: bandwidth, adaptivity, consumption site.

#F5 found the certified channel carries < 1 bit/readout about the network's
weights and feeds it below the noise floor of the nodes it reaches —
twisted, closed, inaudible. This turns the three richness knobs the
findings named and asks the two questions again:

  richness  bits/readout at matched state (vector-aware: Σ_j ½log₂(1+SNR_j²))
  behaviour eats, deaths, valence (live vs blind vs off), geometry (D2)

Configs (twist_mount.py options):
  vec          n_out = 8: one channel per meta node (8× bandwidth), fixed W
  vec+adapt    + self-model adapts: each m_j is a one-step forward model of
               the node it feeds, trained online from the self-encoding
               (blind = same forward model with W frozen at 0)
  vec+adapt+gain   consumption multiplicative on the meta nodes' drive
  vec+adapt+lr     consumption gates Rule 2a's reward modulator (plasticity)

Conditions: live / blind / off. 10 seeds × 15 000 headless steps.
Pre-registered: richness rises with each knob (bits at δ=1.1 above #F5's
0.2–0.7); behaviour stays live ≡ blind unless a knob makes the channel's
content audible — the only way the #F4 null can move.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parents[0] / "embodied"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from fractal_tools import correlation_dimension, participation_ratio, pca_project  # noqa: E402

CONFIGS = {
    "scalar":         dict(twist_n_out=1, twist_adaptive=False, twist_site="drive"),
    "vec":            dict(twist_n_out=8, twist_adaptive=False, twist_site="drive"),
    "vec+adapt":      dict(twist_n_out=8, twist_adaptive=True,  twist_site="drive"),
    "vec+adapt+gain": dict(twist_n_out=8, twist_adaptive=True,  twist_site="gain"),
    "vec+adapt+lr":   dict(twist_n_out=8, twist_adaptive=True,  twist_site="lr"),
}
DELTAS = (1.10, 1.50)


def build(config: str, condition: str, seed: int):
    os.chdir(EMB)
    import seed_brain                  # noqa: PLC0415
    from env import World              # noqa: PLC0415
    from brain import _load_world_config, _LIFE_IDX  # noqa: PLC0415
    raw = json.loads((EMB / "brains" / "configs" / "seed-hemi-64.json").read_text())
    cfg = {**seed_brain.SEED_DEFAULT_CONFIG, **{k: v for k, v in raw.items() if not k.startswith("_")}}
    cfg.update(brain_path="", log_path="", seed=seed, twist_enabled=True, twist_condition=condition)
    cfg.update(CONFIGS[config])
    brain = seed_brain.SeedBrain(config=cfg, device="cpu", seed=seed)
    world_cfg = _load_world_config(cfg["world_config"]) if cfg.get("world_config") else {}
    world = World(seed=seed, no_reset_on_death=False, cfg=world_cfg)
    world.player_active = False
    world.add_agent()
    return brain, world, _LIFE_IDX


def channel_bits(mount, n_jitter: int = 12) -> dict:
    """Vector-aware #F5 magnitude at the current state."""
    from twist_mount import ProbeMapping   # noqa: PLC0415
    obs, enc, sm = mount.observer, mount.encoder, mount.self_model
    res = {d: obs.get_resolution(d) for d in obs.d_meta}

    def m_of(tgt, seed):
        tgt.step_seed = seed
        tgt._k = 0
        e = enc.encode(tgt, res)
        s = obs.internal_state
        for dof in enc.all_dofs:
            v = e.get_value(dof)
            if v is not None:
                s = s.set_value(dof, float(v))
        o = sm(s)
        return np.array([float(o.get_value(d) or 0.0) for d in sm.out])

    base = np.array([m_of(mount.mapping, 5000 + k) for k in range(n_jitter)])
    jitter = base.std(axis=0) + 1e-12
    m0 = base.mean(axis=0)
    out = {}
    for d in DELTAS:
        alt = ProbeMapping(mount.net, mount.ids, mount.u, mount.a, weight_scale=d)
        ms = np.array([m_of(alt, 5000 + k) for k in range(n_jitter)])
        snr = np.abs(ms.mean(axis=0) - m0) / jitter
        out[f"bits@{d}"] = float(np.sum(0.5 * np.log2(1 + snr ** 2)))
        out[f"snr_max@{d}"] = float(snr.max())
    out["jitter_mean"] = float(jitter.mean())
    return out


def run(args):
    config, condition, seed, n_steps, geom_window = args
    brain, world, LIFE_IDX = build(config, condition, seed)
    eats = {"n": 0}
    orig = world._consume_food

    def consume(food):
        eats["n"] += 1
        return orig(food)
    world._consume_food = consume
    mount = brain._twist
    ids = mount.ids
    traj = np.zeros((geom_window, len(ids)))
    prev_life = 1.0
    val, fw, tn, sig, perr = [], [], [], [], []
    t0 = time.time()
    for step in range(n_steps):
        obs = world.get_ai_observation()
        reward = float(world.ai.meters.valence)
        life = float(obs[LIFE_IDX])
        done = bool(life > 0.9 and prev_life < 0.1)
        prev_life = life
        fwd, turn, eat = brain.forward(obs)
        brain.learn(reward)
        if done:
            brain.reset_state()
        world.step(ai_action=(fwd, turn, eat))
        brain.set_executed_action((fwd, turn, eat))
        val.append(reward); fw.append(abs(fwd)); tn.append(abs(turn))
        if step % 50 == 0:
            sig.append(brain.mean_sigma)
            if mount.adaptive:
                perr.append(mount.pred_err)
        if step >= n_steps - geom_window:
            traj[step - (n_steps - geom_window)] = mount.activations()
    cert = mount.certify(seed=seed)
    bits = channel_bits(mount)
    g = pca_project(traj, 20)
    d2, spread = correlation_dimension(g, rng=np.random.default_rng(seed))
    return dict(config=config, condition=condition, seed=seed, eats=eats["n"],
                deaths=int(world.death_count), valence=float(np.mean(val)),
                fwd=float(np.mean(fw)), turn=float(np.mean(tn)), sigma=float(np.mean(sig)),
                pred_err=float(np.nanmean(perr)) if perr else float("nan"),
                nodes=int(brain.node_count), d2=float(d2), pr=participation_ratio(traj),
                sat=float((np.abs(traj) > 0.95).mean()), cert=cert, **bits, secs=time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--geom-window", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--configs", nargs="+", default=["vec", "vec+adapt", "vec+adapt+gain", "vec+adapt+lr"])
    ap.add_argument("--conditions", nargs="+", default=["live", "blind", "off"])
    ap.add_argument("--save", default=str(HERE / "results_09_rich_twist.json"))
    args = ap.parse_args()
    jobs = [(c, cond, s, args.steps, args.geom_window)
            for c in args.configs for cond in args.conditions for s in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run, jobs))
    print(f"#F6 richness knobs — {args.steps} steps, {args.seeds} seeds, {len(rows)} runs ({time.time()-t0:.0f}s)")

    def agg(rs, k):
        v = np.array([r[k] for r in rs], dtype=float)
        return np.nanmean(v), np.nanstd(v)

    print(f"\n  {'config':>15} {'cond':>5} | {'bits@1.1':>8} {'bits@1.5':>8} | {'tw':>5} {'cl':>5} {'co':>5} | "
          f"{'eats':>9} {'deaths':>9} {'valence':>8} {'pred_err':>8} | {'D2':>9} {'sat':>6}")
    for c in args.configs:
        for cond in args.conditions:
            rs = [r for r in rows if r["config"] == c and r["condition"] == cond]
            if not rs:
                continue
            tw = sum(r["cert"]["twisted"] for r in rs); cl = sum(r["cert"]["closed"] for r in rs); co = sum(r["cert"]["conscious"] for r in rs)
            b1, _ = agg(rs, "bits@1.1"); b5, _ = agg(rs, "bits@1.5")
            e, es = agg(rs, "eats"); d, ds = agg(rs, "deaths"); v, _ = agg(rs, "valence"); pe, _ = agg(rs, "pred_err")
            d2, d2s = agg(rs, "d2"); sat, _ = agg(rs, "sat")
            print(f"  {c:>15} {cond:>5} | {b1:8.2f} {b5:8.2f} | {tw:2d}/{len(rs):<2d} {cl:2d}/{len(rs):<2d} {co:2d}/{len(rs):<2d} | "
                  f"{e:4.1f}±{es:3.1f} {d:4.1f}±{ds:3.1f} {v:+8.3f} {pe:8.3f} | {d2:4.2f}±{d2s:3.2f} {sat:6.3f}")
    print("\n  seed-paired live − blind:")
    for c in args.configs:
        live = {r["seed"]: r for r in rows if r["config"] == c and r["condition"] == "live"}
        blind = [r for r in rows if r["config"] == c and r["condition"] == "blind"]
        if not live or not blind:
            continue
        de = [live[r["seed"]]["eats"] - r["eats"] for r in blind]
        dd = [live[r["seed"]]["deaths"] - r["deaths"] for r in blind]
        dv = [live[r["seed"]]["valence"] - r["valence"] for r in blind]
        dg = [live[r["seed"]]["d2"] - r["d2"] for r in blind]
        n = np.sqrt(len(de))
        print(f"    {c:>15}: Δeats {np.mean(de):+.2f}±{np.std(de)/n:.2f}  Δdeaths {np.mean(dd):+.2f}±{np.std(dd)/n:.2f}  "
              f"Δvalence {np.mean(dv):+.3f}±{np.std(dv)/n:.3f}  ΔD2 {np.mean(dg):+.2f}±{np.std(dg)/n:.2f}")
    Path(args.save).write_text(json.dumps(rows, indent=1, default=float))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
