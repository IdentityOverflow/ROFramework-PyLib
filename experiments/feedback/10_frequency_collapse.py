"""#F7 — Where did the frequency layers go? Entrainment under embodied drive.

The Seed's "layers" are not wired: one flat population with log-uniform natural
frequencies (cycles per world step, dt = 1), ring-sorted by frequency, and
bandwidth-limited entrainment that is supposed to carve bands matching the
environment's timescales (seed/04: a two-timescale drive gives two bands).
This measures what the embodied Seed-Hemi actually does with its two decades
of frequencies over a 15k-step run.

Arms
  plain   seeds 0–2, SeedBrain as in #F4 "none"
  live    seeds 0–1, twist mount live, 8 channels (as in #F6 "vec")
  zero    bare SeedNetwork, zero drive, entrainment on / off (controls)

Per run: spread of log10 f every 1000 steps (n alive, mean, sd, min, max),
released / recruited node tallies, end-of-run lag-1 autocorrelation of the
node activations, fraction of negative weights, mean σ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EMB = HERE.parents[0] / "embodied"
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(HERE.parents[1] / "src"))


def _cfg():
    import seed_brain  # noqa: PLC0415
    raw = json.loads((EMB / "brains" / "configs" / "seed-hemi-64.json").read_text())
    return {**seed_brain.SEED_DEFAULT_CONFIG, **{k: v for k, v in raw.items() if not k.startswith("_")}}


def _spread(net, t):
    v = np.log10([n.frequency for n in net.nodes.values()])
    return dict(t=t, n=len(v), mean=float(v.mean()), sd=float(v.std()), lo=float(v.min()), hi=float(v.max()))


def run(job):
    arm, seed, n_steps = job
    os.chdir(EMB)
    t0 = time.time()
    cfg = _cfg()
    if arm.startswith("zero"):
        from ro_framework.seed import SeedConfig, SeedNetwork  # noqa: PLC0415
        keys = SeedConfig.__dataclass_fields__.keys()
        sc = {k: (tuple(cfg[k]) if isinstance(cfg[k], list) else cfg[k]) for k in keys if k in cfg}
        if arm == "zero-eta0":
            sc["freq_learning_rate"] = 0.0
        net = SeedNetwork(config=SeedConfig(**sc), sensor=lambda x, f: dict.fromkeys(f, 0.0),
                          actuator=lambda a: None, seed=seed)
        f0 = {k: n.frequency for k, n in net.nodes.items()}
        course = []
        for t in range(n_steps):
            net.step(np.zeros(1))
            if (t + 1) % 1000 == 0:
                course.append(_spread(net, t + 1))
        brain = None
    else:
        import seed_brain  # noqa: PLC0415
        from env import World  # noqa: PLC0415
        from brain import _load_world_config, _LIFE_IDX  # noqa: PLC0415
        cfg.update(brain_path="", log_path="", seed=seed)
        if arm == "live":
            cfg.update(twist_enabled=True, twist_condition="live", twist_n_out=8)
        brain = seed_brain.SeedBrain(config=cfg, device="cpu", seed=seed)
        world = World(seed=seed, no_reset_on_death=False,
                      cfg=_load_world_config(cfg["world_config"]) if cfg.get("world_config") else {})
        world.player_active = False
        world.add_agent()
        net = brain._seed_net
        f0 = {k: n.frequency for k, n in net.nodes.items()}
        prev, course, tail = 1.0, [], []
        for t in range(n_steps):
            obs = world.get_ai_observation()
            r = float(world.ai.meters.valence)
            life = float(obs[_LIFE_IDX])
            done = life > 0.9 and prev < 0.1
            prev = life
            a = brain.forward(obs)
            brain.learn(r)
            if done:
                brain.reset_state()
            world.step(ai_action=a)
            brain.set_executed_action(a)
            if (t + 1) % 1000 == 0:
                course.append(_spread(net, t + 1))
            if t >= n_steps - 1000:
                tail.append({k: n.activation for k, n in net.nodes.items()})
    gone = [k for k in f0 if k not in net.nodes]
    kept = [k for k in f0 if k in net.nodes]
    new = [k for k in net.nodes if k not in f0]
    out = dict(arm=arm, seed=seed, course=course, n0=len(f0), released=len(gone), recruited=len(new),
               released_log10f=float(np.mean([np.log10(f0[k]) for k in gone])) if gone else None,
               kept_log10f=float(np.mean([np.log10(f0[k]) for k in kept])) if kept else None,
               frac_neg_w=float(np.mean([w < 0 for n in net.nodes.values() for w in n.coupling_weights.values()])),
               secs=time.time() - t0)
    if brain is not None:
        ids = [k for k in net.nodes if all(k in row for row in tail)]
        X = np.array([[row[k] for k in ids] for row in tail])
        ac = [float(np.corrcoef(X[:-1, j], X[1:, j])[0, 1]) for j in range(len(ids)) if X[:, j].std() > 1e-9]
        out.update(ac1_mean=float(np.mean(ac)), ac1_min=float(np.min(ac)), sigma=float(brain.mean_sigma))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--plain-seeds", type=int, default=3)
    ap.add_argument("--live-seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--save", default=str(HERE / "results_10_frequency_collapse.json"))
    args = ap.parse_args()
    jobs = [("plain", s, args.steps) for s in range(args.plain_seeds)] + \
           [("live", s, args.steps) for s in range(args.live_seeds)] + \
           [("zero", 0, args.steps), ("zero-eta0", 0, args.steps)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run, jobs))
    print(f"#F7 frequency collapse — {args.steps} steps")
    for r in rows:
        c0, c1 = r["course"][0], r["course"][-1]
        extra = f"  ac1 {r['ac1_mean']:+.2f} (min {r['ac1_min']:+.2f})  σ {r['sigma']:.2f}" if "ac1_mean" in r else ""
        print(f"\n  {r['arm']:>9} seed {r['seed']}: nodes {r['n0']} → {c1['n']} (released {r['released']}, recruited {r['recruited']}); "
              f"neg w {r['frac_neg_w']:.2f};{extra}")
        print(f"      t      n   mean   sd     [lo, hi]   (log10 f)")
        for c in r["course"]:
            if c["t"] % 3000 == 0 or c["t"] == 1000:
                print(f"    {c['t']:>5} {c['n']:>4}  {c['mean']:+.2f}  {c['sd']:.3f}  [{c['lo']:+.2f},{c['hi']:+.2f}]")
        if r["released"]:
            print(f"    released nodes' initial log10 f {r['released_log10f']:+.2f} vs kept {r['kept_log10f']:+.2f}")
    Path(args.save).write_text(json.dumps(rows, indent=1, default=float))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
