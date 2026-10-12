"""#F4 — The creditable twist with the outer loop closed: embodied Seed-Hemi.

#F3c certified the twist on a Seed listening to a tape. Here the same
channel (experiments/embodied/twist_mount.py, opt-in `twist_*` keys in
seed_brain.py) is mounted on the hemispheric Seed brain acting in the
embodied world: its motor nodes move the body, the body changes what the
sensors see, and the self-model's output is injected back into 8 general
nodes. Both loops are closed.

Conditions × seeds, headless, N steps each:
  none   plain SeedBrain (no mount; the behavioural baseline)
  live   creditable channel, consumed
  stale  frozen constants through the same path
  blind  self-model ignores the encoding (w = 0)
  off    channel present, gain 0

Per run: real eats (world consume calls), deaths, mean valence, motor
magnitudes, mean σ, nodes alive; certification every `--certify-every`
steps on the mounted arms (twist_assessment, closure under the embodied
log, is_conscious = twisted ∧ closed, creditable weight intervention);
geometry on the final window of node activations (D2 on PCA-20,
participation ratio, saturation) with the frozen node list.

Pre-registered
--------------
Certification: live → twisted, with closure now measured under a world the
agent itself perturbs (corr_int vs corr_ext reported; a fail here is a
finding about the embodied closure test, not about the wiring).
Behaviour and geometry: live ≡ blind within seed noise; off and none may
differ from live (closure does work, #F3c). If live differs from blind the
twist does work once the outer loop is closed — the one way the #F3c null
could still be overturned.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EMB))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from fractal_tools import correlation_dimension, participation_ratio, pca_project  # noqa: E402

CONDITIONS = ("none", "live", "stale", "blind", "off")


def build(condition: str, seed: int):
    os.chdir(EMB)                      # seed_brain/env resolve rulesets relative to here
    import seed_brain                  # noqa: PLC0415
    from env import World              # noqa: PLC0415
    from brain import _load_world_config, _LIFE_IDX  # noqa: PLC0415
    raw = json.loads((EMB / "brains" / "configs" / "seed-hemi-64.json").read_text())
    cfg = {**seed_brain.SEED_DEFAULT_CONFIG, **{k: v for k, v in raw.items() if not k.startswith("_")}}
    cfg.update(brain_path="", log_path="", seed=seed)
    if condition != "none":
        cfg.update(twist_enabled=True, twist_condition=condition)
    brain = seed_brain.SeedBrain(config=cfg, device="cpu", seed=seed)
    world_cfg = _load_world_config(cfg["world_config"]) if cfg.get("world_config") else {}
    world = World(seed=seed, no_reset_on_death=False, cfg=world_cfg)
    world.player_active = False
    world.add_agent()
    return brain, world, _LIFE_IDX


def run(args):
    condition, seed, n_steps, certify_every, geom_window = args
    brain, world, LIFE_IDX = build(condition, seed)
    eats = {"n": 0}
    orig_consume = world._consume_food

    def consume(food):
        eats["n"] += 1
        return orig_consume(food)
    world._consume_food = consume

    if brain._twist is not None:
        ids = brain._twist.ids
    else:
        ids = sorted(brain._seed_net.nodes.keys(), key=lambda k: int(k[1:]))

    def acts():
        nodes = brain._seed_net.nodes
        return np.array([nodes[k].activation if k in nodes else 0.0 for k in ids])

    traj = np.zeros((geom_window, len(ids)))
    prev_life, fwd = 1.0, 0.0
    turn = eat = 0.0
    val, fw, tn, sig = [], [], [], []
    certs = []
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
        if step >= n_steps - geom_window:
            traj[step - (n_steps - geom_window)] = acts()
        if brain._twist is not None and certify_every and (step + 1) % certify_every == 0:
            c = brain._twist.certify(seed=seed)
            c["step"] = step + 1
            certs.append(c)
    g = pca_project(traj, 20)
    d2, spread = correlation_dimension(g, rng=np.random.default_rng(seed))
    out = dict(condition=condition, seed=seed, steps=n_steps, eats=eats["n"],
               deaths=int(world.death_count), valence=float(np.mean(val)),
               fwd=float(np.mean(fw)), turn=float(np.mean(tn)), sigma=float(np.mean(sig)),
               nodes=int(brain.node_count), n_frozen=len(ids),
               n_live=int(sum(1 for k in ids if k in brain._seed_net.nodes)),
               d2=float(d2), d2_spread=float(spread), pr=participation_ratio(traj),
               var=float(traj.var(axis=0).mean()), sat=float((np.abs(traj) > 0.95).mean()),
               certs=certs, secs=time.time() - t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--certify-every", type=int, default=5000)
    ap.add_argument("--geom-window", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--save", default=str(HERE / "results_07_embodied_twist.json"))
    args = ap.parse_args()
    jobs = [(c, s, args.steps, args.certify_every, args.geom_window)
            for c in args.conditions for s in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run, jobs))

    print(f"#F4 embodied twist — {args.steps} steps, {args.seeds} seeds  ({time.time()-t0:.0f}s)")
    print("\nBehaviour and geometry (mean over seeds):")
    print(f"  {'cond':>5} {'eats':>6} {'deaths':>6} {'valence':>8} {'|fwd|':>6} {'|turn|':>6} {'σ':>5} {'nodes':>5} {'live':>5} | {'D2':>5} {'±':>4} {'PR':>5} {'var':>6} {'sat':>5}")
    for c in args.conditions:
        rs = [r for r in rows if r["condition"] == c]
        m = {k: np.nanmean([r[k] for r in rs]) for k in ("eats", "deaths", "valence", "fwd", "turn", "sigma", "nodes", "n_live", "d2", "d2_spread", "pr", "var", "sat")}
        print(f"  {c:>5} {m['eats']:6.1f} {m['deaths']:6.1f} {m['valence']:+8.3f} {m['fwd']:6.3f} {m['turn']:6.3f} {m['sigma']:5.2f} {m['nodes']:5.0f} {m['n_live']:5.0f} | "
              f"{m['d2']:5.2f} {m['d2_spread']:4.2f} {m['pr']:5.2f} {m['var']:6.3f} {m['sat']:5.3f}")
    print("\n  per seed: cond  seed  eats deaths  D2")
    for r in rows:
        print(f"    {r['condition']:>5} {r['seed']:>4} {r['eats']:>5} {r['deaths']:>5}  {r['d2']:.2f}")

    print("\nCertification (last checkpoint of each run):")
    print(f"  {'cond':>5} {'seed':>4} | {'struct':>6} {'consum':>6} {'cond.':>5} {'triv':>4} | {'twist':>5} {'closed':>6} {'consc':>5} | {'c_int':>6} {'c_ext':>6} | {'cred':>7} {'foil':>7} | live")
    for r in rows:
        if not r["certs"]:
            continue
        c = r["certs"][-1]
        print(f"  {r['condition']:>5} {r['seed']:>4} | {c['structural']:>6} {c['consumes']:>6} {c['conditional']:>5} {c['trivial']:>4} | "
              f"{c['twisted']:>5} {c['closed']:>6} {c['conscious']:>5} | {c['corr_int']:6.3f} {c['corr_ext']:6.3f} | {c['cred']:7.4f} {c['foil']:7.4f} | {c['n_live']} r{c['resp_live']} m{c['meta_live']}")
    print("\n  closure over checkpoints (live arm): " + "; ".join(
        f"s{r['seed']}: " + ",".join(f"{c['closed']}@{c['step']}" for c in r["certs"])
        for r in rows if r["condition"] == "live"))
    Path(args.save).write_text(json.dumps(rows, indent=1, default=float))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
