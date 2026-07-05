"""
Closing the loop — β>0 A/B on the Seed-Hemi brain.

#6 ended with the bootstrapping argument: memory-shaped reward should only
work for an agent whose behavior already produces recurring situations, and
if it works it should produce MORE structure, making the memory more
prophetic in turn. This experiment runs that loop on the hemispheric Seed
brain (reward-modulated Hebbian plasticity — no readout matrix), starting
from the trained Seed-Hemi-64_af checkpoint.

Protocol: paired headless runs, matched (brain checkpoint, world seed),
differing only in β:

    β = 0.0   reel mounted, observe-only (control — mount overhead and
              recording identical, no influence on learning)
    β = 0.6   rpe = (reward + β·dv̂) − valence_pred: recalled Δvalence acts
              as an episodic value bootstrap inside the RPE that gates
              Hebbian plasticity

The reel needs holo_calib_steps of experience before dv̂ exists, so both
arms are identical for the first ~5k steps; metrics are reported for the
post-calibration window.

Measured per run: mean valence, food eaten (valence spikes > +0.35/step),
deaths, and the mount's own honesty meters (foresight r, retrieval
precision). NOTHING IS SAVED — checkpoints are loaded fresh per run and
never written back.

Requires: numpy + experiments/embodied.  Runtime: ~20-30 min on CPU.
"""

import json
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_embodied = os.path.join(_here, "..", "embodied")
sys.path.insert(0, _embodied)

from env import World                     # noqa: E402
from seed_brain import SeedBrain          # noqa: E402

CHECKPOINT = os.path.join(_embodied, "brains", "Seed-Hemi-64_af.npz")
CONFIG = os.path.join(_embodied, "brains", "configs", "seed-hemi-64_af.json")
RULESET = os.path.join(_embodied, "rulesets", "default.json")

N_STEPS = 15000
CALIB = 5000
WORLD_SEEDS = [7, 21]
BETAS = [0.0, 0.6]


def _load_json(path):
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def run_one(world_seed: int, beta: float) -> dict:
    cfg = _load_json(CONFIG)
    cfg.update({
        "brain_path": "", "log_path": "",          # never save anything
        "holo_enabled": True,
        "holo_beta": beta,
        "holo_calib_steps": CALIB,
    })
    brain = SeedBrain(config=cfg, device="cpu",
                      action_feedback=cfg.get("action_feedback", False),
                      seed=cfg.get("seed", 42))
    brain.load(CHECKPOINT)

    world = World(seed=world_seed, cfg=_load_json(RULESET))
    world.player_active = False
    world.add_agent()

    valences, eats, deaths = [], 0, 0
    prev_life, prev_reward = 1.0, 0.0
    fwd = turn = eat = 0.0
    t0 = time.time()

    for step in range(N_STEPS):
        obs = world.get_ai_observation()
        reward = float(world.ai.meters.valence)
        life = float(obs[260])
        done = bool(life > 0.9 and prev_life < 0.1)
        prev_life = life

        fwd, turn, eat = brain.forward(obs)
        brain.learn(reward)
        if done:
            brain.reset_state()
            deaths += 1
        world.step(ai_action=(fwd, turn, eat))
        brain.set_executed_action((fwd, turn, eat))

        valences.append(reward)
        if reward - prev_reward > 0.35:
            eats += 1
        prev_reward = reward

        if (step + 1) % 5000 == 0:
            print(f"    [seed {world_seed} β={beta}] step {step + 1}: "
                  f"val {np.mean(valences[-5000:]):+.3f}  eats {eats}  "
                  f"deaths {deaths}  ({(step + 1) / (time.time() - t0):.0f}/s)",
                  flush=True)

    v = np.array(valences)
    r, n, r_ev, n_ev = brain._holo.foresight()
    return {
        "world_seed": world_seed, "beta": beta,
        "val_all": float(v.mean()),
        "val_post": float(v[CALIB:].mean()),
        "eats": eats, "eats_post": None,   # eats counted over full run
        "deaths": deaths,
        "records": brain._holo.memory.n_records if brain._holo.ready else 0,
        "prec": brain._holo._prec_hits / max(brain._holo.n_recalls, 1),
        "foresight_r": r, "foresight_n": n,
        "event_r": r_ev, "event_n": n_ev,
    }


def main() -> None:
    print("=" * 70)
    print("Closing the loop — β>0 A/B on Seed-Hemi-64_af")
    print("=" * 70)
    print(f"  {N_STEPS} steps/run, reel live after {CALIB}, "
          f"world seeds {WORLD_SEEDS}, β ∈ {BETAS}\n")

    results = []
    for ws in WORLD_SEEDS:
        for beta in BETAS:
            print(f"  run: world seed {ws}, β={beta}", flush=True)
            results.append(run_one(ws, beta))

    print("\n  " + "-" * 66)
    hdr = (f"  {'world':>5} {'β':>4} {'valence':>8} {'post-cal':>8} "
           f"{'eats':>5} {'deaths':>6} {'prec':>5} {'fore r':>7} {'event r':>8}")
    print(hdr)
    for x in results:
        print(f"  {x['world_seed']:>5} {x['beta']:>4} {x['val_all']:>+8.3f} "
              f"{x['val_post']:>+8.3f} {x['eats']:>5} {x['deaths']:>6} "
              f"{x['prec']:>5.0%} {x['foresight_r']:>+7.2f} "
              f"{x['event_r']:>+8.2f} (n={x['event_n']})")

    print("\n  Paired deltas (β=0.6 minus β=0.0, same world):")
    for ws in WORLD_SEEDS:
        a = next(x for x in results
                 if x["world_seed"] == ws and x["beta"] == 0.0)
        b = next(x for x in results
                 if x["world_seed"] == ws and x["beta"] > 0.0)
        print(f"    world {ws}: Δvalence(post) {b['val_post'] - a['val_post']:+.3f}"
              f"   Δeats {b['eats'] - a['eats']:+d}"
              f"   Δdeaths {b['deaths'] - a['deaths']:+d}")
    print("\nDone.")


if __name__ == "__main__":
    main()
