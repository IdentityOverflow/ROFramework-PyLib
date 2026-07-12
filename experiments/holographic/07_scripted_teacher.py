"""
Scripted teacher — episodic memory as a demonstration amplifier (#8).

#7 ended with teleop as the planned structure injector, and a ~1-hour human
teleop session (2026-07-05, Wim-Teleop-64) produced no visible behavioral
change. The reel post-mortem explains why: 1,820 taught records (mean tag
+0.82) against 15,651 self-play records (mean −0.04) — teaching covered
under 1% of a ~200k-step session on a fresh brain. The machinery worked
(taught flags recorded, demonstrations are the reel's highest-valence
content); the dose was homeopathic and the log shows valence *declining*
across the hour while teaching sat idle in the reel.

This experiment replaces the human with the #5 Braitenberg policy as a
scripted teacher and controls the dose: interleaved 1.5k-step teaching
blocks cover ~25% of steps 4k–18k, followed by a 12k-step retention
window. Fresh WimBrain per run (matched brain seed), matched world seeds,
three arms:

    baseline   no teaching; reel observe-only          (drift control)
    teach      teaching;    reel observe-only (β=0)    (pure supervised nudge)
    teach+dv   teaching;    β=0.4, demo_weight=3       (full stack — does the
               reel keep re-evoking demonstrations after the teacher lets go?)

Measured per run, in windows (pre / gap / retention-early / retention-late):

  imitation   Pearson r between the teacher's turn command and the brain's
              deterministic proposal (raw readout, no exploration noise) on
              the same observation, free-play steps only. Turn is the
              teachable signal (food steering); also eat agreement, fwd MAE.
  behavior    valence mean and eat events (Δvalence > +0.35) per window;
              deaths over the run.
  mechanism   reel foresight r / event r, retrieval precision, and the
              demo-hit rate — fraction of recalls whose top match is a
              taught record — over the whole run and over retention only.
              teach vs teach+dv isolates the dv̂ pathway: identical
              demonstrations, identical recall; only reward shaping differs.

NOTHING IS SAVED — fresh brains and reels per run, nothing written back.

Requires: numpy, torch + experiments/embodied.
Runtime: ~10-20 min/run on GPU at wim-64; 6 runs by default.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_embodied = os.path.join(_here, "..", "embodied")
sys.path.insert(0, _embodied)

from env import World          # noqa: E402
from wim_brain import WimBrain  # noqa: E402

CONFIG = os.path.join(_embodied, "brains", "configs", "wim-64-teleop-holo.json")
RULESET = os.path.join(_embodied, "rulesets", "default.json")

N_STEPS = 30000
TEACH_START = 4000      # after holo_calib_steps=3600 — reel is live
TEACH_END = 18000
BLOCK = 1500            # alternating teach/free blocks
RET_SPLIT = 24000       # retention-early / retention-late boundary

ARMS = [
    ("baseline", dict(teach=False, beta=0.0)),
    ("teach",    dict(teach=True,  beta=0.0)),
    ("teach+dv", dict(teach=True,  beta=0.4)),
]


def _load_json(path):
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ── The teacher: #5's Braitenberg explorer ─────────────────────────────────────

def teacher_policy(obs, rng, state):
    """Braitenberg-ish: steer to strongest food ray, flee close danger, wander."""
    vis = obs[:242].reshape(121, 2)
    food = np.abs(vis[:, 0] - 0.50) < 0.01
    danger = np.abs(vis[:, 0] - 0.75) < 0.01
    fwd, turn = 0.6, 0.0
    if danger.any() and vis[danger, 1].max() > 0.55:
        j = int(np.argmax(np.where(danger, vis[:, 1], 0.0)))
        turn = -np.sign(j - 60) or 1.0
        fwd = 0.25
    elif food.any():
        j = int(np.argmax(np.where(food, vis[:, 1], 0.0)))
        turn = float(np.clip((j - 60) / 45.0, -1.0, 1.0))
        fwd = 0.8
    else:
        state["ou"] = 0.92 * state["ou"] + 0.35 * rng.normal()
        turn = float(np.clip(state["ou"], -1.0, 1.0))
    return fwd, turn, 1.0


def in_teach_block(step):
    if not (TEACH_START <= step < TEACH_END):
        return False
    return ((step - TEACH_START) // BLOCK) % 2 == 0


def window_of(step, teaching):
    if step < TEACH_START:
        return "pre"
    if step < TEACH_END:
        return "teach" if teaching else "gap"
    return "ret1" if step < RET_SPLIT else "ret2"


WINDOWS = ["pre", "gap", "ret1", "ret2"]     # imitation measured free-play only


class _Imit:
    """(teacher, proposal) pairs on the same observation, one window."""

    def __init__(self):
        self.t_turn, self.p_turn = [], []
        self.t_fwd, self.p_fwd = [], []
        self.eat_agree = []

    def add(self, t, p):
        self.t_fwd.append(t[0]);  self.p_fwd.append(p[0])
        self.t_turn.append(t[1]); self.p_turn.append(p[1])
        self.eat_agree.append(float(t[2] == p[2]))

    def report(self):
        a, b = np.array(self.t_turn), np.array(self.p_turn)
        if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(a, b)[0, 1])
        fwd_mae = (float(np.mean(np.abs(np.array(self.t_fwd) - np.array(self.p_fwd))))
                   if self.t_fwd else float("nan"))
        eat = float(np.mean(self.eat_agree)) if self.eat_agree else float("nan")
        return r, fwd_mae, eat, len(a)


def run_one(world_seed, teach, beta, device, n_steps):
    cfg = _load_json(CONFIG)
    cfg.update({
        "brain_path": "", "log_path": "",       # never save anything
        "holo_enabled": True,
        "holo_beta": beta,
        "device": device,
    })
    brain = WimBrain(config=cfg, device=device,
                     action_feedback=cfg.get("action_feedback", False),
                     seed=cfg.get("seed", 42))
    world = World(seed=world_seed, cfg=_load_json(RULESET))
    world.player_active = False
    world.add_agent()

    trng = np.random.default_rng(world_seed + 1000)
    tstate = {"ou": 0.0}
    imit = {w: _Imit() for w in WINDOWS}
    vals = {w: [] for w in WINDOWS + ["teach"]}
    eats = {w: 0 for w in WINDOWS + ["teach"]}
    deaths = 0
    prev_life, prev_reward = 1.0, 0.0
    ret_recalls = ret_demo_hits = 0     # snapshot at retention start
    t0 = time.time()

    for step in range(n_steps):
        obs = world.get_ai_observation()
        life = float(obs[260])
        died = life > prev_life + 0.5           # respawn = life jumped back up
        prev_life = life
        reward = float(world.ai.meters.valence)

        fwd, turn, eat = brain.forward(obs)
        with torch.no_grad():
            raw = brain.W_out @ brain._last_h + brain.b_out
        prop = (float(torch.tanh(raw[0])), float(torch.tanh(raw[1])),
                1.0 if float(raw[2]) > brain._eat_threshold else 0.0)
        t_act = teacher_policy(obs, trng, tstate)

        brain.learn(reward)
        if died:
            brain.reset_state()
            deaths += 1

        teaching = teach and in_teach_block(step)
        if teaching:
            action, tf = t_act, True
        else:
            action, tf = (fwd, turn, eat), False
        world.step(ai_action=action)
        brain.set_executed_action(action, teacher_forced=tf)

        w = window_of(step, teaching)
        vals[w].append(reward)
        if reward - prev_reward > 0.35:
            eats[w] += 1
        prev_reward = reward
        if not teaching and w in imit:
            imit[w].add(t_act, prop)

        if step == TEACH_END and brain._holo is not None:
            ret_recalls = brain._holo.n_recalls
            ret_demo_hits = brain._holo._demo_hits

        if (step + 1) % 5000 == 0:
            h = brain._holo.stats() if brain._holo is not None else ""
            print(f"    [{world_seed}/{'T' if teach else '-'}/β={beta}] "
                  f"step {step + 1}: val {np.mean(vals[w][-2000:]):+.3f}  "
                  f"deaths {deaths}  ({(step + 1) / (time.time() - t0):.0f}/s)  {h}",
                  flush=True)

    holo = brain._holo
    r, n, r_ev, n_ev = holo.foresight()
    n_taught = (sum(sum(s.taught) for s in holo.memory.slides)
                if holo.ready else 0)
    ret_n = holo.n_recalls - ret_recalls
    out = {
        "world_seed": world_seed, "teach": teach, "beta": beta,
        "deaths": deaths,
        "records": holo.memory.n_records if holo.ready else 0,
        "taught": int(n_taught),
        "prec": holo._prec_hits / max(holo.n_recalls, 1),
        "demo_hit": holo._demo_hits / max(holo.n_recalls, 1),
        "demo_hit_ret": (holo._demo_hits - ret_demo_hits) / max(ret_n, 1),
        "foresight_r": r, "foresight_n": n,
        "event_r": r_ev, "event_n": n_ev,
    }
    for w in WINDOWS:
        out[f"imit_{w}"] = imit[w].report()
    for w in WINDOWS + ["teach"]:
        out[f"val_{w}"] = float(np.mean(vals[w])) if vals[w] else float("nan")
        out[f"eats_{w}"] = eats[w]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--seeds", type=str, default="7,21")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arms", type=str, default="baseline,teach,teach+dv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [(name, p) for name, p in ARMS if name in args.arms.split(",")]

    print("=" * 78)
    print("Scripted teacher — episodic memory as a demonstration amplifier (#8)")
    print("=" * 78)
    print(f"  {args.steps} steps/run on {args.device}; teach blocks "
          f"{BLOCK}-step alternating in [{TEACH_START}, {TEACH_END}); "
          f"retention to {args.steps}")
    print(f"  world seeds {seeds}, arms {[a for a, _ in arms]}\n")

    results = []
    for ws in seeds:
        for name, p in arms:
            print(f"  run: world seed {ws}, arm {name}", flush=True)
            res = run_one(ws, p["teach"], p["beta"], args.device, args.steps)
            res["arm"] = name
            results.append(res)

    print("\n  imitation: teacher-turn vs brain-proposal Pearson r "
          "(free play, per window)")
    print(f"  {'world':>5} {'arm':>9} {'pre':>6} {'gap':>6} {'ret1':>6} "
          f"{'ret2':>6}   {'eat-agree ret2':>14} {'n(ret2)':>8}")
    for x in results:
        cells = [f"{x[f'imit_{w}'][0]:+.2f}" for w in WINDOWS]
        print(f"  {x['world_seed']:>5} {x['arm']:>9} "
              + " ".join(f"{c:>6}" for c in cells)
              + f"   {x['imit_ret2'][2]:>14.0%} {x['imit_ret2'][3]:>8}")

    print("\n  behavior + reel")
    print(f"  {'world':>5} {'arm':>9} {'val pre':>8} {'val ret2':>9} "
          f"{'eats ret':>8} {'deaths':>6} {'taught':>6} {'demo-hit(ret)':>13} "
          f"{'fore r':>7} {'event r':>8}")
    for x in results:
        print(f"  {x['world_seed']:>5} {x['arm']:>9} {x['val_pre']:>+8.3f} "
              f"{x['val_ret2']:>+9.3f} {x['eats_ret1'] + x['eats_ret2']:>8} "
              f"{x['deaths']:>6} {x['taught']:>6} {x['demo_hit_ret']:>13.0%} "
              f"{x['foresight_r']:>+7.2f} {x['event_r']:>+8.2f} "
              f"(n={x['event_n']})")

    print("\n  paired deltas vs baseline (same world), retention-late:")
    for ws in seeds:
        base = next((x for x in results
                     if x["world_seed"] == ws and x["arm"] == "baseline"), None)
        if base is None:
            continue
        for x in results:
            if x["world_seed"] == ws and x["arm"] != "baseline":
                print(f"    world {ws} {x['arm']:>9}: "
                      f"Δimit(turn r) {x['imit_ret2'][0] - base['imit_ret2'][0]:+.2f}"
                      f"   Δval {x['val_ret2'] - base['val_ret2']:+.3f}"
                      f"   Δeats {(x['eats_ret1'] + x['eats_ret2']) - (base['eats_ret1'] + base['eats_ret2']):+d}"
                      f"   Δdeaths {x['deaths'] - base['deaths']:+d}")
    print("\nDone.")


if __name__ == "__main__":
    main()
