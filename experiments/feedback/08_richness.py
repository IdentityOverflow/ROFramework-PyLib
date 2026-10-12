"""#F5 — Richness of the conscious-by-criterion Seed.

is_conscious() is the binary in kind. The framework reports richness
separately, in two forms:

  (1) richness()   the graded v1 vector (ConsciousnessMetrics: recursive
                   depth, self-accuracy, architectural similarity,
                   calibration, metacognition, limitation awareness, and
                   the weighted consciousness_score).
  (2) magnitude    v2.2–v2.4: kind = sign of I(d_meta ; D_slow \\ W_meta |
                   S_rest), richness = its magnitude. Operationalized here
                   as a channel measurement at matched fast state: Δm under
                   a slow-DoF intervention (all couplings × δ) against the
                   jitter of m from the probe noise alone (re-encode the
                   same state with fresh probe noise). SNR(δ) and the
                   Gaussian-channel bits 0.5·log2(1 + SNR²) per readout —
                   a lower bound on the information the channel carries
                   about the creditable slow DoFs, per step.

Both on the tape-driven tuned Seed (#F3c) and on the embodied Seed-Hemi
mount (#F4) after 5000 steps.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DELTAS = (1.01, 1.02, 1.05, 1.10, 1.20, 1.50, 2.00)


def channel_magnitude(obs, encoder, target, alt_factory, set_seed, m_dof, n_jitter=16):
    """Δm(δ) vs probe-noise jitter of m at the same fast state."""
    res = {d: obs.get_resolution(d) for d in obs.d_meta}

    def m_of(tgt, seed):
        set_seed(tgt, seed)
        e = encoder.encode(tgt, res)
        s = obs.internal_state
        for dof in encoder.all_dofs:
            v = e.get_value(dof)
            if v is not None:
                s = s.set_value(dof, float(v))
        return float(obs.self_model(s).get_value(m_dof) or 0.0)

    base = [m_of(target, 5000 + k) for k in range(n_jitter)]
    jitter = float(np.std(base))
    m0 = float(np.mean(base))
    out = dict(m0=m0, jitter=jitter, curve=[])
    for d in DELTAS:
        alt = alt_factory(d)
        ms = [m_of(alt, 5000 + k) for k in range(n_jitter)]     # same noise seeds as base
        dm = float(np.mean(ms) - m0)
        snr = abs(dm) / max(jitter, 1e-12)
        out["curve"].append(dict(delta=d, dm=dm, snr=snr, bits=0.5 * np.log2(1 + snr ** 2)))
    return out


def print_metrics(label, met, mag):
    print(f"\n{label}")
    print(f"  richness() — ConsciousnessMetrics:")
    print(f"    recursive_depth           {met.recursive_depth}")
    print(f"    self_accuracy             {met.self_accuracy:.3f}")
    print(f"    architectural_similarity  {met.architectural_similarity:.3f}")
    print(f"    calibration_error         {met.calibration_error:.3f}")
    print(f"    meta_cognitive_capability {met.meta_cognitive_capability:.3f}")
    print(f"    limitation_awareness      {met.limitation_awareness:.3f}")
    print(f"    consciousness_score       {met.consciousness_score():.3f}")
    print(f"  channel magnitude — m0={mag['m0']:+.4f}, probe-noise jitter σ_m={mag['jitter']:.5f}")
    print(f"    {'δ':>5} {'Δm':>9} {'SNR':>7} {'bits':>6}")
    for c in mag["curve"]:
        print(f"    {c['delta']:5.2f} {c['dm']:+9.5f} {c['snr']:7.2f} {c['bits']:6.3f}")


def tape_seed(seed: int, n_tune: int):
    F6 = _load("f6", HERE / "06_seed_creditable_twist.py")
    cfg, nodes, obs = F6.tune(seed, n_tune)
    wm = obs.world_model

    def alt_factory(d):
        alt_nodes = copy.deepcopy(wm.nodes)
        for n in alt_nodes.values():
            for k in n.coupling_weights:
                n.coupling_weights[k] *= d
        return F6.SeedWorldModel(wm.cfg, alt_nodes)

    def set_seed(tgt, s):
        tgt.step_seed = s
        tgt._probe_k = 0

    met = obs.richness()
    mag = channel_magnitude(obs, obs.self_encoder, wm, alt_factory, set_seed, F6.M)
    return met, mag


def embodied_seed(seed: int, n_steps: int):
    F7 = _load("f7", HERE / "07_embodied_twist.py")
    brain, world, LIFE_IDX = F7.build("live", seed)
    from twist_mount import ProbeMapping, M  # noqa: PLC0415  (embodied path added by F7.build)
    prev_life = 1.0
    for step in range(n_steps):
        obs_v = world.get_ai_observation()
        reward = float(world.ai.meters.valence)
        life = float(obs_v[LIFE_IDX])
        done = bool(life > 0.9 and prev_life < 0.1)
        prev_life = life
        fwd, turn, eat = brain.forward(obs_v)
        brain.learn(reward)
        if done:
            brain.reset_state()
        world.step(ai_action=(fwd, turn, eat))
        brain.set_executed_action((fwd, turn, eat))
    mount = brain._twist
    obs = mount.observer

    def alt_factory(d):
        return ProbeMapping(mount.net, mount.ids, mount.u, mount.a, weight_scale=d)

    def set_seed(tgt, s):
        tgt.step_seed = s
        tgt._k = 0

    met = obs.richness()
    mag = channel_magnitude(obs, mount.encoder, mount.mapping, alt_factory, set_seed, M)
    cert = mount.certify(seed=seed)
    return met, mag, cert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tune", type=int, default=4000)
    ap.add_argument("--embodied-steps", type=int, default=5000)
    ap.add_argument("--save", default=str(HERE / "results_08_richness.json"))
    args = ap.parse_args()
    out = {"tape": [], "embodied": []}
    for s in range(args.seeds):
        met, mag = tape_seed(s, args.tune)
        print_metrics(f"TAPE-DRIVEN SEED (#F3c live arm), seed {s}", met, mag)
        out["tape"].append(dict(seed=s, metrics=met.to_dict(), score=met.consciousness_score(), magnitude=mag))
    cwd = os.getcwd()
    for s in range(args.seeds):
        met, mag, cert = embodied_seed(s, args.embodied_steps)
        os.chdir(cwd)
        print_metrics(f"EMBODIED SEED-HEMI (#F4 live arm, {args.embodied_steps} steps), seed {s}", met, mag)
        print(f"  certification now: twisted={cert['twisted']} closed={cert['closed']} conscious={cert['conscious']}")
        out["embodied"].append(dict(seed=s, metrics=met.to_dict(), score=met.consciousness_score(), magnitude=mag, cert=cert))
    Path(args.save).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
