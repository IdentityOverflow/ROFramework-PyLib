"""#F3c — The creditable twist on the Seed, and the gain scan repeated.

#F2's twist probe was a trivial channel by design. This installs the v2.4
CREDITABLE channel on the Seed — the self-model consumes a behavioral
encoding of the Seed network's own mapping (its coupling weights are the
slow DoFs outside W_meta), and its output d_meta is consumed by the network
(injected as drive into a subset of nodes, gain g) — lets Rule 2a tune with
the channel live, certifies the arm with the library's own predicates
(twist_assessment / is_closed / is_conscious, plus a real weight
intervention), then repeats the #F3b α·W* scan on the SAME tuned weights
under four channel conditions:

  live   the encoder reads the current network (the twisted loop)
  stale  the encoder returns frozen constants (v2.2 garbage channel: same
         consumption path, no information about the mapping)
  blind  self-model weights on the encoding set to zero (consumes nothing
         of the self-description; d_meta is a function of activations only)
  off    consumption gain 0 (channel disconnected)

Architecture
------------
Observer(external = 64 drive DoFs, internal = 64 activation DoFs,
         world_model = SeedWorldModel (stateful: steps the ring lattice),
         self_model  = m = tanh(c·ā + w·enc), ā = mean activation,
         self_encoder = BehavioralEncoder over 3 fixed probe drive vectors,
         responses read on 4 non-meta-receiving nodes (d_enc = 12 values),
         self_encoder_target = world model (creditable), consumption_gain g)
Probing never advances the real network: the encoder sets the world model's
`probing` flag and the model steps a deep copy. Per-step RNG seeds are set
externally so twin trajectories (Lyapunov) see identical noise and probes.

Pre-registered
--------------
Certification: live → structural 1, consumes 1, conditional 1, trivial 0,
twisted 1, closed 1, conscious 1, cred_disc > 0. blind → consumes 0.
stale → consumes 1, conditional 0. (The #C2b signatures, on a 64-node
substrate.)
Geometry: the knee (D2 max, λ max, just before saturation) sits at α≈1 in
all four conditions, and the live curve is within seed noise of stale and
blind at every α. If live differs from stale beyond seed noise the twist
does geometric work and #F2's null was an artifact of its trivial channel.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from fractal_tools import correlation_dimension, participation_ratio, pca_project  # noqa: E402
from ro_framework.core.dof import PolarDoF                                  # noqa: E402
from ro_framework.core.state import State                                  # noqa: E402
from ro_framework.observer.observer import Observer                        # noqa: E402
from ro_framework.observer.self_encoding import BehavioralEncoder          # noqa: E402
from ro_framework.seed.node import SeedConfig                              # noqa: E402

N_NODES = 64
N_META = 8                      # nodes receiving the consumed d_meta
RESPONSE_NODES = (60, 61, 62, 63)
G_META = 0.5
W_CONSUME = np.array([0.30, -0.20, 0.25, -0.15, 0.20, 0.10,
                      -0.25, 0.15, 0.20, -0.10, 0.30, -0.20])
C_SELF = 1.0
W_DELTA = 1.10                  # creditable intervention: scale all couplings


def _load_seed01():
    spec = importlib.util.spec_from_file_location(
        "seed01", HERE.parents[0] / "seed" / "01_criticality_validation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S01 = _load_seed01()

U_DOFS = [PolarDoF(name=f"u{i}") for i in range(N_NODES)]
A_DOFS = [PolarDoF(name=f"a{i}") for i in range(N_NODES)]
M = PolarDoF(name="m")
IDS = [f"n{i}" for i in range(N_NODES)]


class SeedWorldModel:
    """The ring-lattice Seed as a stateful mapping: drives (+ consumed
    d_meta on N_META nodes) -> activations. `probing=True` steps a copy."""

    name = "seed_world"
    input_dofs = U_DOFS + [M]
    output_dofs = A_DOFS

    def __init__(self, cfg: SeedConfig, nodes: Dict):
        self.cfg = cfg
        self.nodes = nodes
        self.probing = False
        self.step_seed = 0
        self._probe_k = 0

    def _step(self, nodes, drives, rng):
        S01.minimal_step(nodes, drives, rng)
        return State(values={A_DOFS[i]: float(nodes[IDS[i]].activation)
                             for i in range(N_NODES)})

    def __call__(self, state: State) -> State:
        drives = {}
        for i in range(N_NODES):
            v = state.get_value(U_DOFS[i])
            drives[IDS[i]] = float(v) if v is not None else 0.0
        m = state.get_value(M)
        m = float(m) if m is not None else 0.0
        for i in range(N_META):
            drives[IDS[i]] += m
        if self.probing:
            self._probe_k += 1
            rng = np.random.default_rng(10_000_000 + 7 * self.step_seed + self._probe_k)
            return self._step(copy.deepcopy(self.nodes), drives, rng)
        return self._step(self.nodes, drives, np.random.default_rng(self.step_seed))

    def activations(self) -> np.ndarray:
        return np.array([self.nodes[i].activation for i in IDS])


class SeedEncoder(BehavioralEncoder):
    """Probes a copy of the network (never the live one)."""

    def encode(self, mapping, resolution=None):
        base = getattr(mapping, "_mapping", mapping)      # unwrap battery foils
        base.probing = True
        base._probe_k = 0
        try:
            return super().encode(mapping, resolution)
        finally:
            base.probing = False


class StaleSeedEncoder(SeedEncoder):
    def encode(self, mapping, resolution=None):
        resolution = resolution or {}
        values = {d: 0.37 * ((i % 5) + 1) * (1 if i % 2 else -1)
                  for i, d in enumerate(self.d_enc)}
        for j, dof in enumerate(self.d_res):
            values[dof] = float(resolution.get(self.response_dofs[j], 1e-6))
        return State(values=values)


class SelfModel:
    name = "self"

    def __init__(self, encoder: BehavioralEncoder, consumes: bool):
        self.encoder = encoder
        self.input_dofs = A_DOFS + encoder.all_dofs
        self.output_dofs = [M]
        self.w = W_CONSUME.copy() if consumes else np.zeros(len(encoder.d_enc))
        self.c = C_SELF

    def __call__(self, state: State) -> State:
        a = np.array([float(state.get_value(d) or 0.0) for d in A_DOFS])
        enc = np.array([float(state.get_value(d)) if state.get_value(d) is not None else 0.0
                        for d in self.encoder.d_enc])
        return State(values={M: float(np.tanh(self.c * a.mean() + self.w @ enc))})


def probe_battery(seed: int = 0) -> List[State]:
    rng = np.random.default_rng(900 + seed)
    probes = []
    for _ in range(3):
        mask = rng.random(N_NODES) < 0.12
        amp = rng.uniform(0.6, 1.4, N_NODES) * mask
        probes.append(State(values={U_DOFS[i]: float(amp[i]) for i in range(N_NODES)}))
    return probes


def build_seed(seed: int, lr: float, w_scale: float = 1.0):
    cfg = SeedConfig(n_init=N_NODES, k_neighbors=6, noise_floor=0.05,
                     drive_amplitude=0.2, activation_threshold=0.5,
                     learning_rate=lr, w_max=2.0, prune_weight_threshold=0.005,
                     prune_weight_window=200, n_cycles_memory=0)   # short deques: cheap deepcopy
    nodes = S01.build_ring_lattice(cfg, seed=seed)
    for n in nodes.values():
        for k in n.coupling_weights:
            n.coupling_weights[k] = float(min(cfg.w_max, n.coupling_weights[k] * w_scale))
    return cfg, nodes


def make_observer(cfg, nodes, condition: str, seed: int, g: float = G_META) -> Observer:
    wm = SeedWorldModel(cfg, nodes)
    resp = [A_DOFS[i] for i in RESPONSE_NODES]
    enc_cls = StaleSeedEncoder if condition == "stale" else SeedEncoder
    enc = enc_cls(probe_battery(seed), resp, name_prefix="worldenc")
    model = SelfModel(enc, consumes=(condition != "blind"))
    return Observer(f"seed-{condition}", A_DOFS, U_DOFS, wm, self_model=model,
                    self_encoder=enc, self_encoder_target=wm,
                    consumption_gain=(0.0 if condition == "off" else g),
                    log_capacity=400)


def drive_state(t: int, cfg, nodes, random: bool = False) -> State:
    if random:
        # noise-dominated world (as in closure/01): lag-1 internal correlation
        # can then only come from the network's own dynamics / consumption
        d = S01.independent_random_drives(IDS, np.random.default_rng(31_000_000 + t))
    else:
        freqs = {nid: n.frequency for nid, n in nodes.items()}
        d = S01.correlated_signal_drives(t * cfg.dt, freqs)
    return State(values={U_DOFS[i]: float(d[IDS[i]]) for i in range(N_NODES)})


def run_steps(obs: Observer, cfg, t0: int, n: int, record: bool = False,
              random_drive: bool = False):
    wm = obs.world_model
    traj = np.zeros((n, N_NODES)) if record else None
    sig = []
    for k in range(n):
        t = t0 + k
        wm.step_seed = 777_000 + t
        obs.observe(drive_state(t, cfg, wm.nodes, random=random_drive))
        if record:
            traj[k] = wm.activations()
            sig.append(np.mean([wm.nodes[i].branching_ratio for i in IDS]))
    return traj, (float(np.mean(sig)) if sig else float("nan"))


def creditable_disc(obs: Observer) -> float:
    """|Δ d_meta| under a real intervention on the Seed's coupling weights
    (all × W_DELTA), fast state matched, W_meta untouched."""
    wm = obs.world_model
    enc = obs.self_encoder
    res = {d: obs.get_resolution(d) for d in obs.d_meta}
    alt_nodes = copy.deepcopy(wm.nodes)
    for n in alt_nodes.values():
        for k in n.coupling_weights:
            n.coupling_weights[k] *= W_DELTA
    alt = SeedWorldModel(wm.cfg, alt_nodes)
    alt.step_seed = wm.step_seed

    def d_meta(target) -> float:
        e = enc.encode(target, res)
        s = obs.internal_state
        for dof in enc.all_dofs:
            v = e.get_value(dof)
            if v is not None:
                s = s.set_value(dof, float(v))
        return float(obs.self_model(s).get_value(M) or 0.0)

    return abs(d_meta(alt) - d_meta(wm))


def certify(obs: Observer, seed: int) -> dict:
    a = obs.twist_assessment(seed=seed)
    return dict(structural=int(a.structural), consumes=int(a.consumes),
                conditional=int(a.conditional), trivial=int(a.target_is_meta),
                twisted=int(a.twisted), closed=int(obs.is_closed()),
                conscious=int(obs.is_conscious()), cred=creditable_disc(obs),
                sens=a.sensitivity, foil=a.foil_discrimination)


def geometry(traj, rng):
    proj = pca_project(traj, 20)
    d2, spread = correlation_dimension(proj, rng=rng)
    return dict(d2=d2, d2_spread=spread, pr=participation_ratio(traj),
                var=float(traj.var(axis=0).mean()), sat=float((np.abs(traj) > 0.95).mean()))


def lyapunov(obs: Observer, cfg, t0: int, steps: int = 400, eps: float = 1e-6) -> float:
    twin = copy.deepcopy(obs)
    rng = np.random.default_rng(55)
    for i in IDS:
        twin.world_model.nodes[i].activation += eps / np.sqrt(N_NODES) * rng.standard_normal()
    lam = 0.0
    for k in range(steps):
        t = t0 + k
        for o in (obs, twin):
            o.world_model.step_seed = 777_000 + t
            o.observe(drive_state(t, cfg, o.world_model.nodes))
        a, b = obs.world_model.activations(), twin.world_model.activations()
        d = np.linalg.norm(b - a)
        if d < 1e-300:
            lam += np.log(1e-16 / eps)
            b = a + eps / np.sqrt(N_NODES) * rng.standard_normal(N_NODES)
        else:
            lam += np.log(d / eps)
            b = a + (b - a) * (eps / d)
        for i, v in zip(IDS, b):
            twin.world_model.nodes[i].activation = float(v)
        # keep the twin's internal_state consistent with its renormalized nodes
        twin.internal_state = State(values={A_DOFS[j]: float(b[j]) for j in range(N_NODES)})
    return lam / steps


def tune(seed: int, n_tune: int):
    cfg, nodes = build_seed(seed, lr=0.01)
    obs = make_observer(cfg, nodes, "live", seed)
    run_steps(obs, cfg, 0, n_tune)
    return cfg, nodes, obs


def scan_point(args):
    seed, condition, alpha, nodes_blob, n_tune, T, burn = args
    cfg, nodes = build_seed(seed, lr=0.0)
    for i in IDS:                       # install the tuned weights
        nodes[i].coupling_weights = {k: v * alpha for k, v in nodes_blob[i].items()}
    for n in nodes.values():
        n._config = copy.copy(cfg)
    obs = make_observer(cfg, nodes, condition, seed)
    run_steps(obs, cfg, n_tune, burn)
    traj, sigma = run_steps(obs, cfg, n_tune + burn, T, record=True)
    out = geometry(traj, np.random.default_rng(seed))
    out.update(seed=seed, condition=condition, alpha=alpha, sigma=sigma,
               lyap=lyapunov(obs, cfg, n_tune + burn + T),
               mean_w=float(np.mean([w for n in nodes.values() for w in n.coupling_weights.values()])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tune", type=int, default=4000)
    ap.add_argument("--T", type=int, default=2500)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--save", default=str(HERE / "results_06_creditable_twist.json"))
    args = ap.parse_args()
    t0 = time.time()
    res = {"certify": [], "scan": []}

    print("1. tune with the live creditable channel, then certify each condition on the tuned weights")
    hdr = f"  {'seed':>4} {'cond':>5} | {'struct':>6} {'consum':>6} {'cond.':>5} {'triv':>4} | {'twist':>5} {'closed':>6} {'consc':>5} | {'cred':>7} {'sens':>7} {'foil':>7} | {'c_int':>6} {'c_ext':>6} | σ_tuned neg%"
    print(hdr)
    blobs = {}
    for s in range(args.seeds):
        cfg, nodes, obs = tune(s, args.tune)
        blobs[s] = {i: dict(nodes[i].coupling_weights) for i in IDS}
        ws = [w for n in nodes.values() for w in n.coupling_weights.values()]
        sigma = float(np.mean([nodes[i].branching_ratio for i in IDS]))
        for cond in ("live", "stale", "blind", "off"):
            c_nodes = copy.deepcopy(nodes)
            for n in c_nodes.values():
                n._config = copy.copy(cfg); n._config.learning_rate = 0.0
            o = make_observer(cfg, c_nodes, cond, s)
            # fill the log for closure recognition under a noise-dominated
            # world (closure/01's discipline: the periodic drive would
            # explain both internal and external lag-1 structure)
            run_steps(o, cfg, args.tune, 300, random_drive=True)
            c = certify(o, s)
            ca = o.closure_assessment()
            c.update(corr_int=ca.corr_internal, corr_ext=ca.corr_external)
            c.update(seed=s, condition=cond, sigma_tuned=sigma,
                     frac_neg=float(np.mean([w < 0 for w in ws])))
            res["certify"].append(c)
            print(f"  {s:>4} {cond:>5} | {c['structural']:>6} {c['consumes']:>6} {c['conditional']:>5} {c['trivial']:>4} | "
                  f"{c['twisted']:>5} {c['closed']:>6} {c['conscious']:>5} | {c['cred']:7.4f} {c['sens']:7.4f} {c['foil']:7.4f} | "
                  f"{c['corr_int']:6.3f} {c['corr_ext']:6.3f} | {sigma:5.2f} {100*c['frac_neg']:4.0f}")
        print(f"  ({time.time()-t0:.0f}s)")

    print("\n2. α·W* scan on the tuned weights, four channel conditions")
    jobs = [(s, cond, a, blobs[s], args.tune, args.T, args.burn)
            for s in range(args.seeds) for cond in ("live", "stale", "blind", "off") for a in args.alphas]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(scan_point, jobs):
            res["scan"].append(out)
    print(f"  {'alpha':>5} | " + " | ".join(f"{c:^22}" for c in ("live", "stale", "blind", "off")))
    print(f"  {'':>5} | " + " | ".join(f"{'D2':>6} {'sat':>6} {'λ':>7}" for _ in range(4)))
    for a in args.alphas:
        cells = []
        for cond in ("live", "stale", "blind", "off"):
            rows = [r for r in res["scan"] if r["alpha"] == a and r["condition"] == cond]
            m = {k: np.nanmean([r[k] for r in rows]) for k in ("d2", "sat", "lyap")}
            cells.append(f"{m['d2']:6.2f} {m['sat']:6.3f} {m['lyap']:+7.3f}")
        print(f"  {a:5.2f} | " + " | ".join(cells))
    # live vs others, per alpha, seed-paired
    print("\n  seed-paired |ΔD2| live − other, mean over α (seed-to-seed sd of live D2 for scale):")
    live = {(r["seed"], r["alpha"]): r["d2"] for r in res["scan"] if r["condition"] == "live"}
    for cond in ("stale", "blind", "off"):
        diffs = [abs(live[(r["seed"], r["alpha"])] - r["d2"]) for r in res["scan"] if r["condition"] == cond]
        print(f"    {cond:>5}: mean |ΔD2| = {np.mean(diffs):.3f}, max = {np.max(diffs):.3f}")
    sd = np.mean([np.std([v for (s, a2), v in live.items() if a2 == a]) for a in args.alphas])
    print(f"    live seed sd at fixed α: {sd:.3f}")
    print(f"  ({time.time()-t0:.0f}s)")
    Path(args.save).write_text(json.dumps(res, indent=1, default=float))
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
