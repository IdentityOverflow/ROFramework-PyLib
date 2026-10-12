"""
experiments/embodied/twist_mount.py — a creditable twist channel on a live SeedBrain

Mounts the v2.4 self-indexing loop (experiments/feedback/06, #F3c) on the
embodied hemispheric Seed, the same way holo_mount.py mounts the reel:
opt-in, config-keyed, byte-exact behaviour when disabled.

The loop
--------
  encoder     behavioral encoding of the Seed network's own one-step mapping:
              3 fixed probe drive patterns are applied to a PURE copy of the
              node update (weights, phases and activations read, nothing
              mutated) and the responses of 4 designated nodes are read.
              d_enc = 12 numbers that are functions of the coupling weights —
              slow DoFs outside W_meta: the creditable target.
  self-model  m = tanh(mean activation + w · enc), w fixed (= W_meta).
  consumption m · gain is added to the sensor drive of 8 general nodes
              (4 per hemisphere) on the next step.
Conditions (as in #F3c): live, stale (frozen constants through the same
path), blind (w = 0), off (gain 0 — channel present, not consumed).

A core-library Observer rides alongside for certification: its log is
filled from the brain's own steps (external = the 5 embodied DoFs,
internal = node activations over a node list frozen at mount), so
twist_assessment(), is_closed() and is_conscious() can be asked at any time.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ro_framework.core.dof import PolarDoF
from ro_framework.core.state import State
from ro_framework.observer.observer import Observer, ObservationPair
from ro_framework.observer.self_encoding import BehavioralEncoder

import dofs

M = PolarDoF(name="twist_m")
W_CONSUME = np.array([0.30, -0.20, 0.25, -0.15, 0.20, 0.10,
                      -0.25, 0.15, 0.20, -0.10, 0.30, -0.20])


class ProbeMapping:
    """The network's one-step activation map as a pure function.

    input: per-node drive DoFs (+ M, ignored here: the battery is presented
    with M blank), output: activations of the frozen node list. Reads
    weights / activations / phases from the live network, mutates nothing.
    `weight_scale` lets the creditable intervention build a counterfactual
    mapping (all couplings scaled) without touching the network.
    """

    name = "seed_probe"

    def __init__(self, network, node_ids: List[str], u_dofs, a_dofs,
                 weight_scale: float = 1.0):
        self.net = network
        self.ids = node_ids
        self.u = u_dofs
        self.a = a_dofs
        self.input_dofs = list(u_dofs) + [M]
        self.output_dofs = list(a_dofs)
        self.weight_scale = weight_scale
        self.step_seed = 0
        self._k = 0

    def __call__(self, state: State) -> State:
        nodes = self.net.nodes
        cfg = self.net.config
        self._k += 1
        rng = np.random.default_rng(10_000_000 + 7 * self.step_seed + self._k)
        out = {}
        for i, nid in enumerate(self.ids):
            n = nodes.get(nid)
            if n is None:
                out[self.a[i]] = 0.0
                continue
            v = state.get_value(self.u[i])
            drive = float(v) if v is not None else 0.0
            inp = sum(self.weight_scale * w * nodes[o].activation
                      for o, w in n.coupling_weights.items() if o in nodes)
            osc = n.drive_amplitude * math.sin(n.phase + 2 * math.pi * n.frequency * cfg.dt)
            out[self.a[i]] = float(np.tanh(inp + drive + osc + n.noise_floor * rng.standard_normal()))
        return State(values=out)


class _StaleEncoder(BehavioralEncoder):
    def encode(self, mapping, resolution=None):
        resolution = resolution or {}
        values = {d: 0.37 * ((i % 5) + 1) * (1 if i % 2 else -1)
                  for i, d in enumerate(self.d_enc)}
        for j, dof in enumerate(self.d_res):
            values[dof] = float(resolution.get(self.response_dofs[j], 1e-6))
        return State(values=values)


class _SelfModel:
    """m_j = tanh(c_j · src_j + W_j · enc + b_j), j = 1..n_out.

    n_out = 1 (default): src = mean activation, W = W_CONSUME (or 0 if
    blind) — byte-identical to the original scalar channel.
    n_out = k: one channel per meta node; src_j = that node's activation.
    adapt(): delta rule making m_j a one-step forward model of src_j, with
    W frozen at 0 for a blind model (same forward model, no self-encoding).
    """

    name = "twist_self"

    def __init__(self, encoder: BehavioralEncoder, a_dofs, consumes: bool,
                 n_out: int = 1, src_idx=None, rng=None):
        self.encoder = encoder
        self.a = a_dofs
        self.input_dofs = list(a_dofs) + encoder.all_dofs
        self.n_out = n_out
        self.out = [M] if n_out == 1 else [PolarDoF(name=f"twist_m{j}") for j in range(n_out)]
        self.output_dofs = list(self.out)
        self.src_idx = None if src_idx is None else np.asarray(src_idx)
        self.consumes = consumes
        n_enc = len(encoder.d_enc)
        if n_out == 1:
            self.W = (W_CONSUME.copy() if consumes else np.zeros(n_enc))[None, :]
        else:
            rng = rng or np.random.default_rng(0)
            self.W = rng.standard_normal((n_out, n_enc)) * 0.25 if consumes else np.zeros((n_out, n_enc))
        self.c = np.ones(n_out)
        self.b = np.zeros(n_out)
        self._last = None

    @property
    def w(self):                       # scalar-channel compatibility
        return self.W[0]

    def _inputs(self, state: State):
        a = np.array([float(state.get_value(d) or 0.0) for d in self.a])
        enc = np.array([float(state.get_value(d)) if state.get_value(d) is not None else 0.0
                        for d in self.encoder.d_enc])
        src = a.mean() * np.ones(self.n_out) if self.src_idx is None else a[self.src_idx]
        return src, enc

    def __call__(self, state: State) -> State:
        src, enc = self._inputs(state)
        m = np.tanh(self.c * src + self.W @ enc + self.b)
        self._last = (src, enc, m)
        return State(values={d: float(m[j]) for j, d in enumerate(self.out)})

    def adapt(self, last, target: np.ndarray, eta: float) -> float:
        """One delta-rule step on the prediction made from `last`
        (= (src, enc, m) captured at pre_step). Returns mean |error|."""
        src, enc, m = last
        err = target - m
        g = err * (1.0 - m ** 2)
        if self.consumes:
            self.W += eta * np.outer(g, enc)
        self.c += eta * g * src
        self.b += eta * g
        return float(np.mean(np.abs(err)))


class TwistMount:
    def __init__(self, network, condition: str = "live", gain: float = 0.5,
                 n_meta: int = 8, n_resp: int = 4, seed: int = 0,
                 log_capacity: int = 400, n_out: int = 1, adaptive: bool = False,
                 site: str = "drive", eta: float = 0.01) -> None:
        assert condition in ("live", "stale", "blind", "off")
        assert site in ("drive", "gain", "lr")
        assert n_out in (1, n_meta), "vector channel = one output per meta node"
        self.net = network
        self.condition = condition
        self.gain = 0.0 if condition == "off" else float(gain)
        self.n_out, self.adaptive, self.site, self.eta = n_out, adaptive, site, eta
        self.lr_factor = 1.0
        self.pred_err = float("nan")
        self.ids = sorted(network.nodes.keys(), key=lambda k: int(k[1:]))
        self.u = [PolarDoF(name=f"tw_u_{k}") for k in self.ids]
        self.a = [PolarDoF(name=f"tw_a_{k}") for k in self.ids]
        # meta-receiving and response nodes: general nodes, split by hemisphere
        gen = [k for k in self.ids if network.nodes[k].node_role == "general"]
        L = [k for k in gen if k in network._hemisphere_L]
        R = [k for k in gen if k in network._hemisphere_R]
        h = n_meta // 2
        self.meta_nodes = L[:h] + R[:h]
        self.resp_nodes = L[h:h + n_resp // 2] + R[h:h + n_resp // 2]
        # The self-model's window must sit on the protected core: Rule 5
        # (release) otherwise prunes the response nodes and the encoding
        # goes constant (first embodied run, #F4: channel dead by 5k steps
        # in 3/3 seeds). is_seed_node is the library's release guard.
        for k in self.meta_nodes + self.resp_nodes:
            network.nodes[k].is_seed_node = True
        # probe battery: sparse random drive patterns over the frozen node list
        rng = np.random.default_rng(900 + seed)
        probes = []
        for _ in range(3):
            mask = rng.random(len(self.ids)) < 0.12
            amp = rng.uniform(0.6, 1.4, len(self.ids)) * mask
            probes.append(State(values={self.u[i]: float(amp[i]) for i in range(len(self.ids))}))
        resp = [self.a[self.ids.index(k)] for k in self.resp_nodes]
        enc_cls = _StaleEncoder if condition == "stale" else BehavioralEncoder
        self.encoder = enc_cls(probes, resp, name_prefix="tw_worldenc")
        self.mapping = ProbeMapping(network, self.ids, self.u, self.a)
        src_idx = None if n_out == 1 else [self.ids.index(k) for k in self.meta_nodes]
        self.self_model = _SelfModel(self.encoder, self.a, consumes=(condition != "blind"),
                                     n_out=n_out, src_idx=src_idx,
                                     rng=np.random.default_rng(1234 + seed))
        self.observer = Observer("seed_twist", self.a, dofs.EXTERNAL_DOFS, self.mapping,
                                 self_model=self.self_model, self_encoder=self.encoder,
                                 self_encoder_target=self.mapping,
                                 consumption_gain=self.gain, log_capacity=log_capacity)
        self.m = np.zeros(n_out)
        self.step = 0
        self.n_live = len(self.ids)
        self._last = None

    # -- per-step hooks ------------------------------------------------------

    def activations(self) -> np.ndarray:
        nodes = self.net.nodes
        return np.array([nodes[k].activation if k in nodes else 0.0 for k in self.ids])

    def pre_step(self):
        """Compute d_meta from the current state; return (additive, multiplicative)
        drive injections for the sensor wrapper and set lr_factor."""
        self.mapping.step_seed = self.step
        self.mapping._k = 0
        if self.observer.internal_state is None:
            self.m = np.zeros(self.n_out)
            self._last = None
        else:
            s = self.observer._augment_with_self_encoding(self.observer.internal_state)
            out = self.self_model(s)
            self.m = np.array([float(out.get_value(d) or 0.0) for d in self.self_model.out])
            self._last = self.self_model._last
        m_mean = float(self.m.mean())
        self.lr_factor = 1.0 + self.gain * m_mean if self.site == "lr" else 1.0
        add, mult = {}, {}
        if self.gain == 0.0 or self.site == "lr":
            return add, mult
        for j, k in enumerate(self.meta_nodes):
            mj = float(self.m[j]) if self.n_out > 1 else m_mean
            if self.site == "drive":
                add[k] = self.gain * mj
            else:                                   # "gain": multiplicative on the drive
                mult[k] = 1.0 + self.gain * mj
        return add, mult

    def post_step(self, obs: np.ndarray) -> None:
        acts = self.activations()
        if self.adaptive and self._last is not None:
            src_idx = self.self_model.src_idx
            target = acts.mean() * np.ones(self.n_out) if src_idx is None else acts[src_idx]
            self.pred_err = self.self_model.adapt(self._last, target, self.eta)
        st = State(values={self.a[i]: float(acts[i]) for i in range(len(self.ids))})
        self.observer.internal_state = st
        self.observer.observation_log.append(ObservationPair(
            external_state=dofs.obs_to_state(obs), internal_state=st,
            timestamp=float(self.step)))
        self.n_live = sum(1 for k in self.ids if k in self.net.nodes)
        self.step += 1

    # -- certification -------------------------------------------------------

    def creditable_disc(self, scale: float = 1.10) -> float:
        obs = self.observer
        res = {d: obs.get_resolution(d) for d in obs.d_meta}
        alt = ProbeMapping(self.net, self.ids, self.u, self.a, weight_scale=scale)
        alt.step_seed = self.mapping.step_seed

        def d_meta(target):
            self.mapping._k = 0
            alt._k = 0
            e = self.encoder.encode(target, res)
            s = obs.internal_state
            for dof in self.encoder.all_dofs:
                v = e.get_value(dof)
                if v is not None:
                    s = s.set_value(dof, float(v))
            o = self.self_model(s)
            return np.array([float(o.get_value(d) or 0.0) for d in self.self_model.out])

        return float(np.linalg.norm(d_meta(alt) - d_meta(self.mapping)))

    def certify(self, seed: int = 0) -> dict:
        a = self.observer.twist_assessment(seed=seed)
        c = self.observer.closure_assessment()
        return dict(structural=int(a.structural), consumes=int(a.consumes),
                    conditional=int(a.conditional), trivial=int(a.target_is_meta),
                    twisted=int(a.twisted), closed=int(c.closed),
                    corr_int=float(c.corr_internal), corr_ext=float(c.corr_external),
                    conscious=int(a.twisted and c.closed),
                    cred=self.creditable_disc(), sens=float(a.sensitivity),
                    foil=float(a.foil_discrimination), n_live=self.n_live,
                    resp_live=int(sum(1 for k in self.resp_nodes if k in self.net.nodes)),
                    meta_live=int(sum(1 for k in self.meta_nodes if k in self.net.nodes)))

    def stats(self) -> str:
        return (f"[twist] cond={self.condition} g={self.gain} site={self.site} "
                f"m̄={float(np.mean(self.m)):+.3f} pred_err={self.pred_err:.3f} "
                f"nodes_live={self.n_live}/{len(self.ids)}")


class InjectingSensor:
    """Wraps the brain's sensor; adds the mount's injection to the drives."""

    def __init__(self, inner, mount: TwistMount):
        self.inner = inner
        self.mount = mount
        self.injection = ({}, {})          # (additive, multiplicative)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, external_input, node_frequencies):
        drives = self.inner(external_input, node_frequencies)
        add, mult = self.injection
        for k, v in mult.items():
            if k in drives:
                drives[k] *= v
        for k, v in add.items():
            if k in drives:
                drives[k] += v
        return drives
