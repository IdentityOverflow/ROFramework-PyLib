"""#C2b — The creditable twist: W_meta excluded, the old flagship as negative control.

ro_framework.md v2.4: every activation is a function of its own generating
weights, so a channel probing the very mapping that computes d_meta reads
only W_meta — the trivial channel, no twist credit. The creditable
criterion: I(d_meta ; D_slow(O) \\ W_meta | S_rest) > 0 — the meta channel
must know slow DoFs OUTSIDE its own generating set. Test: vary the
creditable slow DoFs (here: the world model's weights), match the fast
state, HOLD W_meta.

Arms:
  Tw : self-model consumes an encoding of the WORLD model
       (self_encoder_target = world_model) — the creditable channel.
  Ts : self-model consumes an encoding of ITSELF — #C2's former flagship,
       now the negative control: live, consumed, and trivial.
  U  : world-targeted channel, blind self-model.
  G  : world-targeted STALE encoder (frozen constants), consuming model.

Measurements per arm (5 seeds):
  - assess_twist: consumes (white-box), conditional (battery-foil),
    target_is_meta, twisted; is_closed / is_conscious with g=0.5.
  - creditable_disc: REAL weight intervention on the world model
    (a -> a + delta), fast state matched, W_meta untouched: |Δ d_meta|.
    Predictions: Tw > 0 (channel reaches the creditable slow DoFs);
    Ts = 0 exactly (its channel reads only its own W_meta); U = 0
    (blind); G = 0 (stale).

Pre-registered: Tw passes everything; Ts shows the v2.4 signature —
consumes 1, conditional-on-its-own-channel 1, trivial 1, twisted 0,
creditable_disc 0; U fails consumes; G dissociates (consumes 1,
conditional 0). If Ts scored creditable_disc > 0 the exclusion would be
wrong; if Tw scored 0 the creditable channel would not be reaching the
world weights and the architecture claim would be broken.

numpy + core lib; CPU; seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ro_framework.core.dof import PolarDoF                          # noqa: E402
from ro_framework.core.state import State                          # noqa: E402
from ro_framework.observer.observer import Observer                # noqa: E402
from ro_framework.observer.self_encoding import BehavioralEncoder  # noqa: E402

X = PolarDoF(name="x")
Y = PolarDoF(name="y")
M = PolarDoF(name="m")

WORLD_PROBES = [State(values={X: v}) for v in (-0.8, 0.0, 0.8)]
SELF_PROBES = [State(values={Y: v}) for v in (-0.8, 0.0, 0.8)]
W_DELTA = 0.3


class WorldModel:
    """y = tanh(1.5*(a*x + b*m)) — (a, b) are the world's slow DoFs."""

    name = "world"
    input_dofs = [X, M]
    output_dofs = [Y]

    def __init__(self, a: float = 0.6, b: float = 1.0):
        self.a, self.b = a, b

    def __call__(self, state: State) -> State:
        x = float(state.get_value(X) or 0.0)
        m = state.get_value(M)
        m = float(m) if m is not None else 0.0
        return State(values={Y: float(np.tanh(1.5 * (self.a * x
                                                     + self.b * m)))})


class SelfModel:
    """meta = tanh(c*y + w.enc); w nonzero = consumes its encoding inputs."""

    name = "self"

    def __init__(self, encoder: BehavioralEncoder, consumes: bool):
        self.encoder = encoder
        self.input_dofs = [Y] + encoder.all_dofs
        self.output_dofs = [M]
        n = len(encoder.d_enc)
        self.w = (np.array([0.30, -0.20, 0.25])[:n] if consumes
                  else np.zeros(n))

    def __call__(self, state: State) -> State:
        y = float(state.get_value(Y) or 0.0)
        enc = np.array([
            float(state.get_value(d)) if state.get_value(d) is not None
            else 0.0
            for d in self.encoder.d_enc
        ])
        return State(values={M: float(np.tanh(1.0 * y + self.w @ enc))})


class StaleEncoder(BehavioralEncoder):
    def encode(self, mapping, resolution=None):
        resolution = resolution or {}
        values = {}
        for i, dof in enumerate(self.d_enc):
            values[dof] = 0.37 * (i + 1)
        for j, dof in enumerate(self.d_res):
            values[dof] = float(resolution.get(self.response_dofs[j], 1e-6))
        return State(values=values)


def build(arm: str) -> Observer:
    wm = WorldModel()
    if arm == "Ts":
        enc = BehavioralEncoder(SELF_PROBES, [M], name_prefix="selfenc")
        model = SelfModel(enc, consumes=True)
        return Observer("c2b-Ts", [Y], [X], wm, self_model=model,
                        self_encoder=enc, consumption_gain=0.5,
                        log_capacity=400)          # target defaults to self
    enc_cls = StaleEncoder if arm == "G" else BehavioralEncoder
    enc = enc_cls(WORLD_PROBES, [Y], name_prefix="worldenc")
    model = SelfModel(enc, consumes=(arm in ("Tw", "G")))
    return Observer(f"c2b-{arm}", [Y], [X], wm, self_model=model,
                    self_encoder=enc, self_encoder_target=wm,
                    consumption_gain=0.5, log_capacity=400)


def creditable_disc(obs: Observer) -> float:
    """|Δ d_meta| under a REAL world-weight intervention (a -> a+delta),
    fast state matched, W_meta untouched."""
    state = obs.internal_state
    encoder = obs.self_encoder
    target = obs.self_encoder_target or obs.self_model
    resolution = {d: obs.get_resolution(d) for d in obs.d_meta}
    world_mod = WorldModel(a=obs.world_model.a + W_DELTA,
                           b=obs.world_model.b)
    probe_target = world_mod if target is obs.world_model else target

    def d_meta_val(tgt) -> float:
        enc = encoder.encode(tgt, resolution)
        s = state
        for dof in encoder.all_dofs:
            v = enc.get_value(dof)
            if v is not None:
                s = s.set_value(dof, float(v))
        return float(obs.self_model(s).get_value(M) or 0.0)

    return abs(d_meta_val(probe_target) - d_meta_val(target))


def run_arm(arm: str, seed: int, n_drive: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = build(arm)
    for t in range(n_drive):
        x = float(0.3 * np.sin(0.5 * t) + 0.8 * rng.normal())
        obs.observe(State(values={X: x}))
    a = obs.twist_assessment(seed=seed)
    return {
        "consumes": int(a.consumes),
        "conditional": int(a.conditional),
        "trivial": int(a.target_is_meta),
        "twisted": int(a.twisted),
        "closed": int(obs.is_closed()),
        "conscious": int(obs.is_conscious()),
        "cred": creditable_disc(obs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--drive", type=int, default=200)
    args = ap.parse_args()

    print(f"#C2b creditable twist — seeds={args.seeds} drive={args.drive} "
          f"world-weight delta={W_DELTA}")
    hdr = (f"{'arm':>4} | {'consumes':>8} {'cond':>5} {'trivial':>7} | "
           f"{'twisted':>7} {'closed':>6} {'conscious':>9} | {'cred_disc':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows_all = {}
    for arm in ("Tw", "Ts", "U", "G"):
        rows = [run_arm(arm, s, args.drive) for s in range(args.seeds)]
        rows_all[arm] = rows
        def agg(k):
            return float(np.mean([r[k] for r in rows]))
        print(f"{arm:>4} | {agg('consumes'):>8.2f} {agg('conditional'):>5.2f} "
              f"{agg('trivial'):>7.2f} | {agg('twisted'):>7.2f} "
              f"{agg('closed'):>6.2f} {agg('conscious'):>9.2f} | "
              f"{agg('cred'):>9.4f}")

    tw = all(r["twisted"] == 1 and r["conscious"] == 1 and r["cred"] > 1e-6
             for r in rows_all["Tw"])
    ts = all(r["consumes"] == 1 and r["conditional"] == 1
             and r["trivial"] == 1 and r["twisted"] == 0
             and r["cred"] < 1e-12 for r in rows_all["Ts"])
    u = all(r["consumes"] == 0 and r["twisted"] == 0 for r in rows_all["U"])
    g = all(r["consumes"] == 1 and r["conditional"] == 0
            and r["twisted"] == 0 for r in rows_all["G"])
    print(f"\npre-registered: Tw passes all [{tw}], Ts = live+consumed+trivial "
          f"with cred_disc EXACTLY 0 [{ts}], U fails consumes [{u}], "
          f"G dissociates [{g}]")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
