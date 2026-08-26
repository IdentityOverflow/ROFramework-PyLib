"""#C2 — The state-matched intervention test: does the conditional clause add content?

ro_framework.md v2.2 states the twist as a conditional information quantity:
twisted(O) iff I(d_meta ; M_self | S_internal) > 0, operationalized by the
battery-foil test (interventions on the mapping that agree with it on the
entire runtime history, routed through the encoder; d_meta must distinguish
foil from original at matched state).

This experiment measures whether that clause discriminates beyond the
white-box consumption checks (sensitivity/permutation discrimination),
using three arms:

  T (live)    : self-model consumes a live behavioral self-encoding.
                Prediction: passes white-box AND conditional.
  U (blind)   : identical architecture, ignores the encoding.
                Prediction: fails both.
  G (garbage) : consuming self-model on a STALE encoder (frozen constants;
                the channel no longer tracks the mapping).
                Prediction: passes white-box, FAILS conditional — the
                dissociation that shows the conditional clause is not
                redundant with the white-box checks.

Also reported: foil-discrimination magnitude vs foil scale (the graded
richness reading), and per-arm is_conscious() with consumption on.

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

PROBE_VALUES = (-0.8, 0.0, 0.8)


class WorldModel:
    name = "world"
    input_dofs = [X, M]
    output_dofs = [Y]

    def __call__(self, state: State) -> State:
        x = float(state.get_value(X) or 0.0)
        m = state.get_value(M)
        m = float(m) if m is not None else 0.0
        return State(values={Y: float(np.tanh(1.5 * (0.6 * x + m)))})


class SelfModel:
    """meta = tanh(c*y + w.enc + b); w nonzero = consumes the encoding."""

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
    """Garbage channel: encode() ignores the mapping (frozen constants)."""

    def encode(self, mapping, resolution=None):
        resolution = resolution or {}
        values = {}
        for i, dof in enumerate(self.d_enc):
            values[dof] = 0.37 * (i + 1)
        for j, dof in enumerate(self.d_res):
            values[dof] = float(resolution.get(self.response_dofs[j], 1e-6))
        return State(values=values)


def build(arm: str, seed: int) -> Observer:
    probes = [State(values={Y: v}) for v in PROBE_VALUES]
    enc_cls = StaleEncoder if arm == "G" else BehavioralEncoder
    encoder = enc_cls(probes, [M])
    model = SelfModel(encoder, consumes=(arm in ("T", "G")))
    return Observer(f"c2-{arm}", [Y], [X], WorldModel(),
                    self_model=model, self_encoder=encoder,
                    consumption_gain=0.5, log_capacity=400)


def run_arm(arm: str, seed: int, n_drive: int, foil_scales) -> dict:
    rng = np.random.default_rng(seed)
    obs = build(arm, seed)
    for t in range(n_drive):
        x = float(0.3 * np.sin(0.5 * t) + 0.8 * rng.normal())
        obs.observe(State(values={X: x}))

    a = obs.twist_assessment(seed=seed)
    scales = {}
    for fs in foil_scales:
        scales[fs] = obs.twist_assessment(seed=seed, foil_scale=fs
                                          ).foil_discrimination
    return {
        "consumes": int(a.consumes),
        "sens": a.sensitivity,
        "foil": a.foil_discrimination,
        "conditional": int(a.conditional),
        "twisted": int(a.twisted),
        "closed": int(obs.is_closed()),
        "conscious": int(obs.is_conscious()),
        "scales": scales,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--drive", type=int, default=200)
    ap.add_argument("--foil-scales", type=float, nargs="+",
                    default=[0.1, 0.3, 0.6])
    args = ap.parse_args()

    print(f"#C2 conditional twist — seeds={args.seeds} drive={args.drive}")
    hdr = (f"{'arm':>4} | {'consumes':>8} {'sens':>7} | {'foil_disc':>9} "
           f"{'cond':>5} | {'twisted':>7} {'closed':>6} {'conscious':>9}")
    print(hdr)
    print("-" * len(hdr))
    all_rows = {}
    for arm in ("T", "U", "G"):
        rows = [run_arm(arm, s, args.drive, args.foil_scales)
                for s in range(args.seeds)]
        all_rows[arm] = rows
        def agg(k):
            return float(np.mean([r[k] for r in rows]))
        print(f"{arm:>4} | {agg('consumes'):>8.2f} {agg('sens'):>7.4f} | "
              f"{agg('foil'):>9.4f} {agg('conditional'):>5.2f} | "
              f"{agg('twisted'):>7.2f} {agg('closed'):>6.2f} "
              f"{agg('conscious'):>9.2f}")

    print("\nfoil discrimination vs intervention scale (richness reading):")
    print(f"{'arm':>4} | " + "  ".join(f"fs={fs:>4}" for fs in args.foil_scales))
    for arm in ("T", "U", "G"):
        means = [float(np.mean([r['scales'][fs] for r in all_rows[arm]]))
                 for fs in args.foil_scales]
        print(f"{arm:>4} | " + "  ".join(f"{m:>7.4f}" for m in means))

    t_ok = all(r["twisted"] == 1 and r["conditional"] == 1
               for r in all_rows["T"])
    u_ok = all(r["twisted"] == 0 for r in all_rows["U"])
    g_ok = all(r["consumes"] == 1 and r["conditional"] == 0
               and r["twisted"] == 0 for r in all_rows["G"])
    print(f"\npre-registered: T passes both [{t_ok}], U fails both [{u_ok}], "
          f"G dissociates (white-box pass, conditional fail) [{g_ok}]")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
