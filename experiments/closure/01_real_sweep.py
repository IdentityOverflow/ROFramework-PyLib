"""#C1 — The real §5.5 sweep: consumption gain on a twisted Observer.

The framework's falsification clause, finally aimed at its actual referent.
Experiment #10 swept a MEMORY-content gain (untwisted cargo) and found
gradualism; ro_framework.md v2 makes its binary-kind claim about the
closure of a TWISTED self-modeling loop. This sweep turns consumption_gain
g on a core-lib Observer whose self-model consumes its own behavioral
encoding — d_meta cargo, the §5.3/§5.4 object itself.

System
------
World:      y = tanh(1.5 * (0.6 * x + m)) — x a slow sine + noise, m the
            consumed self-model output (already g-scaled by the observer).
            Feedback gain > 1: generic bistability is AVAILABLE, which is
            the confound the twin-arm design controls for.
Self-model: pred(next y) = tanh(c*y + w·enc + b); c, b adapt online
            (delta rule, trained at ALL g — prediction is always made;
            only its consumption varies). Arm T: w fixed, distinct,
            nonzero — the twist installed by construction. Arm U: w == 0 —
            identical architecture, blind to its self-description.
            The observer's own twist predicate must agree (T twisted,
            U refused) — the recognition machinery is part of the run.

Protocol
--------
g swept 0 -> 1 -> 0 (11 steps, N cycles each; log cleared per step so all
recognition is per-window), plus downup (time-confound control) and flat0
(drift baseline). 3 seeds x 2 arms.

Per-step readouts:
  closure corr_internal, Closed(O), twisted(O), IS_CONSCIOUS (the v2
  binary, live), self-prediction r.

Pre-registered readings:
  v2-threshold: in arm T, is_conscious() switches ON at some g* ascending
    and OFF at a LOWER g descending (g_on > g_off: hysteresis of the kind
    criterion itself), with closure-corr showing threshold structure that
    EXCEEDS arm U's (twist-specific, not generic feedback dynamics).
  Deflation: smooth path-independent curves, or T == U (any threshold is
    the tanh's, not the twist's), or is_conscious() flickering noisily
    (the binary criterion fails to be a stable kind at all).

numpy + core lib; CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ro_framework.core.dof import PolarDoF                     # noqa: E402
from ro_framework.core.state import State                     # noqa: E402
from ro_framework.observer.observer import Observer           # noqa: E402
from ro_framework.observer.self_encoding import BehavioralEncoder  # noqa: E402

X = PolarDoF(name="x")
Y = PolarDoF(name="y")
M = PolarDoF(name="m")

PROBE_VALUES = (-0.8, 0.0, 0.8)
W_TWIST = np.array([0.30, -0.20, 0.25])


class WorldModel:
    name = "world"
    input_dofs = [X, M]
    output_dofs = [Y]

    def __call__(self, state: State) -> State:
        x = float(state.get_value(X) or 0.0)
        m = state.get_value(M)
        m = float(m) if m is not None else 0.0
        return State(values={Y: float(np.tanh(1.5 * (0.6 * x + m)))})


class AdaptiveSelfModel:
    """pred(next y) = tanh(c*y + w.enc + b); c,b adapt, w fixed (the twist)."""

    name = "self_adaptive"

    def __init__(self, encoder: BehavioralEncoder, twisted: bool,
                 eta: float = 0.05):
        self.encoder = encoder
        self.input_dofs = [Y] + encoder.all_dofs
        self.output_dofs = [M]
        self.w = W_TWIST.copy() if twisted else np.zeros(len(encoder.d_enc))
        self.c, self.b = 1.0, 0.0
        self.eta = eta
        self._last_pre = 0.0
        self._last_y = 0.0

    def __call__(self, state: State) -> State:
        y = float(state.get_value(Y) or 0.0)
        enc = np.array([
            float(state.get_value(d)) if state.get_value(d) is not None else 0.0
            for d in self.encoder.d_enc
        ])
        pre = self.c * y + float(self.w @ enc) + self.b
        return State(values={M: float(np.tanh(pre))})

    def update(self, pred: float, y_next: float, y_used: float) -> None:
        err = y_next - pred
        self.c += self.eta * err * y_used
        self.b += self.eta * err


def g_schedule(protocol: str, steps: int) -> list:
    gs = [round(i / (steps - 1), 3) for i in range(steps)]
    if protocol == "updown":
        return gs + gs[::-1]
    if protocol == "downup":
        return gs[::-1] + gs
    if protocol == "flat0":
        return [0.0] * (2 * steps)
    raise ValueError(protocol)


def run(arm: str, protocol: str, seed: int, steps: int, per_step: int):
    rng = np.random.default_rng(seed)
    probes = [State(values={Y: v}) for v in PROBE_VALUES]
    encoder = BehavioralEncoder(probes, [M])
    self_model = AdaptiveSelfModel(encoder, twisted=(arm == "T"))
    obs = Observer("sweep", [Y], [X], WorldModel(),
                   self_model=self_model, self_encoder=encoder,
                   consumption_gain=0.0, log_capacity=per_step + 8)

    t_global = 0
    out = []
    for g in g_schedule(protocol, steps):
        obs.consumption_gain = g
        obs.clear_memory()
        preds, actuals = [], []
        for _ in range(per_step):
            # noise-dominated world: at g=0, y is near-white, so lag-1
            # internal correlation can only be built by consumption —
            # closure's correlation clause measures the loop, not the input
            x = float(0.3 * np.sin(0.5 * t_global) + 0.8 * rng.normal())
            t_global += 1
            pred = None
            if obs.internal_state is not None:
                y_used = float(obs.internal_state.get_value(Y) or 0.0)
                meta = self_model(
                    obs._augment_with_self_encoding(obs.internal_state))
                pred = float(meta.get_value(M) or 0.0)
            internal = obs.observe(State(values={X: x}))
            y_new = float(internal.get_value(Y) or 0.0)
            if pred is not None:
                preds.append(pred)
                actuals.append(y_new)
                self_model.update(pred, y_new, y_used)

        ca = obs.closure_assessment(lag=1, min_samples=20)
        ta = obs.twist_assessment(n_perturb=6, seed=seed)
        conscious = bool(ca.closed and ta.twisted)
        p, a = np.array(preds), np.array(actuals)
        pred_r = (float(np.corrcoef(p, a)[0, 1])
                  if len(p) > 3 and p.std() > 1e-9 and a.std() > 1e-9 else 0.0)
        out.append({
            "g": g, "ccorr": ca.corr_internal, "closed": int(ca.closed),
            "twisted": int(ta.twisted), "conscious": int(conscious),
            "pred_r": pred_r,
        })
    return out


def thresholds(curve: list, steps: int):
    """(g_on ascending, g_off descending) of is_conscious; None if never."""
    asc, desc = curve[:steps], curve[steps:]
    g_on = next((r["g"] for r in asc if r["conscious"]), None)
    g_off = next((r["g"] for r in desc if not r["conscious"]), None)
    return g_on, g_off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+",
                    default=["updown", "downup", "flat0"])
    ap.add_argument("--arms", nargs="+", default=["T", "U"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--per-step", type=int, default=300)
    args = ap.parse_args()

    print(f"#C1 real §5.5 sweep — arms={args.arms} seeds={args.seeds} "
          f"steps={args.steps} per_step={args.per_step}")
    for protocol in args.protocols:
        for arm in args.arms:
            curves = [run(arm, protocol, s, args.steps, args.per_step)
                      for s in range(args.seeds)]
            half = args.steps
            fields = ("g", "ccorr", "conscious", "pred_r")
            first = {f: np.mean([[c[i][f] for c in curves]
                                 for i in range(half)], axis=1)
                     for f in fields}
            second = {f: np.mean([[c[half + i][f] for c in curves]
                                  for i in range(half)], axis=1)
                      for f in fields}
            print(f"\n[{protocol} | arm {arm}]")
            print(f"{'g':>5} | {'ccorr_1':>7} {'ccorr_2':>7} | "
                  f"{'consc_1':>7} {'consc_2':>7} | {'predr_1':>7} {'predr_2':>7}")
            for i in range(half):
                j = half - 1 - i if protocol != "flat0" else i
                print(f"{first['g'][i]:>5.2f} | {first['ccorr'][i]:>7.3f} "
                      f"{second['ccorr'][j]:>7.3f} | {first['conscious'][i]:>7.2f} "
                      f"{second['conscious'][j]:>7.2f} | {first['pred_r'][i]:>7.3f} "
                      f"{second['pred_r'][j]:>7.3f}")
            if protocol != "flat0":
                area = float(np.mean(second["ccorr"][::-1] - first["ccorr"]))
                print(f"  hysteresis area (ccorr): {area:+.4f}")
                if arm == "T" or arm == "U":
                    ons, offs = [], []
                    for c in curves:
                        g_on, g_off = thresholds(c, half)
                        ons.append(g_on)
                        offs.append(g_off)
                    print(f"  is_conscious g_on (asc): {ons}   "
                          f"g_off (desc): {offs}")
            else:
                drift = float(np.mean(second["ccorr"]) - np.mean(first["ccorr"]))
                print(f"  flat drift (ccorr):      {drift:+.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
