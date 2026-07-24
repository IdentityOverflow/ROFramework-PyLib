"""#9c — The mounted listener: episodic control in melody space (file 12).

The plan's 9c, absorbing the action-side-tags + recollection-as-suggestion
experiment queued since #8 — in the one setting where side-tags come free:
for a listener, the "action" IS the next symbol, so the film's stored
continuation already carries policy-relevant content. #8 proved value-only
tags cannot steer; here the reel's content is exactly what steering needs.

Agent: a numpy ESN (fixed random reservoir, leaky units) with an online
softmax readout predicting the next-symbol distribution — the parametric
learner. The reel mounts beside it, recording the same stream (valence =
z(-surprisal) + phrase-affect spikes, amp 1.0).

World: songbook at temperature tau with an exposure schedule —
  COMMON songs: many training performances (parametrically learnable)
  RARE songs:   exactly 2 training performances (episodic-only territory)
Test: every song re-performed once, readout learning frozen.

Arms (2 training modes x 2 test read policies):
  base : ESN alone
  sugg : + suggestion — when the identity gate fires (margin_slide >= 0.01),
         the recalled continuation OVERRIDES the readout (the Gate result:
         fired recalls are ~99% right, so blend = full override); ungated
         queries fall through to the ESN untouched (dual read policy).
  dv   : + gated dv̂ consumed during TRAINING as learning-rate modulation
         eta_t = eta * (1 + beta * clip(dv̂, -1, 1)) — RPE-like plasticity
         gating, agent-side only (dv̂ = 0 when the gate is closed; no oracle
         access). The value channel, given its fairest shot post-#8.
  both : dv training + sugg test policy.

Pre-registered predictions:
  1. The dissociation: sugg lifts RARE accuracy far above base (toward the
     reel's recall ceiling) while COMMON accuracy is unchanged-to-slightly-up.
     Episodic memory owns what the parametric learner cannot: 2-exposure
     material.
  2. Suggestion hit-rate when fired stays ~0.9+ (the gate's precision
     transfers from 9a-harness to the mounted setting).
  3. dv changes little on any accuracy metric (the value channel modulates
     HOW MUCH is learned, not WHAT; #7/#8 lineage) — included to close the
     question fairly rather than by assumption.

numpy only; CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embodied"))

from holo_memory import HoloEpisodicMemory, PhasorEncoder, VQCodebook  # noqa: E402
from melody import K, PHRASE_LEN, MelodyModel, Songbook  # noqa: E402

CUE_LEN = 3
THETA = 0.01
BETA = 0.5
AFFECT_AMP = 1.0
N_COMMON, N_RARE = 8, 8
COMMON_REPS, RARE_REPS = 12, 2


def one_hot(sym: int) -> np.ndarray:
    v = np.zeros(K)
    v[sym] = 1.0
    return v


def query_gated(mem: HoloEpisodicMemory, cue: list[np.ndarray],
                horizon: int = 4, top_m: int = 8):
    """Top match + margin_agree: lead over the nearest DISAGREEING candidate.

    Discovered by this experiment's smoke test: with repeated exposures the
    reel holds duplicate slides of the same song, so margin_slide reads
    exact ties everywhere and the gate never fires — but duplicates AGREE,
    and agreement is not ambiguity. Confidence for value/policy import is
    the lead over the best candidate that predicts something DIFFERENT
    (another continuation symbol, or conflicting tags). Candidate for
    promotion into holo_memory.query() if it survives this experiment.
    """
    res = mem.query(cue, horizon=horizon, top_m=top_m)
    if not res or not res[0]["next_symbols"]:
        return None, 0.0
    top = res[0]
    top_tag = float(np.mean(top["next_tags"])) if top["next_tags"] else 0.0
    margin_agree = 1.0
    for cand in res[1:]:
        if not cand["next_symbols"]:
            continue
        tag = float(np.mean(cand["next_tags"])) if cand["next_tags"] else 0.0
        disagrees = (cand["next_symbols"][0] != top["next_symbols"][0]
                     or abs(tag - top_tag) > 0.5)
        if disagrees:
            margin_agree = (top["score"] - cand["score"]) / max(top["score"], 1e-9)
            break
    return top, float(margin_agree)


class ESN:
    """Leaky echo-state network with an online softmax readout."""

    def __init__(self, n_res: int, seed: int, leak: float = 0.3,
                 rho: float = 0.9, eta: float = 0.05) -> None:
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1, (n_res, n_res))
        W *= rho / max(abs(np.linalg.eigvals(W)).max(), 1e-9)
        self.W = W
        self.W_in = rng.normal(0, 1, (n_res, K)) * 0.5
        self.W_out = np.zeros((K, n_res))
        self.leak, self.eta = leak, eta
        self.state = np.zeros(n_res)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, x: np.ndarray) -> None:
        pre = self.W @ self.state + self.W_in @ x
        self.state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)

    def predict(self) -> np.ndarray:
        z = self.W_out @ self.state
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def learn(self, target: int, eta_scale: float = 1.0) -> None:
        p = self.predict()
        g = -p
        g[target] += 1.0
        self.W_out += (self.eta * eta_scale) * np.outer(g, self.state)


def make_schedule(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    common = list(range(N_COMMON))
    rare = list(range(N_COMMON, N_COMMON + N_RARE))
    train = common * COMMON_REPS + rare * RARE_REPS
    rng.shuffle(train)
    return train, common, rare


def run_seed(tau: float, seed: int, n_motifs: int, use_dv: bool) -> dict:
    rng = np.random.default_rng(seed)
    model = MelodyModel(tau=tau)
    book = Songbook(N_COMMON + N_RARE, n_motifs, model, rng,
                    affect_amp=AFFECT_AMP)
    train, common, rare = make_schedule(rng)

    esn = ESN(256, seed)
    encoder = PhasorEncoder(obs_dim=K, n=512, scale=1.0, seed=seed)
    mem = HoloEpisodicMemory(encoder, VQCodebook(np.eye(K), encoder),
                             budget=PHRASE_LEN + 4)

    # ── training: predict-then-learn, reel records alongside ──
    for pid in train:
        phrase, val = book.phrases[pid], book.valence[pid]
        esn.reset()
        recent: list[int] = []
        for pos in range(PHRASE_LEN):
            sym = int(phrase[pos])
            if pos > 0:
                eta_scale = 1.0
                if use_dv and len(recent) >= CUE_LEN:
                    cue = [one_hot(s) for s in recent[-CUE_LEN:]]
                    top, magree = query_gated(mem, cue)
                    if top is not None and magree >= THETA and top["next_tags"]:
                        dv = float(np.mean(top["next_tags"])
                                   - top["matched_tag"])
                        eta_scale = 1.0 + BETA * float(np.clip(dv, -1, 1))
                esn.learn(sym, eta_scale)
            mem.observe(one_hot(sym), tag=float(val[pos]))
            esn.step(one_hot(sym))
            recent.append(sym)
        mem.boundary()

    # ── test: every song once, learning frozen, both read policies ──
    stats = {arm: {"c": [0, 0], "r": [0, 0]} for arm in ("base", "sugg")}
    fired = fired_hit = 0
    dv_hat, dv_real = [], []
    for pid in common + rare:
        phrase = book.phrases[pid]
        bucket = "c" if pid in common else "r"
        esn.reset()
        for pos in range(PHRASE_LEN - 1):
            sym = int(phrase[pos])
            esn.step(one_hot(sym))
            truth = int(phrase[pos + 1])
            base_pred = int(np.argmax(esn.predict()))
            sugg_pred = base_pred
            if pos >= CUE_LEN - 1:
                cue = [one_hot(int(phrase[pos - j]))
                       for j in range(CUE_LEN - 1, -1, -1)]
                m, magree = query_gated(mem, cue)
                if m is not None and magree >= THETA:
                    sugg_pred = int(m["next_symbols"][0])
                    fired += 1
                    fired_hit += int(sugg_pred == truth)
                    if m["next_tags"]:
                        dv_hat.append(float(np.mean(m["next_tags"])
                                            - m["matched_tag"]))
                        dv_real.append(book.realized_dv(pid, pos, 4))
            stats["base"][bucket][0] += int(base_pred == truth)
            stats["base"][bucket][1] += 1
            stats["sugg"][bucket][0] += int(sugg_pred == truth)
            stats["sugg"][bucket][1] += 1

    def acc(arm, b):
        h, n = stats[arm][b]
        return h / max(n, 1)

    r_dv = float("nan")
    if len(dv_hat) >= 3 and np.std(dv_hat) > 1e-9 and np.std(dv_real) > 1e-9:
        r_dv = float(np.corrcoef(dv_hat, dv_real)[0, 1])

    return {
        "base_c": acc("base", "c"), "base_r": acc("base", "r"),
        "sugg_c": acc("sugg", "c"), "sugg_r": acc("sugg", "r"),
        "fire": fired / max(sum(stats["base"][b][1] for b in "cr"), 1),
        "hit": fired_hit / max(fired, 1),
        "r_dv": r_dv,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.8, 2.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--motifs", type=int, default=8)
    args = ap.parse_args()

    print(f"#9c mounted listener — taus={args.taus} seeds={args.seeds} "
          f"common={N_COMMON}x{COMMON_REPS} rare={N_RARE}x{RARE_REPS} "
          f"theta={THETA} beta={BETA} affect={AFFECT_AMP}")
    hdr = (f"{'tau':>4} {'train':>6} | {'common: base':>12} {'sugg':>6} | "
           f"{'rare: base':>10} {'sugg':>6} | {'fire%':>5} {'hit':>5} {'r_dv':>6}")
    print(hdr)
    print("-" * len(hdr))
    for tau in args.taus:
        for use_dv in (False, True):
            rows = [run_seed(tau, s, args.motifs, use_dv)
                    for s in range(args.seeds)]
            def agg(k):
                return float(np.nanmean([r[k] for r in rows]))
            label = "dv" if use_dv else "plain"
            print(f"{tau:>4.1f} {label:>6} | {agg('base_c'):>12.3f} "
                  f"{agg('sugg_c'):>6.3f} | {agg('base_r'):>10.3f} "
                  f"{agg('sugg_r'):>6.3f} | {100*agg('fire'):>4.1f}% "
                  f"{agg('hit'):>5.3f} {agg('r_dv'):>6.3f}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
