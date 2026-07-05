"""
Embodied Episodic Memory — the reel meets the world.

Couples the slide-based holographic memory (experiments/embodied/holo_memory.py,
built from #3's film and #4's nonlinear decider) to the embodied environment,
headless. The agent is a scripted Braitenberg explorer (steer to food, avoid
danger, wander); the memory watches its observations and records symbol
transitions into bounded slides.

Pipeline refinements forced by first contact with real data (v1 diagnosis):
  • PCA-whitening before VQ/encoding — raw obs space is a bad metric (242
    sparse vision dims dominate; prototypes end up σ-equiv 1.26 apart, i.e.
    neighboring symbols look like noisy copies of each other in #4-B terms).
  • Top-M vote over retrieved instances — the film retrieves single past
    moments; natural dynamics are stochastic, so fair episodic prediction
    aggregates several retrieved continuations (instance-based learning,
    still no global transition table).
  • Valence forecast as Δ, not level — valence decays at 0.001/step, so
    level-persistence is unbeatable at short horizons (r≈1.0); the useful
    signal is predicting CHANGE (eating/danger spikes), where persistence
    predicts zero by construction.

Protocol (three runs):
  1. PRETRAIN  — explore; fit whitener + VQ codebook; set phasor scale so
     median quantization error lands at σ_eff ≈ 1.2 (#4-B tolerance ≈ 2).
  2. MEMORY    — explore with codebook frozen; record the reel.
  3. TRANSFER  — explore a DIFFERENT world layout (new seed, same physics);
     never recorded.

Questions asked of the reel:
  FIDELITY  — do SIC-decoded slides match what was recorded? (#4-A on
     natural, bursty symbol statistics)
  RETRIEVAL — does the top match land on the cue's own symbol (precision),
     and in replay, on the exact recorded moment?
  REPLAY / TRANSFER — next-symbol prediction vs marginal + Markov-1
     baselines (Markov = a purpose-built global transition table over the
     same data = semantic memory; the reel is episodic instance retrieval).
  VALENCE Δ — does retrieved next-tag change forecast actual near-future
     valence change? ('this smelled good/bad last time')

Requires: numpy + experiments/embodied (env.py, holo_memory.py).
Runtime: ~1-2 min on CPU, headless (no pygame needed).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embodied"))

from env import World                                    # noqa: E402
from holo_memory import (                                # noqa: E402
    HoloEpisodicMemory, PhasorEncoder, VQCodebook, Whitener)

N_PHASOR = 512
K_PROTO = 64
WHITEN_D = 48
BUDGET = 120
HORIZON = 3
TOP_M = 5
CONTEXT = 5      # cue = trajectory stub of the last 5 transition features


# ── Scripted explorer ──────────────────────────────────────────────────────────


def policy(obs, rng, state):
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


def run_world(seed, n_steps, on_frame):
    """Headless run; calls on_frame(obs, valence, died) each step."""
    world = World(seed=seed)
    world.player_active = False
    world.add_agent()
    rng = np.random.default_rng(seed + 1000)
    state = {"ou": 0.0}
    prev_life = 1.0
    for _ in range(n_steps):
        obs = world.get_ai_observation()
        life = float(obs[260])
        died = life > prev_life + 0.5      # respawn = life jumped back up
        prev_life = life
        valence = float(world.ai.meters.valence)
        on_frame(obs, valence, died)
        world.step(ai_action=policy(obs, rng, state))


# ── Main ───────────────────────────────────────────────────────────────────────


def main(seed=0, pretrain_steps=6000, memory_steps=20000, transfer_steps=8000):
    t0 = time.time()
    print("=" * 70)
    print("Embodied Episodic Memory — the reel meets the world")
    print("=" * 70)

    # 1. PRETRAIN: whitener + codebook + phasor scale
    samples = []
    run_world(seed, pretrain_steps,
              lambda obs, v, d: samples.append(obs.copy()))
    samples = np.array(samples[::3])
    whiten = Whitener(samples, d=WHITEN_D)
    feats = whiten.transform(samples)
    enc_probe = PhasorEncoder(WHITEN_D, N_PHASOR, scale=1.0, seed=seed)
    cb_probe = VQCodebook.fit(feats, K_PROTO, enc_probe, seed=seed)
    qdists = np.array([cb_probe.quantize(f)[1] for f in feats])
    scale = 1.2 / np.median(qdists)
    encoder = PhasorEncoder(WHITEN_D, N_PHASOR, scale=scale, seed=seed)
    codebook = VQCodebook(cb_probe.prototypes, encoder)

    pd = codebook.prototypes
    proto_d = np.sqrt(((pd[:, None] - pd[None]) ** 2).sum(-1))
    np.fill_diagonal(proto_d, np.inf)
    print(f"\n  PCA to d={WHITEN_D} (no whitening); codebook K={K_PROTO} "
          f"from {len(samples)} obs")
    print(f"  phasor scale s={scale:.3f}  →  σ_eff: median "
          f"{scale * np.median(qdists):.2f}, p90 "
          f"{scale * np.percentile(qdists, 90):.2f}   (#4-B tolerance ≈ 2)")
    print(f"  nearest-prototype separation: median σ-equiv "
          f"{scale * np.median(proto_d.min(1)):.2f}")

    # 2. MEMORY run: record the reel + ground-truth log
    memory = HoloEpisodicMemory(encoder, codebook, budget=BUDGET)
    log = {"sym": [], "val": [], "feat": [], "slide": [], "pos": []}

    def record_frame(obs, valence, died):
        if died:
            memory.boundary()
        feat = whiten.transform(obs)
        sym = memory.observe(feat, tag=valence)
        if sym is not None:
            log["sym"].append(sym)
            log["val"].append(valence)
            log["feat"].append(feat)
            log["slide"].append(len(memory.slides) - 1)
            log["pos"].append(memory.slides[-1].count - 1)

    run_world(seed, memory_steps, record_frame)
    print(f"\n  memory run: {memory_steps} steps → {memory.n_records} records "
          f"(symbol transitions) in {len(memory.slides)} slides")

    # FIDELITY: decoded slides vs ground truth
    agree = total = 0
    for si, slide in enumerate(memory.slides):
        if slide.count == 0:
            continue
        dec = memory._decode_slide(slide, force_sic=True)
        truth = [log["sym"][i] for i in range(len(log["sym"]))
                 if log["slide"][i] == si]
        agree += sum(int(a == b) for a, b in zip(dec, truth))
        total += slide.count
    print(f"  FIDELITY: SIC decode matches recorded symbol at "
          f"{agree / total:.1%} of {total} positions")

    # Baselines from the memory run's ground truth
    syms = np.array(log["sym"])
    marginal = int(np.bincount(syms, minlength=K_PROTO).argmax())
    trans = np.zeros((K_PROTO, K_PROTO))
    for a, b in zip(syms[:-1], syms[1:]):
        trans[a, b] += 1
    markov = trans.argmax(1)
    markov[trans.sum(1) == 0] = marginal

    def eval_queries(cue_feat, cue_sym, next_true, dval_true, true_where=None):
        """Query the reel per cue; vote across top-M retrieved instances."""
        hits = {"film": 0, "markov": 0, "marginal": 0}
        prec = exact = 0
        dv_pred, dv_true = [], []
        n = 0
        for i in range(len(cue_feat)):
            matches = memory.query(cue_feat[i], horizon=HORIZON, top_m=TOP_M)
            matches = [m for m in matches if m["next_symbols"]]
            if not matches:
                continue
            votes = {}
            dv_w = w_sum = 0.0
            for m in matches:
                votes[m["next_symbols"][0]] = (
                    votes.get(m["next_symbols"][0], 0.0) + m["score"])
                dv_w += m["score"] * (np.mean(m["next_tags"])
                                      - m["matched_tag"])
                w_sum += m["score"]
            pred = max(votes, key=votes.get)
            hits["film"] += pred == next_true[i]
            hits["markov"] += markov[cue_sym[i]] == next_true[i]
            hits["marginal"] += marginal == next_true[i]
            prec += matches[0]["matched_symbol"] == cue_sym[i]
            if true_where is not None:
                exact += ((matches[0]["slide"], matches[0]["position"])
                          == true_where[i])
            dv_pred.append(dv_w / w_sum)
            dv_true.append(dval_true[i])
            n += 1
        r_dv = (np.corrcoef(dv_pred, dv_true)[0, 1]
                if len(dv_pred) > 2 and np.std(dv_pred) > 0 else 0.0)
        return n, {k: v / n for k, v in hits.items()}, prec / n, exact / n, r_dv

    # 3a. REPLAY queries (from the recorded run itself)
    rng = np.random.default_rng(seed + 7)
    idx = CONTEXT - 1 + rng.choice(
        len(syms) - HORIZON - CONTEXT,
        size=min(300, len(syms) // 3), replace=False)
    n, hits, prec, exact, r_dv = eval_queries(
        [np.stack(log["feat"][i - CONTEXT + 1: i + 1]) for i in idx],
        [log["sym"][i] for i in idx],
        [log["sym"][i + 1] for i in idx],
        [float(np.mean(log["val"][i + 1: i + 1 + HORIZON]) - log["val"][i])
         for i in idx],
        true_where=[(log["slide"][i], log["pos"][i]) for i in idx],
    )
    print(f"\n  REPLAY ({n} queries — cue is a raw obs the reel has seen):")
    print(f"    retrieval: top-1 symbol precision {prec:.1%}, "
          f"exact moment {exact:.1%}")
    print(f"    next-symbol: film {hits['film']:.1%}   "
          f"markov {hits['markov']:.1%}   marginal {hits['marginal']:.1%}")
    print(f"    Δvalence forecast r: film {r_dv:+.2f}   (persistence: 0 by "
          f"construction)")

    # 3b. TRANSFER queries (new world layout, never recorded)
    tlog = {"sym": [], "val": [], "feat": []}
    last = [None]

    def transfer_frame(obs, valence, died):
        feat = whiten.transform(obs)
        sym, _ = codebook.quantize(feat)
        if sym != last[0]:
            tlog["sym"].append(sym)
            tlog["val"].append(valence)
            tlog["feat"].append(feat)
            last[0] = sym

    run_world(seed + 999, transfer_steps, transfer_frame)
    tsyms = tlog["sym"]
    idx = CONTEXT - 1 + rng.choice(
        len(tsyms) - HORIZON - CONTEXT,
        size=min(300, len(tsyms) // 3), replace=False)
    n, hits, prec, _, r_dv = eval_queries(
        [np.stack(tlog["feat"][i - CONTEXT + 1: i + 1]) for i in idx],
        [tsyms[i] for i in idx],
        [tsyms[i + 1] for i in idx],
        [float(np.mean(tlog["val"][i + 1: i + 1 + HORIZON]) - tlog["val"][i])
         for i in idx],
    )
    print(f"\n  TRANSFER ({n} queries — new world layout, seed {seed + 999}):")
    print(f"    retrieval: top-1 symbol precision {prec:.1%}")
    print(f"    next-symbol: film {hits['film']:.1%}   "
          f"markov {hits['markov']:.1%}   marginal {hits['marginal']:.1%}")
    print(f"    Δvalence forecast r: film {r_dv:+.2f}")

    print(f"\n  total runtime {time.time() - t0:.0f}s")
    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  • FIDELITY = #4-A's SIC recall on natural, bursty statistics.")
    print("  • REPLAY film vs markov = one superposed phasor vector per episode")
    print("    against a purpose-built transition table over the same data.")
    print("  • TRANSFER above marginal = 'what happened next last time' carries")
    print("    to situations never recorded — content-addressed prediction the")
    print("    fading echo cannot give (it forgets in ~16 steps; the reel never).")
    print("  • Δvalence r > 0 = the tags ride back with recall — the hook for")
    print("    RPE-gated behavior ('this smelled good/bad last time').")
    print("\nDone.")


if __name__ == "__main__":
    main()
