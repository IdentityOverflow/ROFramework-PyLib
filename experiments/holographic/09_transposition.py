"""#9b — Transposition: values vs differences (experiment file 09; see 09_music_plan.md).

The RO framework's derived-DoF claim as a recall experiment: musical identity
lives in *differences* on the pitch DoF, with the key as reference frame. If
so, a memory that stores relative structure (intervals) should transfer to
transposed re-performances, while a memory storing absolute values (degrees)
should collapse.

Protocol
--------
Songbook phrases confined to degrees 0..9 (masked sampling) so a +4-degree
transposition stays inside the 14-symbol alphabet. Two reels record the SAME
songbook:

  ABS : one-hot of the scale degree (14 symbols), cue = last 4 notes
  REL : one-hot of the successive interval Δ ∈ [-9..9] (19 symbols, Δ=0
        never occurs), cue = last 3 intervals — the same 4-note window

Test conditions: in-key (identical re-performance) and transposed (+4 on
every note; REL streams are bit-identical by construction, ABS streams are
disjoint one-hots).

Metrics per (encoding × condition): phrase-identity accuracy, position
accuracy, next-note accuracy (REL predicts next interval, converted to a
note via the current absolute pitch). Foresight is deliberately not scored
(#9a showed the surprisal-valence currency is parametrically saturated).

Predictions (pre-registered):
  1. Transposed: REL retains in-key performance; ABS collapses toward 1/P.
  2. In-key: ABS >= REL — interval streams are self-similar under scale
     runs (+1,+1,... appears in many phrases), an ambiguity absolute
     encoding does not pay. If confirmed, that asymmetry is the
     invariance <-> absolute-information complementarity pair (ro_framework
     §7.2), measured.

Recorder note: interval streams contain legitimate consecutive repeats
(scale runs), which HoloEpisodicMemory.observe() would collapse via
event-based sampling; the recorder resets mem.last_sym before every record
so each interval event is written.

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
from melody import K, PHRASE_LEN, MelodyModel, make_motifs  # noqa: E402

MAX_DEG = 9          # melodies confined to 0..MAX_DEG
SHIFT = 4            # transposition in degrees
K_REL = 19           # interval alphabet: Δ+9 → 0..18 (index 9 = Δ0, unused)
WINDOW = 4           # notes of context for both encodings
HORIZON = 3


def one_hot(i: int, k: int) -> np.ndarray:
    v = np.zeros(k)
    v[i] = 1.0
    return v


def sample_phrase_masked(model: MelodyModel, motif: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Phrase generation with symbols > MAX_DEG masked out."""
    notes = list(motif)
    for pos in range(4, PHRASE_LEN):
        p = model.next_dist(notes[-1], pos).copy()
        p[MAX_DEG + 1:] = 0.0
        p /= p.sum()
        notes.append(int(rng.choice(K, p=p)))
    return np.array(notes, dtype=int)


def make_masked_motifs(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    motifs, seen = [], set()
    low = MelodyModel(tau=0.7)
    while len(motifs) < n:
        seq = [int(rng.integers(1, MAX_DEG + 1))]
        for pos in range(1, 4):
            p = low.next_dist(seq[-1], pos).copy()
            p[MAX_DEG + 1:] = 0.0
            p /= p.sum()
            seq.append(int(rng.choice(K, p=p)))
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            motifs.append(np.array(seq, dtype=int))
    return motifs


def intervals(phrase: np.ndarray) -> np.ndarray:
    return np.diff(phrase)                     # length 15, values in [-9..9]\{0}


def record_stream(mem: HoloEpisodicMemory, symbols: np.ndarray, k: int) -> None:
    for s in symbols:
        mem.last_sym = None                    # defeat event-collapse (see header)
        mem.observe(one_hot(int(s), k), tag=0.0)
    mem.boundary()


def build_reels(phrases: list[np.ndarray], seed: int):
    enc_abs = PhasorEncoder(obs_dim=K, n=512, scale=1.0, seed=seed)
    enc_rel = PhasorEncoder(obs_dim=K_REL, n=512, scale=1.0, seed=seed + 1)
    mem_abs = HoloEpisodicMemory(enc_abs, VQCodebook(np.eye(K), enc_abs),
                                 budget=PHRASE_LEN + 4)
    mem_rel = HoloEpisodicMemory(enc_rel, VQCodebook(np.eye(K_REL), enc_rel),
                                 budget=PHRASE_LEN + 4)
    for ph in phrases:
        record_stream(mem_abs, ph, K)
        record_stream(mem_rel, intervals(ph) + 9, K_REL)
    return mem_abs, mem_rel


def score(mem: HoloEpisodicMemory, encoding: str, test: list[np.ndarray],
          true_ids: list[int]) -> dict:
    """Query every position with a WINDOW-note context; top-1 scoring."""
    id_hit = pos_hit = next_hit = n = 0
    for pid, ph in zip(true_ids, test):
        iv = intervals(ph) + 9
        for t in range(WINDOW - 1, PHRASE_LEN - 1):
            if encoding == "abs":
                cue = [one_hot(int(ph[t - j]), K)
                       for j in range(WINDOW - 1, -1, -1)]
                want_pos = t
            else:
                # causal window: intervals BETWEEN notes t-3..t only —
                # iv[t] leads into note t+1 and must stay out of the cue
                cue = [one_hot(int(iv[t - 1 - j]), K_REL)
                       for j in range(WINDOW - 2, -1, -1)]
                want_pos = t - 1               # slide position of iv[t-1]
            res = mem.query(cue, horizon=HORIZON, top_m=1)
            if not res:
                continue
            m = res[0]
            n += 1
            ok_id = m["slide"] == pid
            id_hit += int(ok_id)
            pos_hit += int(ok_id and m["position"] == want_pos)
            if m["next_symbols"]:
                if encoding == "abs":
                    pred = m["next_symbols"][0]
                else:
                    pred = int(ph[t]) + (m["next_symbols"][0] - 9)
                next_hit += int(pred == ph[t + 1])
    return {"id": id_hit / max(n, 1), "pos": pos_hit / max(n, 1),
            "next": next_hit / max(n, 1), "n": n}


def run_config(tau: float, n_phrases: int, seed: int, n_motifs: int,
               test_phrases: int) -> dict:
    rng = np.random.default_rng(seed)
    model = MelodyModel(tau=tau)
    motifs = make_masked_motifs(n_motifs, rng)
    phrases = [sample_phrase_masked(model, motifs[int(m)], rng)
               for m in rng.integers(0, n_motifs, size=n_phrases)]
    mem_abs, mem_rel = build_reels(phrases, seed)

    ids = list(rng.permutation(n_phrases)[:min(test_phrases, n_phrases)])
    inkey = [phrases[i] for i in ids]
    trans = [phrases[i] + SHIFT for i in ids]

    return {
        "abs_in": score(mem_abs, "abs", inkey, ids),
        "rel_in": score(mem_rel, "rel", inkey, ids),
        "abs_tr": score(mem_abs, "abs", trans, ids),
        "rel_tr": score(mem_rel, "rel", trans, ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.5])
    ap.add_argument("--phrases", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--motifs", type=int, default=8)
    ap.add_argument("--test-phrases", type=int, default=32)
    args = ap.parse_args()

    print(f"#9b transposition — shift=+{SHIFT}, window={WINDOW} notes, "
          f"taus={args.taus} phrases={args.phrases} seeds={args.seeds}")
    hdr = (f"{'tau':>4} {'P':>4} |{'— in-key id% —':^17}|{'— transposed id% —':^19}|"
           f"{'— transposed next —':^20}")
    sub = (f"{'':>4} {'':>4} |{'ABS':>8} {'REL':>8} |{'ABS':>9} {'REL':>9} |"
           f"{'ABS':>9} {'REL':>10}")
    print(hdr)
    print(sub)
    print("-" * len(hdr))
    for tau in args.taus:
        for P in args.phrases:
            rows = [run_config(tau, P, s, args.motifs, args.test_phrases)
                    for s in range(args.seeds)]
            def agg(cond, key):
                return float(np.mean([r[cond][key] for r in rows]))
            print(f"{tau:>4.1f} {P:>4} |{100*agg('abs_in','id'):>7.1f}% "
                  f"{100*agg('rel_in','id'):>7.1f}% |{100*agg('abs_tr','id'):>8.1f}% "
                  f"{100*agg('rel_tr','id'):>8.1f}% |{agg('abs_tr','next'):>9.3f} "
                  f"{agg('rel_tr','next'):>10.3f}   (chance id={100/P:.1f}%)",
                  flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal {time.time() - t0:.1f}s")
