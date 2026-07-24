"""melody.py — Markov melodies with an exact listener model (shared infra for #9).

Design (see 09_music_plan.md):
  * Alphabet: K=14 scale degrees (two diatonic octaves, indices 0..13).
    Tonic = 0 (and 7), dominant = 4, submediant = 5. No rests.
  * Grammar: first-order chain with step-interval logits (-|Δ|/step_scale),
    temperature τ, and NO self-transitions (the reel's event-based sampling
    collapses repeats, so the generative process never emits them).
  * Phrases: PHRASE_LEN=16 notes. Positions 0..3 come from a motif library
    (trajectory-stub cues with known identity). Position 14 is the dominant
    (position 13 excludes it so the transition always records); position 15
    resolves: tonic with p_authentic, submediant otherwise (deceptive cadence).
  * Listener model: the chain + cadence rule + no-repeat exclusions, with a
    uniform prior at position 0. Generation draws positions 0..3 from the
    motif schedule, which the listener does not know — valence is defined by
    LISTENER surprisal, i.e. the grammar-only observer this experiment
    baselines against.
  * Valence: v = z(-surprisal), normalizer frozen on the record stream.
  * Oracles: markov_dv() is the grammar-only forecast of future Δvalence by
    Monte-Carlo rollout of the listener model — the parametric ceiling.
    Perfect episodic recall scores r = 1.0 by construction (re-performances
    are note-identical, so surprisal sequences are too).

numpy only.
"""

from __future__ import annotations

import numpy as np

K = 14
TONIC, DOMINANT, SUBMEDIANT = 0, 4, 5
PHRASE_LEN = 16
CADENCE_POS = 14  # dominant here; CADENCE_POS+1 resolves


class MelodyModel:
    """The listener model: exact next-symbol distributions, position-aware."""

    def __init__(self, tau: float = 1.0, p_authentic: float = 0.7,
                 step_scale: float = 2.0) -> None:
        self.tau = tau
        self.p_auth = p_authentic
        d = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]).astype(float)
        logits = -d / step_scale
        np.fill_diagonal(logits, -np.inf)          # no repeats, ever
        self._logits = logits

    def next_dist(self, prev: int | None, pos: int) -> np.ndarray:
        """P(symbol at `pos` | symbol at pos-1 = `prev`)."""
        if pos == 0 or prev is None:
            return np.full(K, 1.0 / K)
        if pos == CADENCE_POS:
            p = np.zeros(K)
            p[DOMINANT] = 1.0
            return p
        if pos == CADENCE_POS + 1:
            p = np.zeros(K)
            p[TONIC] = self.p_auth
            p[SUBMEDIANT] = 1.0 - self.p_auth
            return p
        z = self._logits[prev] / self.tau
        if pos == CADENCE_POS - 1:                 # next note forces V: differ now
            z = z.copy()
            z[DOMINANT] = -np.inf
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def surprisal(self, phrase: np.ndarray) -> np.ndarray:
        """Per-note listener surprisal (nats) of a phrase."""
        s = np.empty(len(phrase))
        prev = None
        for pos, sym in enumerate(phrase):
            p = self.next_dist(prev, pos)
            s[pos] = -np.log(max(p[sym], 1e-12))
            prev = int(sym)
        return s

    def sample_from(self, prev: int | None, pos: int,
                    rng: np.random.Generator) -> int:
        return int(rng.choice(K, p=self.next_dist(prev, pos)))


def make_motifs(n_motifs: int, rng: np.random.Generator,
                model: MelodyModel | None = None) -> list[np.ndarray]:
    """Distinct 4-note incipits drawn from a low-temperature chain."""
    model = model or MelodyModel(tau=0.7)
    motifs: list[np.ndarray] = []
    seen: set[tuple] = set()
    while len(motifs) < n_motifs:
        first = int(rng.integers(1, K))            # avoid always-tonic openings
        seq = [first]
        for pos in range(1, 4):
            seq.append(model.sample_from(seq[-1], pos, rng))
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            motifs.append(np.array(seq, dtype=int))
    return motifs


def make_phrase(motif: np.ndarray, model: MelodyModel,
                rng: np.random.Generator) -> np.ndarray:
    """motif (positions 0..3) + chain continuation + cadence."""
    notes = list(motif)
    for pos in range(4, PHRASE_LEN):
        notes.append(model.sample_from(notes[-1], pos, rng))
    return np.array(notes, dtype=int)


class Songbook:
    """P phrases over a motif library, with frozen valence normalization.

    affect_amp > 0 adds one phrase-specific valence spike per phrase — a
    memory-only-knowable value event ("the moment in this song") at a random
    position in [affect_lo, affect_hi], random sign, amplitude affect_amp.
    A phrase-CONSTANT affect offset would cancel out of Δvalence (within-
    phrase difference), so episodic value must be an event, not a bias.
    The listener grammar knows nothing of the spikes; markov_dv() therefore
    remains the honest parametric ceiling.
    """

    def __init__(self, n_phrases: int, n_motifs: int, model: MelodyModel,
                 rng: np.random.Generator, affect_amp: float = 0.0,
                 affect_lo: int = 6, affect_hi: int = 12) -> None:
        self.model = model
        self.motifs = make_motifs(n_motifs, rng)
        self.motif_of = rng.integers(0, n_motifs, size=n_phrases)
        self.phrases = [make_phrase(self.motifs[m], model, rng)
                        for m in self.motif_of]
        s_all = np.concatenate([model.surprisal(ph) for ph in self.phrases])
        self.mu, self.sd = float(s_all.mean()), float(s_all.std() + 1e-9)
        self.valence = [self._z(model.surprisal(ph)) for ph in self.phrases]
        self.affect_pos = np.full(n_phrases, -1)
        self.affect_sign = np.zeros(n_phrases)
        if affect_amp > 0:
            for pid in range(n_phrases):
                pos = int(rng.integers(affect_lo, affect_hi + 1))
                sign = float(rng.choice([-1.0, 1.0]))
                self.valence[pid][pos] += sign * affect_amp
                self.affect_pos[pid] = pos
                self.affect_sign[pid] = sign

    def _z(self, surprisal: np.ndarray) -> np.ndarray:
        return -(surprisal - self.mu) / self.sd

    def markov_dv(self, phrase_id: int, pos: int, horizon: int,
                  rng: np.random.Generator, n_roll: int = 128) -> float:
        """Grammar-only forecast of mean(v[pos+1..pos+h]) - v[pos]."""
        phrase = self.phrases[phrase_id]
        v_now = self.valence[phrase_id][pos]
        h = min(horizon, PHRASE_LEN - 1 - pos)
        if h <= 0:
            return 0.0
        acc = 0.0
        for _ in range(n_roll):
            prev, tot = int(phrase[pos]), 0.0
            for p in range(pos + 1, pos + 1 + h):
                dist = self.model.next_dist(prev, p)
                sym = int(rng.choice(K, p=dist))
                tot += self._z(np.array([-np.log(max(dist[sym], 1e-12))]))[0]
                prev = sym
            acc += tot / h
        return acc / n_roll - v_now

    def realized_dv(self, phrase_id: int, pos: int, horizon: int) -> float:
        v = self.valence[phrase_id]
        h = min(horizon, PHRASE_LEN - 1 - pos)
        if h <= 0:
            return 0.0
        return float(v[pos + 1: pos + 1 + h].mean() - v[pos])
