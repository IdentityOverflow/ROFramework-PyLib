"""
holo_memory.py — Slide-based holographic episodic memory for embodied agents.

Implements the architecture earned by holographic experiments #3 and #4
(see experiments/holographic/findings.md) as a drop-in memory for embodied
brains, with one structural addition: SLIDES.

A single film has a hard load threshold (~0.3N bindings for clean SIC recall,
#4-A) and a periodic clock (timestamps alias after ~N steps). Instead of
fighting either limit, the long-term store is a *reel* of bounded slides:
each slide is one film recording one episode; when the slide reaches its
budget — or an event boundary (death, valence spike) closes it — a fresh
slide starts. Within-slide time is fine-grained (clock phases); across-slide
time is ordinal (slide index). Capacity grows linearly with the reel and
episodes never crosstalk.

Pipeline:

    obs (263,) --[VQ codebook]--> symbol id --[phasor code ⊙ clock]--> slide
    obs (263,) --[phasor encoder]--> cue --> recognition scan over the reel
                                             --> (slide, position, what came next)

The phasor encoder is a fixed random projection into phase space:
    z(x) = exp(i · s · W @ x)
so for two observations u, v the per-dimension phase difference is
N(0, s²·‖u−v‖²) — exactly the phase-noise model validated in #4-B. A cue's
effective noise is σ_eff = s·‖obs − prototype‖: quantization error IS cue
noise, and #4-B's tolerance (σ ≲ 2) becomes a concrete quantization budget.

Recall is nonlinear throughout (the #4 decider):
  • slide readout uses SIC (decode most confident position, subtract, repeat);
  • episodic query projects each stored position onto the codebook and scores
    the CLEAN candidate against the raw cue.

Side metadata: each record keeps a scalar tag (e.g. valence) in a plain
array aligned with the slide — the film is the content-addressable index,
the tags are what an RL brain wants back (was this moment good?).

Requires: numpy only.
"""

from __future__ import annotations

import numpy as np


# ── Feature preprocessing ──────────────────────────────────────────────────────


class Whitener:
    """Top-d PCA fit on exploration samples (optionally whitened).

    Keeps the top principal directions and drops sparse noise dimensions.
    Full whitening (dividing by sqrt eigenvalues) is OFF by default: it
    amplifies low-variance jitter directions, which packs VQ prototypes
    tighter relative to frame-to-frame noise and hurts decode margins
    (#5 v2 diagnosis).
    """

    def __init__(self, samples: np.ndarray, d: int = 48,
                 whiten: bool = False, eps: float = 1e-6) -> None:
        self.mean = samples.mean(0)
        cov = np.cov((samples - self.mean).T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:d]
        self.proj = vecs[:, order]
        if whiten:
            self.proj = self.proj / np.sqrt(vals[order] + eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.proj


# ── Phasor encoding ────────────────────────────────────────────────────────────


class PhasorEncoder:
    """Fixed random projection of real vectors into unit phasors."""

    def __init__(self, obs_dim: int, n: int = 512, scale: float = 1.0,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n = n
        self.scale = scale
        self.W = rng.normal(0.0, 1.0, (n, obs_dim))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.scale * (self.W @ x))


class VQCodebook:
    """K prototype observations + their phasor codes (the symbol attractors)."""

    def __init__(self, prototypes: np.ndarray, encoder: PhasorEncoder) -> None:
        self.prototypes = prototypes                      # (K, obs_dim)
        self.phasors = np.stack([encoder.encode(p) for p in prototypes])

    @classmethod
    def fit(cls, samples: np.ndarray, k: int, encoder: PhasorEncoder,
            iters: int = 25, seed: int = 0) -> "VQCodebook":
        """Plain k-means on collected observations."""
        rng = np.random.default_rng(seed)
        protos = samples[rng.choice(len(samples), k, replace=False)].copy()
        for _ in range(iters):
            d = ((samples[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            for j in range(k):
                m = assign == j
                if m.any():
                    protos[j] = samples[m].mean(0)
        return cls(protos, encoder)

    def quantize(self, x: np.ndarray) -> tuple[int, float]:
        d = ((self.prototypes - x) ** 2).sum(-1)
        j = int(d.argmin())
        return j, float(np.sqrt(d[j]))


# ── Slide (one episode = one film) ─────────────────────────────────────────────


class Slide:
    def __init__(self, n: int) -> None:
        self.film = np.zeros(n, dtype=complex)
        self.tags: list[float] = []
        self.taught: list[int] = []    # 1 = recorded under teacher forcing
        self.written: list[int] = []   # session-known symbols (working memory)
        self.count = 0
        self.closed = False
        self._decoded: np.ndarray | None = None

    def record(self, sym: int, phasor: np.ndarray, omega: np.ndarray,
               tag: float, taught: bool = False) -> None:
        self.film += phasor * np.exp(-1j * omega * self.count)
        self.tags.append(tag)
        self.taught.append(int(taught))
        self.written.append(int(sym))
        self.count += 1
        self._decoded = None


# ── Episodic memory (the reel) ─────────────────────────────────────────────────


class HoloEpisodicMemory:
    """A reel of bounded holographic slides with nonlinear (SIC) recall.

    Parameters
    ----------
    encoder, codebook : the phasor encoder and frozen VQ codebook.
    budget : records per slide; keep under the #4-A load threshold (~0.3N
        for a 32-symbol codebook; default is conservative for 64 symbols).
    """

    def __init__(self, encoder: PhasorEncoder, codebook: VQCodebook,
                 budget: int = 120, min_dwell: int = 1,
                 demo_weight: float = 1.0) -> None:
        self.encoder = encoder
        self.codebook = codebook
        self.budget = budget
        self.min_dwell = min_dwell
        # Recall-score multiplier for records written under demonstration
        # (teacher forcing): "what did the teacher do here" outranks "what
        # did I stumble into last time". 1.0 = no preference.
        self.demo_weight = demo_weight
        self.n = encoder.n
        self.omega = 2 * np.pi * np.arange(1, self.n + 1) / self.n
        self.slides: list[Slide] = [Slide(self.n)]
        self.last_sym: int | None = None
        self._cand_sym: int | None = None
        self._cand_dwell = 0

    # ── recording ──────────────────────────────────────────────────────────

    def observe(self, obs: np.ndarray, tag: float = 0.0,
                taught: bool = False) -> int | None:
        """Record obs if its symbol differs from the last recorded one.

        Event-based sampling: consecutive same-symbol frames collapse into
        one record, so slide time advances only when the world (as seen
        through the codebook) actually changes. Returns the recorded symbol
        id, or None if nothing was recorded.
        """
        sym, _ = self.codebook.quantize(obs)
        if sym == self.last_sym:
            self._cand_sym, self._cand_dwell = None, 0
            return None
        # dwell gate: a new symbol must persist min_dwell consecutive calls
        # before it counts as a transition (rejects quantization flicker)
        if self.min_dwell > 1:
            if sym == self._cand_sym:
                self._cand_dwell += 1
            else:
                self._cand_sym, self._cand_dwell = sym, 1
            if self._cand_dwell < self.min_dwell:
                return None
            self._cand_sym, self._cand_dwell = None, 0
        slide = self.slides[-1]
        if slide.closed or slide.count >= self.budget:
            self.boundary()
            slide = self.slides[-1]
        slide.record(sym, self.codebook.phasors[sym], self.omega, tag,
                     taught=taught)
        self.last_sym = sym
        return sym

    def boundary(self) -> None:
        """Close the current slide (event boundary: death, surprise, budget)."""
        if self.slides[-1].count > 0:
            self.slides[-1].closed = True
            self.slides.append(Slide(self.n))
        self.last_sym = None    # next symbol always opens the new slide

    # ── slide readout (SIC, from #4-A) ─────────────────────────────────────

    def _decode_slide(self, slide: Slide, force_sic: bool = False) -> np.ndarray:
        """Read a slide's symbol sequence.

        Closed slides are read holographically (SIC through the film, decoded
        once and cached — the past exists only in the medium). The ACTIVE
        slide is the present episode: the brain knows what it just wrote
        (working memory), so its session-known symbols are returned directly
        unless force_sic is set — re-running full SIC on every new record
        would be O(count²) per step for information the writer already has.
        """
        if not slide.closed and not force_sic and slide.written:
            return np.asarray(slide.written, dtype=int)
        if slide._decoded is not None and not force_sic:
            return slide._decoded
        cb = self.codebook.phasors
        t_steps = slide.count
        residual = slide.film.copy()
        decoded = np.full(t_steps, -1, dtype=int)
        remaining = list(range(t_steps))
        batch = max(1, int(t_steps * 0.05))
        while remaining:
            scored = []
            for k in remaining:
                m = np.abs(cb.conj() @ (residual * np.exp(1j * self.omega * k)))
                top2 = np.argpartition(m, -2)[-2:]
                best = top2[np.argmax(m[top2])]
                scored.append((m[best] - min(m[top2]), k, int(best)))
            scored.sort(reverse=True)
            for _, k, sym in scored[:batch]:
                decoded[k] = sym
                residual -= cb[sym] * np.exp(-1j * self.omega * k)
                remaining.remove(k)
        slide._decoded = decoded
        return decoded

    # ── episodic query (recognition scan, from #4-B) ───────────────────────

    def _scan_index(self):
        """Per-slide clean phasor matrices.

        Closed slides are immutable: their phasor stacks are cached forever.
        Only the active slide's stack is rebuilt per call.
        """
        if not hasattr(self, "_closed_cache"):
            self._closed_cache = {}
        index = []
        for si, slide in enumerate(self.slides):
            if slide.count < 2:
                continue
            if slide.closed:
                if si not in self._closed_cache:
                    self._closed_cache[si] = (
                        self.codebook.phasors[self._decode_slide(slide)])
                index.append((si, self._closed_cache[si]))
            else:
                index.append((si, self.codebook.phasors[
                    np.asarray(slide.written, dtype=int)]))
        return index

    def query(self, recent, horizon: int = 3, top_m: int = 1):
        """'When was I in a situation like this, and what happened next?'

        ``recent`` is either one feature vector or a short trajectory stub
        [oldest, ..., newest]. Context matters: all occurrences of one symbol
        have IDENTICAL clean phasors, so a single-moment cue can only find
        *a* time the symbol occurred — trajectory cues disambiguate WHICH
        time by aligning the whole window against each stored position.

        Every stored position with a successor is scored by the summed match
        of the (raw, unquantized) cue window against the SIC-cleaned phasors
        ending at that position. Returns the top_m matches, best first, each
        carrying its recalled continuation and side tags. Returns [] if the
        reel is empty.
        """
        recent = np.asarray(recent)
        if recent.ndim == 1:
            recent = recent[None, :]
        cues = np.stack([self.encoder.encode(f) for f in recent])   # (L, n)
        L = len(cues)
        scored = []
        for si, phasors in self._scan_index():
            count = len(phasors)
            if count < L + 1:
                continue
            m = np.abs(phasors.conj() @ cues.T)          # (count, L)
            # combined[k] = sum_j m[k-(L-1)+j, j] for window ends
            # k = L-1 .. count-2 (each needs a successor)
            combined = np.zeros(count - L)
            for j in range(L):
                combined += m[j: count - L + j, j]
            if self.demo_weight != 1.0:
                slide = self.slides[si]
                t = np.asarray(slide.taught[L - 1: count - 1], dtype=float)
                combined *= 1.0 + (self.demo_weight - 1.0) * t
            for kk in range(count - L):
                scored.append((combined[kk], si, kk + L - 1))
        if not scored:
            return []
        scored.sort(reverse=True)
        # decode a candidate window beyond top_m so margin_agree (below) can
        # look for the nearest DISAGREEING competitor, not just the nearest one
        n_cand = min(len(scored), max(top_m, 8))
        cands = []
        for score, si, k in scored[:n_cand]:
            slide = self.slides[si]
            cleaned = self._decode_slide(slide)
            h = min(horizon, slide.count - 1 - k)
            cands.append((score, si, k,
                          cleaned[k + 1: k + 1 + h].tolist(),
                          slide.tags[k + 1: k + 1 + h]))
        out = []
        for i, (score, si, k) in enumerate(scored[:top_m]):
            slide = self.slides[si]
            cleaned = self._decode_slide(slide)
            h = min(horizon, slide.count - 1 - k)
            # margin: relative lead over the runner-up in the FULL ranking —
            # the recall-confidence signal (#6); 1.0 when unopposed.
            # margin_slide: lead over the best candidate from a DIFFERENT
            # slide — identity confidence (same-slide neighbors are near-ties
            # by window overlap, so raw margin under-reads confident recalls).
            if i + 1 < len(scored):
                margin = float((score - scored[i + 1][0]) / max(score, 1e-9))
            else:
                margin = 1.0
            margin_slide = 1.0
            for s2, si2, _ in scored[i + 1:]:
                if si2 != si:
                    margin_slide = float((score - s2) / max(score, 1e-9))
                    break
            # margin_agree: lead over the nearest candidate that predicts
            # something DIFFERENT (continuation symbol or conflicting tags).
            # Re-experiencing writes duplicate slides that tie on score while
            # agreeing on content — corroboration, not ambiguity (#9c). This
            # is the confidence signal for value/policy import.
            my_next = cands[i][3] if i < len(cands) else []
            my_tag = float(np.mean(cands[i][4])) if i < len(cands) and cands[i][4] else 0.0
            margin_agree = 1.0 if my_next else 0.0
            for s2, si2, k2, nxt2, tags2 in cands[i + 1:]:
                if not my_next or not nxt2:
                    continue
                tag2 = float(np.mean(tags2)) if tags2 else 0.0
                if nxt2[0] != my_next[0] or abs(tag2 - my_tag) > 0.5:
                    margin_agree = float((score - s2) / max(score, 1e-9))
                    break
            out.append({
                "slide": si,
                "position": k,
                "score": float(score),
                "margin": margin,
                "margin_slide": margin_slide,
                "margin_agree": margin_agree,
                "matched_symbol": int(cleaned[k]),
                "matched_tag": slide.tags[k],
                "next_symbols": cleaned[k + 1: k + 1 + h].tolist(),
                "next_tags": slide.tags[k + 1: k + 1 + h],
            })
        return out

    # ── stats ──────────────────────────────────────────────────────────────

    @property
    def n_records(self) -> int:
        return sum(s.count for s in self.slides)
