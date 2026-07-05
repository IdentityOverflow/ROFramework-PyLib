"""
holo_mount.py — Mount a holographic episodic reel inside a live brain.

Wraps HoloEpisodicMemory (holo_memory.py, validated in holographic #5) as an
online component for any reservoir brain: feed it the hidden state and the
reward each step, and it returns dv̂ — the recalled Δvalence of "the last time
the recent trajectory looked like this".

This is episodic control in the film substrate: instead of a k-NN table of
(state, value) pairs, the store is one superposed phasor vector per episode,
addressed by trajectory-stub recognition with the nonlinear decider inside
the loop (#4-B), and the recalled value rides back as side tags.

Lifecycle (all online, no offline passes):

  CALIBRATE  — collect (projected) hidden states for `calib_steps` steps;
               then fit PCA (no whitening — #5 v2), k-means VQ prototypes,
               and the phasor scale (median quantization σ_eff = 1.2, inside
               #4-B's tolerance ≈ 2). Reservoirs larger than `rp_dim` are
               first sketched with a fixed random projection so the PCA fit
               stays O(rp_dim²) regardless of reservoir size.
  RECORD     — event-based: a record happens only when the VQ symbol of the
               hidden state changes; reward rides along as the side tag.
               Slides close at the #4-A budget or on an episode boundary.
  RECALL     — on every new transition, query the reel with the last
               `context` transition features and vote dv̂ across the top-M
               retrieved continuations. Matches inside the query's own
               recent tail (active slide, last context+horizon records) are
               excluded — recalling "one second ago" is not foresight.
  FORESIGHT  — every dv̂ is held as a pending prediction and resolved once
               `horizon` further transitions have arrived: realized Δvalence
               = mean of their tags − tag at prediction time. The running
               Pearson r between predicted and realized Δvalence is the
               live honesty meter (persistence predicts 0 by construction).

Requires: numpy + holo_memory.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from holo_memory import (
    HoloEpisodicMemory, PhasorEncoder, VQCodebook, Whitener)


class HoloMount:
    """Online episodic reel for a live reservoir brain.

    Parameters
    ----------
    state_dim : dimension of the hidden state fed to step().
    calib_steps : steps of experience collected before the reel goes live.
    beta is NOT here — reward shaping is the brain's decision; the mount
    only reports dv̂.
    """

    def __init__(self, state_dim: int, calib_steps: int = 6000,
                 calib_stride: int = 3, rp_dim: int = 256, pca_d: int = 48,
                 k_proto: int = 64, n_phasor: int = 512, budget: int = 120,
                 context: int = 5, horizon: int = 3, top_m: int = 5,
                 sigma_target: float = 1.2, ema_tau: float = 12.0,
                 min_dwell: int = 2, demo_weight: float = 2.5,
                 seed: int = 0) -> None:
        self.ready = False
        self._calib_target = max(1, calib_steps // calib_stride)
        self._calib_stride = calib_stride
        self._calib: list[np.ndarray] = []
        self._step_count = 0

        rng = np.random.default_rng(seed)
        self._rp = None
        if state_dim > rp_dim:
            self._rp = (rng.normal(0.0, 1.0, (rp_dim, state_dim))
                        / np.sqrt(state_dim)).astype(np.float32)

        self._pca_d = pca_d
        self._k_proto = k_proto
        self._n_phasor = n_phasor
        self._budget = budget
        self._context = context
        self._horizon = horizon
        self._top_m = top_m
        self._sigma_target = sigma_target
        self._seed = seed
        # A noisy reservoir flips its VQ symbol every frame; the mount tracks
        # the SITUATION, so the sketched state is EMA-smoothed (leaky readout
        # of the echo) and transitions must dwell before they count.
        self._ema_alpha = 1.0 / max(ema_tau, 1.0)
        self._ema: np.ndarray | None = None
        self._min_dwell = min_dwell
        self._demo_weight = demo_weight
        self._last_taught = False

        self.memory: HoloEpisodicMemory | None = None
        self._whiten: Whitener | None = None
        self._recent: deque = deque(maxlen=context)
        self._pending: list[dict] = []
        self._preds: list[float] = []
        self._reals: list[float] = []
        self.last_dv = 0.0
        self.last_score = 0.0
        self.n_recalls = 0
        self._prec_hits = 0     # top match lands on the cue's own symbol

    # ── per-step entry point ───────────────────────────────────────────────

    def step(self, h: np.ndarray, tag: float, taught: bool = False) -> float:
        """Feed one hidden state + reward; returns current dv̂ (0 until live).

        `taught` marks steps executed under teacher forcing: a toggle closes
        the slide (demonstrations are their own episodes) and taught records
        get demo_weight priority at recall.
        """
        self._step_count += 1
        s = self._sketch(h).astype(np.float32)
        if self._ema is None:
            self._ema = s.copy()
        else:
            self._ema += self._ema_alpha * (s - self._ema)
        if not self.ready:
            if self._step_count % self._calib_stride == 0:
                self._calib.append(self._ema.copy())
                if len(self._calib) >= self._calib_target:
                    self._fit()
            return 0.0

        if taught != self._last_taught:
            self.boundary()          # demo start/end = event boundary
            self._last_taught = taught

        feat = self._whiten.transform(self._ema)
        sym = self.memory.observe(feat, tag=float(tag), taught=taught)
        if sym is None:
            return self.last_dv

        # a new transition landed: resolve foresight, then predict anew
        self._recent.append(feat)
        self._resolve_pending(float(tag))
        self._recall(float(tag))
        return self.last_dv

    def boundary(self) -> None:
        """Episode boundary (death/reset): close the slide, drop cross-episode
        context and pending predictions (their futures died with the episode)."""
        if self.memory is not None:
            self.memory.boundary()
        self._recent.clear()
        self._pending.clear()
        self.last_dv = 0.0

    # ── internals ──────────────────────────────────────────────────────────

    def _sketch(self, h: np.ndarray) -> np.ndarray:
        return self._rp @ h if self._rp is not None else h

    def _fit(self) -> None:
        samples = np.array(self._calib)
        self._calib = []
        self._whiten = Whitener(samples, d=self._pca_d, whiten=False)
        feats = self._whiten.transform(samples)
        probe = PhasorEncoder(self._pca_d, self._n_phasor, scale=1.0,
                              seed=self._seed)
        k = max(2, min(self._k_proto, len(feats) // 2))
        cb = VQCodebook.fit(feats, k, probe, seed=self._seed)
        qd = np.array([cb.quantize(f)[1] for f in feats])
        scale = self._sigma_target / max(np.median(qd), 1e-9)
        encoder = PhasorEncoder(self._pca_d, self._n_phasor, scale=scale,
                                seed=self._seed)
        self.memory = HoloEpisodicMemory(
            encoder, VQCodebook(cb.prototypes, encoder), budget=self._budget,
            min_dwell=self._min_dwell, demo_weight=self._demo_weight)
        self.ready = True
        print(f"[holo] reel is live: K={self._k_proto} prototypes, "
              f"scale s={scale:.3f}, budget {self._budget}/slide")

    def _resolve_pending(self, tag: float) -> None:
        done = []
        for p in self._pending:
            p["acc"].append(tag)
            if len(p["acc"]) >= self._horizon:
                self._preds.append(p["dv"])
                self._reals.append(float(np.mean(p["acc"])) - p["tag0"])
                done.append(p)
        for p in done:
            self._pending.remove(p)
        if len(self._preds) > 2000:          # rolling window
            self._preds = self._preds[-1000:]
            self._reals = self._reals[-1000:]

    def _recall(self, tag: float) -> None:
        mem = self.memory
        if mem.n_records < self._context + self._horizon + 2:
            return
        matches = mem.query(np.stack(self._recent), horizon=self._horizon,
                            top_m=self._top_m * 2)
        # foresight, not memory of a second ago: drop matches overlapping the
        # query stub's own tail in the active slide
        active = len(mem.slides) - 1
        cutoff = mem.slides[-1].count - (self._context + self._horizon)
        matches = [m for m in matches
                   if m["next_tags"]
                   and not (m["slide"] == active and m["position"] >= cutoff)]
        matches = matches[: self._top_m]
        if not matches:
            return
        w = np.array([m["score"] for m in matches])
        dv = np.array([np.mean(m["next_tags"]) - m["matched_tag"]
                       for m in matches])
        self.last_dv = float((w @ dv) / w.sum())
        self.last_score = float(w.max())
        self.n_recalls += 1
        self._prec_hits += int(matches[0]["matched_symbol"] == mem.last_sym)
        self._pending.append({"dv": self.last_dv, "tag0": tag, "acc": []})

    # ── reporting ──────────────────────────────────────────────────────────

    def foresight(self, event_min: float = 0.05):
        """(overall r, n, event r, event n): Pearson r between predicted and
        realized Δvalence, overall and restricted to moments where valence
        actually moved (|realized Δ| > event_min) — flat stretches dominate n
        and dilute r, but events are what recall is for."""
        p, q = np.array(self._preds), np.array(self._reals)
        n = len(p)

        def _r(a, b):
            if len(a) < 10 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        ev = np.abs(q) > event_min if n else np.zeros(0, bool)
        return _r(p, q), n, _r(p[ev], q[ev]), int(ev.sum())

    def stats(self) -> str:
        if not self.ready:
            pct = 100.0 * len(self._calib) / self._calib_target
            return f"holo: calibrating {pct:.0f}%"
        r, n, r_ev, n_ev = self.foresight()
        prec = self._prec_hits / max(self.n_recalls, 1)
        n_demo = sum(sum(s.taught) for s in self.memory.slides)
        demo = f"  demo={n_demo}" if n_demo else ""
        return (f"holo: {self.memory.n_records} rec / "
                f"{len(self.memory.slides)} slides{demo}  "
                f"dv̂={self.last_dv:+.3f}  "
                f"prec={prec:.0%}  foresight r={r:+.2f} (n={n})  "
                f"event r={r_ev:+.2f} (n={n_ev})")

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if not self.ready:
            return
        mem = self.memory
        np.savez(
            path,
            rp=self._rp if self._rp is not None else np.zeros(0),
            mean=self._whiten.mean, proj=self._whiten.proj,
            protos=mem.codebook.prototypes,
            scale=np.array(mem.encoder.scale),
            films=np.stack([s.film for s in mem.slides]),
            counts=np.array([s.count for s in mem.slides]),
            tags=np.array([np.pad(np.array(s.tags, dtype=np.float64),
                                  (0, self._budget - s.count))
                           for s in mem.slides]),
            taught=np.array([np.pad(np.array(s.taught, dtype=np.int8),
                                    (0, self._budget - s.count))
                             for s in mem.slides]),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        if data["rp"].size:
            self._rp = data["rp"]
        self._whiten = Whitener.__new__(Whitener)
        self._whiten.mean = data["mean"]
        self._whiten.proj = data["proj"]
        encoder = PhasorEncoder(self._pca_d, self._n_phasor,
                                scale=float(data["scale"]), seed=self._seed)
        self.memory = HoloEpisodicMemory(
            encoder, VQCodebook(data["protos"], encoder), budget=self._budget,
            min_dwell=self._min_dwell, demo_weight=self._demo_weight)
        self.memory.slides = []
        from holo_memory import Slide
        taught_arr = (data["taught"] if "taught" in data
                      else np.zeros_like(data["tags"], dtype=np.int8))
        for film, count, tags, taught in zip(data["films"], data["counts"],
                                             data["tags"], taught_arr):
            s = Slide(self._n_phasor)
            s.film = film
            s.count = int(count)
            s.tags = list(tags[: int(count)])
            s.taught = [int(v) for v in taught[: int(count)]]
            s.closed = True
            self.memory.slides.append(s)
        self.memory.boundary()      # resume recording on a fresh slide
        self.ready = True
        print(f"[holo] reel loaded: {self.memory.n_records} records / "
              f"{len(self.memory.slides)} slides")


# ── Self-test ──────────────────────────────────────────────────────────────────
# `python holo_mount.py` mounts the reel on a structured scripted explorer
# (the #5 Braitenberg policy) in the headless world. This isolates the mount
# from the brain: a policy with repeating behavioral motifs (approach food,
# eat, flee danger) should produce foresight r > 0; a noise-driven policy
# will not ("the reel is only as prophetic as the life it records").

def _self_test(n_steps: int = 45000, seed: int = 3) -> None:
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import World  # noqa: PLC0415

    mount = HoloMount(state_dim=263, calib_steps=5000, horizon=6, seed=seed)
    world = World(seed=seed)
    world.player_active = False
    world.add_agent()
    rng = np.random.default_rng(seed + 1)
    ou, prev_life = 0.0, 1.0

    for step in range(n_steps):
        obs = world.get_ai_observation()
        life = float(obs[260])
        if life > prev_life + 0.5:
            mount.boundary()
        prev_life = life
        mount.step(obs, float(world.ai.meters.valence))

        vis = obs[:242].reshape(121, 2)
        food = np.abs(vis[:, 0] - 0.50) < 0.01
        danger = np.abs(vis[:, 0] - 0.75) < 0.01
        fwd, turn = 0.6, 0.0
        if danger.any() and vis[danger, 1].max() > 0.55:
            j = int(np.argmax(np.where(danger, vis[:, 1], 0.0)))
            turn, fwd = -np.sign(j - 60) or 1.0, 0.25
        elif food.any():
            j = int(np.argmax(np.where(food, vis[:, 1], 0.0)))
            turn, fwd = float(np.clip((j - 60) / 45.0, -1.0, 1.0)), 0.8
        else:
            ou = 0.92 * ou + 0.35 * rng.normal()
            turn = float(np.clip(ou, -1.0, 1.0))
        world.step(ai_action=(fwd, turn, 1.0))

        if (step + 1) % 5000 == 0:
            print(f"step {step + 1:>6}  {mount.stats()}")


if __name__ == "__main__":
    _self_test()
