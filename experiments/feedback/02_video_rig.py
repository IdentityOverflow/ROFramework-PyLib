"""#F2 — The video-feedback rig as a closure experiment.

A camera pointed at its own monitor: frame_{t+1} = sat(g · A(frame_t) + in_t),
where A is the optical map (rotate by θ, zoom by s, slight blur), g is the
loop gain (brightness/contrast knob), sat is the sensor's saturation, in_t a
fixed seed object held in front of the screen (a bright arc). This is the
paradigm CLOSED, UNTWISTED loop of ro_framework.md §5.3 — the output is
consumed by the camera's own next exposure, and nothing in it indexes its
own mapping.

Sweeps
------
g from 0.5 → 1.6 → 0.5 (up-down, state carried over) for zoom-out (s<1,
copies shrink inward: an iterated-function-system attractor) and zoom-in
(s>1, copies grow outward: the tunnel).

Measures per g: mean luminance, saturated fraction, box-counting dimension
D_box of the lit mask (luminance > 0.5), spectral slope, and the one-step
growth factor of a tiny perturbation (image-space Lyapunov).

Protocols:
  seeded   the seed object is present every frame.
  flash    the seed object is shown for the first 10 frames only, then
           removed. Whether the pattern self-sustains at a given g is the
           rig's famous hysteresis (black is always a fixed point at g>1
           with no input; lit stays lit).

Twin arm (the twist probe, pre-registered null):
  The mapping drifts slowly (the artist turning the camera):
  θ(t) = θ0 + 3° sin(2πt/400), s(t) = s0 + 0.02 sin(2πt/400 + 1).
  T  a small glyph rendering the CURRENT (θ, s) — a self-portrait of the
     mapping — is injected into every frame: the frame carries a
     representation of its own transform, consumed by the next exposure.
  U  the same glyph rendered at the FIXED (θ0, s0): identical pixel budget,
     no information about the live mapping.
  Prediction (from §5.4 and from #C1's T≡U): the D_box curves of T and U
  are identical within seed noise. The fractal is a property of closure +
  gain, not of self-indexing.

Nothing here is a consciousness claim. The experiment measures what the
geometry of a closed untwisted loop does across the gain axis so the same
axis can be read on the reservoir and the Seed (#F3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fractal_tools import box_counting_dimension, effective_depth, spectral_slope  # noqa: E402

N = 128
CENTER = np.array([N / 2, N / 2])


def seed_object(n: int = N) -> np.ndarray:
    """A bright arc off-centre: the hand / light held in front of the screen."""
    yy, xx = np.indices((n, n))
    cy, cx = n * 0.62, n * 0.58
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.arctan2(yy - cy, xx - cx)
    arc = (np.abs(r - n * 0.18) < 1.5) & (ang > -0.3) & (ang < 2.2)
    return arc.astype(float)


def glyph(theta: float, s: float, n: int = N) -> np.ndarray:
    """A self-portrait of the mapping: a short bar at angle theta with length
    proportional to zoom, drawn in a corner. T renders the live (θ,s),
    U renders a frozen pair."""
    img = np.zeros((n, n))
    cy, cx = n * 0.15, n * 0.15
    L = n * 0.06 * (s / 0.92)
    for t in np.linspace(-1, 1, 60):
        y = int(round(cy + t * L * np.sin(theta * 6)))
        x = int(round(cx + t * L * np.cos(theta * 6)))
        if 0 <= y < n and 0 <= x < n:
            img[y, x] = 1.0
    return ndimage.gaussian_filter(img, 0.6)


def optical_map(frame: np.ndarray, theta: float, s: float, blur: float) -> np.ndarray:
    """Rotate by theta, zoom by s about the centre, then blur (optics)."""
    c, si = np.cos(theta), np.sin(theta)
    # output coords -> input coords: inverse of (zoom s, rotate theta)
    M = np.array([[c, -si], [si, c]]) / s
    offset = CENTER - M @ CENTER
    out = ndimage.affine_transform(frame, M, offset=offset, order=1, mode="constant", cval=0.0)
    if blur > 0:
        out = ndimage.gaussian_filter(out, blur)
    return out


PIVOT = 0.5


def sat(pre: np.ndarray, contrast: float) -> np.ndarray:
    """Sensor/monitor transfer curve: a contrast gain about the pivot grey
    level, then hard saturation. contrast=1 is a plain clip; contrast>1 is a
    sharpening decider (monitor gamma + camera black level in a real rig).
    The pivot is the black level: anything below pivot*(1-1/contrast) is cut."""
    return np.clip(contrast * (pre - PIVOT) + PIVOT, 0.0, 1.0)


def step(frame, g, theta, s, blur, inp, noise, rng, contrast=1.0):
    pre = g * optical_map(frame, theta, s, blur) + inp + noise * rng.standard_normal(frame.shape)
    return sat(pre, contrast)


def measure(frame, g, theta, s, blur, inp, rng, contrast=1.0):
    lit = frame > 0.5
    dbox, _, _ = box_counting_dimension(lit, max_box=N // 4)
    # one-step perturbation growth (image-space Lyapunov, deterministic part)
    eps = 1e-4
    pert = frame + eps * rng.standard_normal(frame.shape)
    a = sat(g * optical_map(frame, theta, s, blur) + inp, contrast)
    b = sat(g * optical_map(pert, theta, s, blur) + inp, contrast)
    growth = float(np.linalg.norm(b - a) / (eps * np.sqrt(frame.size)))
    return dict(lum=float(frame.mean()), sat=float((frame >= 0.999).mean()),
                lit=float(lit.mean()), dbox=float(dbox),
                slope=spectral_slope(frame), growth=growth)


def g_schedule(lo, hi, n):
    up = list(np.round(np.linspace(lo, hi, n), 3))
    return up + up[::-1]


def run(arm: str, protocol: str, s0: float, theta0: float, seed: int,
        n_gain: int, per_gain: int, blur: float = 0.7, noise: float = 0.002,
        a_in: float = 0.35, a_glyph: float = 0.6, drift_period: int = 400,
        contrast: float = 1.0):
    rng = np.random.default_rng(seed)
    obj = a_in * seed_object()
    frame = np.zeros((N, N))
    t = 0
    rows = []
    frames = {}
    for g in g_schedule(0.5, 1.6, n_gain):
        for k in range(per_gain):
            if arm in ("T", "U"):
                theta = theta0 + np.deg2rad(3.0) * np.sin(2 * np.pi * t / drift_period)
                s = s0 + 0.02 * np.sin(2 * np.pi * t / drift_period + 1.0)
                gl = glyph(theta, s) if arm == "T" else glyph(theta0, s0)
                inp = obj + a_glyph * gl
            else:
                theta, s = theta0, s0
                inp = obj if (protocol == "seeded" or t < 10) else np.zeros_like(obj)
            frame = step(frame, g, theta, s, blur, inp, noise, rng, contrast)
            t += 1
        m = measure(frame, g, theta, s, blur, inp, rng, contrast)
        m.update(g=float(g), t=t, depth=effective_depth(g))
        rows.append(m)
        frames[len(rows) - 1] = frame.copy()
    return rows, frames


def summarize(label, rows, n_gain):
    up, down = rows[:n_gain], rows[n_gain:]
    print(f"\n--- {label} ---")
    print(f"  {'g':>5} {'lum↑':>6} {'lit↑':>6} {'sat↑':>6} {'Dbox↑':>6} {'grow↑':>6} | "
          f"{'lum↓':>6} {'lit↓':>6} {'Dbox↓':>6}")
    for u, d in zip(up, down[::-1]):
        print(f"  {u['g']:5.2f} {u['lum']:6.3f} {u['lit']:6.3f} {u['sat']:6.3f} {u['dbox']:6.3f} "
              f"{u['growth']:6.3f} | {d['lum']:6.3f} {d['lit']:6.3f} {d['dbox']:6.3f}")
    hyst = float(np.trapz([d['lit'] for d in down[::-1]], [d['g'] for d in down[::-1]])
                 - np.trapz([u['lit'] for u in up], [u['g'] for u in up]))
    print(f"  hysteresis area (lit, down − up) = {hyst:+.4f}")
    return hyst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gain", type=int, default=12)
    ap.add_argument("--per-gain", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--contrast", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--pivot", type=float, default=0.5)
    ap.add_argument("--save", default=str(HERE / "results_02_video_rig.json"))
    ap.add_argument("--figs", default=str(HERE / "figures"))
    args = ap.parse_args()

    global PIVOT
    PIVOT = args.pivot
    theta0 = np.deg2rad(7.0)
    results = {}
    t0 = time.time()
    for contrast in args.contrast:
      c = f"c{contrast:g}" + (f"p{args.pivot:g}" if args.pivot != 0.5 else "")
      for zoom_name, s0 in (("zoom_out", 0.92), ("zoom_in", 1.08)):
        for protocol in ("seeded", "flash"):
            for seed in range(args.seeds):
                rows, frames = run("plain", protocol, s0, theta0, seed, args.n_gain, args.per_gain,
                                   contrast=contrast)
                key = f"plain/{c}/{zoom_name}/{protocol}/s{seed}"
                results[key] = dict(rows=rows, hyst=summarize(key, rows, args.n_gain))
                if seed == 0:
                    _save_strip(frames, rows, args.n_gain, Path(args.figs), key.replace("/", "_"))
        for arm in ("T", "U"):
            for seed in range(args.seeds):
                rows, frames = run(arm, "seeded", s0, theta0, seed, args.n_gain, args.per_gain,
                                   contrast=contrast)
                key = f"{arm}/{c}/{zoom_name}/drift/s{seed}"
                results[key] = dict(rows=rows, hyst=summarize(key, rows, args.n_gain))
                if seed == 0 and arm == "T":
                    _save_strip(frames, rows, args.n_gain, Path(args.figs), key.replace("/", "_"))
        # twin-arm comparison
        for seed in range(args.seeds):
            T = results[f"T/{c}/{zoom_name}/drift/s{seed}"]["rows"]
            U = results[f"U/{c}/{zoom_name}/drift/s{seed}"]["rows"]
            dT = np.array([r["dbox"] for r in T]); dU = np.array([r["dbox"] for r in U])
            lT = np.array([r["lit"] for r in T]); lU = np.array([r["lit"] for r in U])
            print(f"\n  TWIN {c} {zoom_name} seed {seed}: max|ΔDbox|={np.abs(dT-dU).max():.3f} "
                  f"mean|ΔDbox|={np.abs(dT-dU).mean():.3f}  max|Δlit|={np.abs(lT-lU).max():.3f}")
    print(f"\ntotal {time.time()-t0:.1f}s")
    Path(args.save).write_text(json.dumps(results, indent=1, default=float))
    print(f"saved {args.save}")


def _save_strip(frames, rows, n_gain, figdir: Path, name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figdir.mkdir(parents=True, exist_ok=True)
    idx = list(range(0, n_gain, max(1, n_gain // 6)))[:6]
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.8))
    for a, i in zip(axes[0], idx):
        a.imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"g={rows[i]['g']:.2f} ↑\nD={rows[i]['dbox']:.2f}", fontsize=8)
        a.axis("off")
    for a, i in zip(axes[1], idx):
        j = 2 * n_gain - 1 - i
        a.imshow(frames[j], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"g={rows[j]['g']:.2f} ↓\nD={rows[j]['dbox']:.2f}", fontsize=8)
        a.axis("off")
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / f"02_{name}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
