"""Cross-substrate gain curves from the saved results of #F2, #F3, #F3b."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def mean_curve(rows, xkey, ykey):
    xs = sorted({r[xkey] for r in rows})
    ys = [np.nanmean([r[ykey] for r in rows if r[xkey] == x]) for x in xs]
    return np.array(xs), np.array(ys)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rig = json.loads((HERE / "results_02_video_rig.json").read_text())
    rig_c3 = json.loads((HERE / "results_02_video_rig_c3p02.json").read_text())
    rig.update(rig_c3)
    sweep = json.loads((HERE / "results_03_gain_sweep.json").read_text())
    scan_p = HERE / "results_04_tuned_scan.json"
    scan = json.loads(scan_p.read_text()) if scan_p.exists() else None

    tw_p = HERE / "results_06_creditable_twist.json"
    tw = json.loads(tw_p.read_text()) if tw_p.exists() else None
    fig, axes = plt.subplots(1, 4 if tw else 3, figsize=(20 if tw else 15, 4.4))

    # (a) rig: D_box vs g, up-sweep, seed 0
    ax = axes[0]
    for key, style in (("plain/c1/zoom_out/seeded/s0", ("C0", "-", "clip, zoom out")),
                       ("plain/c1/zoom_in/seeded/s0", ("C1", "-", "clip, zoom in")),
                       ("plain/c3p0.2/zoom_out/seeded/s0", ("C0", "--", "decider, zoom out")),
                       ("plain/c3p0.2/zoom_in/seeded/s0", ("C1", "--", "decider, zoom in"))):
        if key not in rig:
            continue
        rows = rig[key]["rows"]
        n = len(rows) // 2
        up, down = rows[:n], rows[n:][::-1]
        c, ls, lab = style
        ax.plot([r["g"] for r in up], [r["dbox"] for r in up], color=c, ls=ls, marker="o", ms=3, label=lab)
        ax.plot([r["g"] for r in down], [r["dbox"] for r in down], color=c, ls=ls, alpha=0.35)
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("loop gain g"); ax.set_ylabel("box-counting dimension of lit mask")
    ax.set_title("(a) video rig  (faint = down-sweep)"); ax.legend(fontsize=8)

    # (b) ESN: D2 and λ vs g
    ax = axes[1]
    g, d2 = mean_curve(sweep["esn"], "g", "d2")
    _, lam = mean_curve(sweep["esn"], "g", "lyap")
    _, sat = mean_curve(sweep["esn"], "g", "sat")
    ax.plot(g, d2, "C2o-", label="D2 (PCA-20)")
    ax.plot(g, 10 * sat, "C3s--", ms=3, label="saturated fraction ×10")
    ax.set_xlabel("spectral radius g"); ax.set_ylabel("D2")
    ax2 = ax.twinx(); ax2.plot(g, lam, "k^:", ms=4, label="λ (Benettin)"); ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("largest Lyapunov exponent")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_title("(b) echo-state reservoir"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

    # (c) Seed: frozen scale sweep, self-tuned points, and the α·W* scan
    ax = axes[2]
    w, d2f = mean_curve(sweep["seed_frozen"], "w_scale", "d2")
    _, satf = mean_curve(sweep["seed_frozen"], "w_scale", "sat")
    ax.plot(w, d2f, "C0o-", label="frozen positive weights ×w")
    ax.plot(w, 5 * satf, "C0s:", ms=3, alpha=0.6, label="  saturated ×5")
    wt, d2t = mean_curve(sweep["seed_tuned"], "w_scale", "d2")
    ax.plot(wt, d2t, "C3*", ms=11, label="self-tuned (Rule 2a), σ≈1")
    if scan:
        a, d2a = mean_curve(scan["scan"], "alpha", "d2")
        _, sata = mean_curve(scan["scan"], "alpha", "sat")
        _, lama = mean_curve(scan["scan"], "alpha", "lyap")
        ax.plot(a, d2a, "C3o--", label="α·W* around self-tuned")
        ax.plot(a, 5 * sata, "C3s:", ms=3, alpha=0.6, label="  saturated ×5")
        ax.axhline(np.mean(scan["drive_d2"]), color="gray", lw=0.8, ls="--", label="drive alone D2")
        ax.axhline(np.nanmean([r["d2"] for r in scan["zero_w"]]), color="gray", lw=0.8, ls="-.", label="drive+noise, no loop")
    ax.set_xscale("log")
    ax.set_xlabel("weight scale (w for frozen, α for scan)"); ax.set_ylabel("D2")
    ax.set_title("(c) Seed"); ax.legend(fontsize=7)

    if tw:
        ax = axes[3]
        for cond, st in (("live", "C3o-"), ("blind", "C0s--"), ("stale", "C2^:"), ("off", "k.-.")):
            rows = [r for r in tw["scan"] if r["condition"] == cond]
            a, d2 = mean_curve(rows, "alpha", "d2")
            ax.plot(a, d2, st, label=f"{cond}")
        ax.set_xlabel("α (scale of the tuned weights W*)"); ax.set_ylabel("D2")
        ax.set_title("(d) Seed with a creditable twist: channel live / blind / stale / off")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "figures" / "05_gain_axis_three_substrates.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
