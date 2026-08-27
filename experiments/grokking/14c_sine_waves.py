"""
Where the Sine Waves Are — Grokked Network, Literal Waveforms

Companion to 14b: the DFT heatmaps show spectral bands, but the sinusoids
can be seen directly if you look in the right coordinates.

1. A hidden neuron's sum-averaged tuning curve h(s) is a nearly pure cosine
   after grokking (He et al. resonance selection) — plotted against its
   single-frequency DFT reconstruction, with the same neuron at init.
2. A single raw embedding column mixes all ~5 winning frequencies, so it
   looks like structured wiggle, not one wave.
3. Projecting the embedding rows onto one frequency's 2D Fourier plane
   shows the phasor code: the 97 inputs arranged on a circle, ordered by
   (k·a) mod p.

Outputs (in checkpoints/): fig_sine_waves.png
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_DIR = Path(__file__).parent / "checkpoints"
P = 97

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "axes.titlesize": 10,
})


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def sum_averaged_hidden(sd):
    """Post-ReLU hidden activations averaged by sum class, from a state_dict."""
    Ea = torch.tensor(sd["embed_a.weight"].numpy())
    Eb = torch.tensor(sd["embed_b.weight"].numpy())
    W1, b1 = torch.tensor(sd["fc1.weight"].numpy()), torch.tensor(sd["fc1.bias"].numpy())
    grid_a = torch.arange(P).repeat_interleave(P)
    grid_b = torch.arange(P).repeat(P)
    x = torch.cat([Ea[grid_a], Eb[grid_b]], dim=-1)
    h = torch.relu(x @ W1.T + b1).numpy()
    sums = ((grid_a + grid_b) % P).numpy()
    h_avg = np.zeros((P, h.shape[1]))
    for s in range(P):
        h_avg[s] = h[sums == s].mean(axis=0)
    return h_avg


def main():
    init = torch.load(CKPT_DIR / "ckpt_init.pt", map_location="cpu")
    final = torch.load(CKPT_DIR / "ckpt_e7499.pt", map_location="cpu")

    h_init = sum_averaged_hidden(init)
    h_final = sum_averaged_hidden(final)

    # Pick the neuron whose tuning curve is the purest single sinusoid,
    # restricted to low frequencies so the wave is visually readable.
    dft = np.fft.fft(h_final - h_final.mean(axis=0), axis=0)
    power = np.abs(dft[1:(P - 1) // 2 + 1]) ** 2
    purity = power.max(axis=0) / power.sum(axis=0)
    dom_k = power.argmax(axis=0) + 1
    low = np.where(dom_k <= 12)[0]
    j = int(low[np.argmax(purity[low])]) if len(low) else int(np.argmax(purity))
    kj = int(dom_k[j])

    # Single-frequency reconstruction from the DFT component.
    s = np.arange(P)
    coef = dft[kj, j]
    recon = h_final[:, j].mean() + 2 / P * (
        coef.real * np.cos(2 * np.pi * kj * s / P)
        - coef.imag * np.sin(2 * np.pi * kj * s / P)
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    ax = axes[0]
    style_axes(ax)
    ax.plot(s, h_init[:, j], color=MUTED, linewidth=1.5, label="init")
    ax.plot(s, h_final[:, j], color=BLUE, linewidth=2, label="after grokking")
    ax.plot(s, recon, color=INK, linewidth=1, linestyle="--",
            label=f"pure cos, k={kj}")
    ax.set_title(f"Hidden neuron {j}: tuning curve over sum class\n"
                 f"(purity {purity[j]:.1%} in one frequency)", color=INK, loc="left")
    ax.set_xlabel("s = (a+b) mod 97")
    ax.set_ylabel("mean activation")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))  # headroom so the legend clears the wave
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right", ncol=3,
              columnspacing=1.0, handlelength=1.4)

    # A raw embedding column: superposition of the winning frequencies.
    Ea = final["embed_a.weight"].numpy()
    col_power = (np.abs(np.fft.fft(Ea, axis=0)[1:(P - 1) // 2 + 1]) ** 2).sum(axis=0)
    d = int(np.argmax(col_power))
    ax = axes[1]
    style_axes(ax)
    ax.plot(np.arange(P), Ea[:, d], color=BLUE, linewidth=2)
    ax.set_title(f"Raw embedding dim {d} over input index\n"
                 "(all 5 frequencies superposed — no single wave)",
                 color=INK, loc="left")
    ax.set_xlabel("input index a")
    ax.set_ylabel("weight value")

    # Fourier-plane projection at the dominant frequency: the phasor circle.
    k_dom = 7
    z = np.fft.fft(Ea, axis=0)[k_dom]           # complex loading per dim
    u, v = z.real, z.imag
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    x, y = Ea @ u, Ea @ v
    order = np.argsort((k_dom * np.arange(P)) % P)
    ax = axes[2]
    style_axes(ax)
    ax.plot(x[order], y[order], color=GRID, linewidth=1, zorder=1)
    sc = ax.scatter(x, y, c=(k_dom * np.arange(P)) % P, cmap="twilight",
                    s=22, zorder=2, edgecolors=SURFACE, linewidths=0.5)
    for a in range(0, P, 12):
        ax.annotate(str(a), (x[a], y[a]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=INK2)
    ax.set_aspect("equal")
    ax.set_title(f"Embeddings in the k={k_dom} Fourier plane\n"
                 f"(the phasor circle, ordered by {k_dom}·a mod 97)",
                 color=INK, loc="left")
    ax.set_xlabel(f"cos(2π·{k_dom}·a/97) direction")
    ax.set_ylabel(f"sin(2π·{k_dom}·a/97) direction")

    fig.suptitle("The sine waves, seen directly (grokked network, epoch 7499)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(CKPT_DIR / "fig_sine_waves.png", dpi=150)

    print(f"purest neuron: {j} (k={kj}, purity {purity[j]:.3f})")
    print(f"mean purity across neurons: {purity.mean():.3f}")
    print(f"figure written to {CKPT_DIR / 'fig_sine_waves.png'}")


if __name__ == "__main__":
    main()
