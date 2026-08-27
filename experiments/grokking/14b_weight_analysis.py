"""
Weight Pattern Comparison — Before vs After Grokking

Loads the checkpoints saved by 14_weight_checkpoints.py and compares the
network's weight patterns before training and after grokking.

The central point: in raw weight space, the init and grokked networks look
similarly unstructured. The grokked structure lives in the Fourier basis —
taking the DFT of each embedding column along the input index (and of each
fc2 column along the output class index) reveals that after grokking, all
spectral power concentrates on the handful of frequencies the model selected,
while at init the spectrum is flat.

Outputs (in checkpoints/):
    fig_embeddings.png — embed_a raw + DFT heatmaps, init vs grokked
    fig_readout_dynamics.png — fc2 spectra, spectral concentration, norms
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CKPT_DIR = Path(__file__).parent / "checkpoints"
P = 97
MAX_K = (P - 1) // 2  # 48 unique non-DC frequencies

# Palette (validated): surface/ink/muted + diverging blue<->red over neutral
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIVERGING = LinearSegmentedColormap.from_list(
    "ro_div", ["#2a78d6", "#f0efec", "#e34948"]
)
SEQ = LinearSegmentedColormap.from_list(
    "ro_seq", ["#f0efec", "#86b6ef", "#2a78d6", "#184f95"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "font.size": 9,
    "axes.titlesize": 10,
})


def load(name):
    return torch.load(CKPT_DIR / name, map_location="cpu")


def spectrum(W, axis=0):
    """|DFT|^2 of W along `axis` (the index-0..p-1 axis), frequencies 1..MAX_K.

    Returns (MAX_K, other_dim) power array and the per-frequency fraction of
    total non-DC power.
    """
    dft = np.fft.fft(W, axis=axis)
    power = np.abs(dft) ** 2
    power = np.moveaxis(power, axis, 0)[1:MAX_K + 1]  # drop DC + conjugates
    frac = power.sum(axis=1) / power.sum()
    return power, frac


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    init = load("ckpt_init.pt")
    final = load("ckpt_e7499.pt")
    with open(CKPT_DIR / "history.json") as f:
        history = json.load(f)

    Ea_i = init["embed_a.weight"].numpy()   # (97, 128)
    Ea_f = final["embed_a.weight"].numpy()
    fc2_i = init["fc2.weight"].numpy()      # (97, 128)
    fc2_f = final["fc2.weight"].numpy()

    # ---------------- Figure 1: embedding weights ----------------
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))

    for ax, W, label in [(axes[0, 0], Ea_i, "init"), (axes[0, 1], Ea_f, "after grokking")]:
        v = np.percentile(np.abs(W), 99)
        im = ax.imshow(W.T, aspect="auto", cmap=DIVERGING, vmin=-v, vmax=v,
                       interpolation="nearest")
        ax.set_title(f"embed_a raw weights — {label}", color=INK, loc="left")
        ax.set_xlabel("input index a")
        ax.set_ylabel("embedding dim")
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).outline.set_visible(False)

    for ax, W, label in [(axes[1, 0], Ea_i, "init"), (axes[1, 1], Ea_f, "after grokking")]:
        power, _ = spectrum(W, axis=0)      # (48, 128)
        im = ax.imshow(power, aspect="auto", cmap=SEQ,
                       vmax=np.percentile(power, 99.5), interpolation="nearest")
        ax.set_title(f"|DFT over a|² per embedding dim — {label}", color=INK, loc="left")
        ax.set_xlabel("embedding dim")
        ax.set_ylabel("frequency k")
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).outline.set_visible(False)

    fig.suptitle("Embedding weights before vs after grokking (p=97, seed 42)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CKPT_DIR / "fig_embeddings.png", dpi=150)
    plt.close(fig)

    # ---------------- Figure 2: spectra, concentration, norms ----------------
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))

    ks = np.arange(1, MAX_K + 1)
    for ax, (Wi, Wf), title in [
        (axes[0, 0], (Ea_i, Ea_f), "embed_a: power fraction per frequency (DFT over a)"),
        (axes[0, 1], (fc2_i, fc2_f), "fc2: power fraction per frequency (DFT over output class)"),
    ]:
        _, frac_i = spectrum(Wi, axis=0)
        _, frac_f = spectrum(Wf, axis=0)
        style_axes(ax)
        ax.plot(ks, frac_i, color=MUTED, linewidth=2, label="init")
        ax.plot(ks, frac_f, color=BLUE, linewidth=2, label="after grokking")
        top = np.argsort(frac_f)[::-1][:5]
        for k_idx in top:
            if frac_f[k_idx] > 0.04:
                ax.annotate(f"k={ks[k_idx]}", (ks[k_idx], frac_f[k_idx]),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", color=INK2, fontsize=8)
        ax.set_ylim(0, max(frac_f.max(), frac_i.max()) * 1.3)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("frequency k")
        ax.set_ylabel("fraction of non-DC power")
        ax.legend(frameon=False, labelcolor=INK2, loc="upper right", fontsize=8)

    # Norm trajectory across checkpoints
    ax = axes[1, 0]
    style_axes(ax)
    epochs = [0, 250, 500, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 6000, 7499]
    # embed_a/embed_b and fc1/fc2 trajectories nearly coincide — stagger the
    # direct labels vertically so they stay readable.
    names = [("embed_a.weight", "embed_a", BLUE, 5),
             ("embed_b.weight", "embed_b", ORANGE, -7),
             ("fc1.weight", "fc1", AQUA, 5),
             ("fc2.weight", "fc2", YELLOW, -7)]
    for key, label, color, dy in names:
        norms = []
        for e in epochs:
            sd = init if e == 0 else load(f"ckpt_e{e}.pt")
            norms.append(np.linalg.norm(sd[key].numpy()))
        norms = np.array(norms) / norms[0]
        ax.plot(epochs, norms, color=color, linewidth=2)
        ax.annotate(label, (epochs[-1], norms[-1]), textcoords="offset points",
                    xytext=(5, dy), color=color, fontsize=8)
    ax.axvline(4000, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("grok", (4000, ax.get_ylim()[1]), textcoords="offset points",
                xytext=(4, -10), color=MUTED, fontsize=8)
    ax.set_xlim(0, 8600)
    ax.set_title("Frobenius norm per layer (relative to init)", color=INK, loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("‖W‖ / ‖W_init‖")

    # Accuracy trajectory
    ax = axes[1, 1]
    style_axes(ax)
    h_ep = [h["epoch"] for h in history]
    ax.plot(h_ep, [h["train_acc"] for h in history], color=MUTED, linewidth=2,
            label="train")
    ax.plot(h_ep, [h["test_acc"] for h in history], color=BLUE, linewidth=2,
            label="test")
    ax.axvline(4000, color=MUTED, linewidth=1, linestyle="--")
    ax.set_title("Accuracy (context)", color=INK, loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, labelcolor=INK2, loc="center right")

    fig.suptitle("Spectral concentration and weight dynamics through grokking",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(CKPT_DIR / "fig_readout_dynamics.png", dpi=150)
    plt.close(fig)

    # ---------------- Console summary ----------------
    for name, Wi, Wf in [("embed_a", Ea_i, Ea_f), ("fc2", fc2_i, fc2_f)]:
        _, fi = spectrum(Wi, axis=0)
        _, ff = spectrum(Wf, axis=0)
        top_f = np.argsort(ff)[::-1][:5]
        top_share_f = ff[top_f].sum()
        top_share_i = np.sort(fi)[::-1][:5].sum()
        print(f"{name}: top-5 frequency share  init={top_share_i:.1%}  "
              f"grokked={top_share_f:.1%}  (winners: {[int(k)+1 for k in top_f]})")
    print(f"Figures written to {CKPT_DIR}")


if __name__ == "__main__":
    main()
