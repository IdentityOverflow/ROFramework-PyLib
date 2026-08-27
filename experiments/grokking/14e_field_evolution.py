"""
Field Evolution — Watching the Static Resolve into Stripes

Renders the continuous response field of 14d_response_field.py at every
saved checkpoint, laid out as a filmstrip through training: the untrained
static organizing into the diagonal cyclic stripes of (x+y) mod 97.

Each panel is annotated with the test accuracy at that epoch and the mean
circular error of the field against the continuous target.

Outputs (in checkpoints/): fig_field_evolution.png
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

CKPT_DIR = Path(__file__).parent / "checkpoints"
P = 97
RES = 3          # samples per integer step (coarser than 14d: 12 renders)
CMAP = cm.get_cmap("twilight")
EPOCHS = [0, 250, 500, 1000, 2000, 3000, 3500, 4000, 4500, 5000, 6000, 7499]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "font.size": 9,
})


def fourier_interp(E, t):
    """Bandlimited continuation of embedding table E (P, D) at positions t."""
    Z = np.fft.fft(E, axis=0)
    k = np.fft.fftfreq(P, d=1.0 / P)
    phase = np.exp(2j * np.pi * np.outer(t, k) / P)
    return (phase @ Z).real / P


def response_field(sd, t):
    """Argmax class over the (x, y) grid for one state_dict."""
    Ea = fourier_interp(sd["embed_a.weight"].numpy(), t)
    Eb = fourier_interp(sd["embed_b.weight"].numpy(), t)
    W1 = sd["fc1.weight"].numpy()
    b1 = sd["fc1.bias"].numpy()
    W2 = sd["fc2.weight"].numpy()
    b2 = sd["fc2.bias"].numpy()

    D = Ea.shape[1]
    A = Ea @ W1[:, :D].T
    B = Eb @ W1[:, D:].T + b1

    out = np.zeros((len(t), len(t)), dtype=np.int32)
    for i in range(len(t)):
        h = np.maximum(A[i][None, :] + B, 0.0)
        out[i] = (h @ W2.T + b2).argmax(axis=1)
    return out


def main():
    with open(CKPT_DIR / "history.json") as f:
        history = json.load(f)

    def test_acc(epoch):
        return min(history, key=lambda h: abs(h["epoch"] - epoch))["test_acc"]

    t = np.arange(0, P, 1.0 / RES)
    target = (t[:, None] + t[None, :]) % P

    fig, axes = plt.subplots(3, 4, figsize=(12.8, 9.9))

    for ax, epoch in zip(axes.flat, EPOCHS):
        name = "ckpt_init.pt" if epoch == 0 else f"ckpt_e{epoch}.pt"
        sd = torch.load(CKPT_DIR / name, map_location="cpu")
        field = response_field(sd, t)

        circ = np.abs(field - np.round(target)) % P
        err = np.minimum(circ, P - circ).mean()

        im = ax.imshow(field, origin="lower", cmap=CMAP, vmin=0, vmax=P - 1,
                       extent=[0, P, 0, P], interpolation="nearest")
        label = "init" if epoch == 0 else f"epoch {epoch}"
        ax.set_title(f"{label} — test {test_acc(epoch):.0%}, err {err:.1f}",
                     color=INK, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                        label="output class (cyclic)")
    cbar.outline.set_visible(False)

    fig.suptitle("Static resolving into stripes — continuous response field "
                 "through training (grok at ~4000)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.savefig(CKPT_DIR / "fig_field_evolution.png", dpi=150,
                bbox_inches="tight")
    print(f"figure written to {CKPT_DIR / 'fig_field_evolution.png'}")


if __name__ == "__main__":
    main()
