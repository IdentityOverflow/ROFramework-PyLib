"""
Response Field — the Network as a Picture

Feed a continuous 2D coordinate (x, y) into the grokked modular-addition MLP
and color each pixel by the network's output class.

The network only defines embeddings at integer indices 0..96, so between
integers we extend each embedding dimension by Fourier (trigonometric)
interpolation — the bandlimited continuation, which is also the network's
own representational basis. If the grokked embeddings really implement the
phasor code e^(i·2πka/97), this continuation is exact for it, and the
network should compute *continuous* modular addition: diagonal cyclic
stripes z = (x+y) mod 97. The init network, painted the same way, is the
control.

Outputs (in checkpoints/): fig_response_field.png
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

CKPT_DIR = Path(__file__).parent / "checkpoints"
P = 97
RES = 5          # samples per integer step → (P*RES)² pixels
CMAP = cm.get_cmap("twilight")  # cyclic: class 0 and class 96 are neighbors

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "axes.titlesize": 10,
})


def fourier_interp(E, t):
    """Bandlimited continuation of embedding table E (P, D) at positions t.

    Evaluates each column's DFT as a sum over frequencies k = -48..48 at
    continuous t, exact at integers. Returns (len(t), D).
    """
    Z = np.fft.fft(E, axis=0)                       # (P, D) complex
    k = np.fft.fftfreq(P, d=1.0 / P)                # signed frequencies
    phase = np.exp(2j * np.pi * np.outer(t, k) / P)  # (T, P)
    return (phase @ Z).real / P


def response_field(sd, t):
    """Argmax class over the (x, y) grid for one state_dict, at positions t."""
    Ea = fourier_interp(sd["embed_a.weight"].numpy(), t)
    Eb = fourier_interp(sd["embed_b.weight"].numpy(), t)
    W1 = sd["fc1.weight"].numpy()
    b1 = sd["fc1.bias"].numpy()
    W2 = sd["fc2.weight"].numpy()
    b2 = sd["fc2.bias"].numpy()

    D = Ea.shape[1]
    A = Ea @ W1[:, :D].T                            # (T, H) a-contribution
    B = Eb @ W1[:, D:].T + b1                       # (T, H) b-contribution + bias

    out = np.zeros((len(t), len(t)), dtype=np.int32)
    for i in range(len(t)):                         # row-chunked to bound memory
        h = np.maximum(A[i][None, :] + B, 0.0)      # (T, H)
        out[i] = (h @ W2.T + b2).argmax(axis=1)
    return out


def main():
    init = torch.load(CKPT_DIR / "ckpt_init.pt", map_location="cpu")
    final = torch.load(CKPT_DIR / "ckpt_e7499.pt", map_location="cpu")

    t = np.arange(0, P, 1.0 / RES)
    target = (t[:, None] + t[None, :]) % P

    field_final = response_field(final, t)
    field_init = response_field(init, t)

    # Accuracy on the integer lattice; circular error off-lattice.
    lattice = field_final[::RES, ::RES]
    lat_target = (np.arange(P)[:, None] + np.arange(P)[None, :]) % P
    lattice_acc = (lattice == lat_target).mean()
    circ = np.abs(field_final - np.round(target)) % P
    circ = np.minimum(circ, P - circ)
    off_err = circ.mean()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    panels = [
        (target, "target: z = (x+y) mod 97"),
        (field_final, f"grokked network (epoch 7499)\n"
                      f"lattice {lattice_acc:.0%} · circ. err {off_err:.2f}"),
        (field_init, "init network (untrained)"),
    ]
    for ax, (F, title) in zip(axes, panels):
        im = ax.imshow(F, origin="lower", cmap=CMAP, vmin=0, vmax=P - 1,
                       extent=[0, P, 0, P], interpolation="nearest")
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("x (continuous input a)")
        ax.set_ylabel("y (continuous input b)")
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                        label="output class (cyclic)")
    cbar.outline.set_visible(False)

    fig.suptitle("The network as a picture — continuous (x, y) in, argmax class out "
                 "(Fourier-interpolated embeddings)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.savefig(CKPT_DIR / "fig_response_field.png", dpi=150,
                bbox_inches="tight")

    print(f"lattice accuracy: {lattice_acc:.1%}")
    print(f"off-lattice mean circular error: {off_err:.3f} classes")
    print(f"figure written to {CKPT_DIR / 'fig_response_field.png'}")


if __name__ == "__main__":
    main()
