"""
Iterated Map — Chaos In the Code, Error Correction At the Readout

The grokked MLP computes (a+b) mod 97. Tie its inputs and iterate:

    z_{n+1} = decode(net(z_n, z_n))  ~  2·z_n mod 97

— the doubling map, the minimal chaotic system (Lyapunov exponent ln 2,
conjugate to the binary shift). Whether the iterated network is actually
chaotic turns out to depend entirely on WHERE you read the state out:

* Output level (softmax over the 97 classes → circular mean): the readout
  is a quantizer — a staircase in z. Every iteration snaps the state back
  to the nearest integer code word, the continuous fraction (where the
  chaos lives) is discarded, nearby orbits merge, and the dynamics reduce
  to the exact integer permutation z → 2z mod 97, which is periodic with
  period ord(2) mod 97 = 48. Chaos erased by error correction.

* Representation level (matched-filter phase readout from the hidden
  layer, using the network's own Fourier templates at its winning
  frequencies): the fraction survives, the map is a smooth degree-2
  expanding circle map, and iteration is genuinely chaotic — separation
  of nearby orbits grows with slope ≈ ln 2, and the (z_0, n) fan shows
  the doubling map's self-similar halving bands.

Outputs (in checkpoints/): fig_iterated_map.png
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
LN2 = np.log(2)

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


class ContinuousNet:
    """The MLP with Fourier-interpolated embeddings and two readouts."""

    def __init__(self, sd):
        self.Za = np.fft.fft(sd["embed_a.weight"].numpy(), axis=0)
        self.Zb = np.fft.fft(sd["embed_b.weight"].numpy(), axis=0)
        self.k = np.fft.fftfreq(P, d=1.0 / P)
        self.W1 = sd["fc1.weight"].numpy()
        self.b1 = sd["fc1.bias"].numpy()
        self.W2 = sd["fc2.weight"].numpy()
        self.b2 = sd["fc2.bias"].numpy()
        self.class_phasor = np.exp(2j * np.pi * np.arange(P) / P)
        self._build_templates()

    def _embed(self, Z, t):
        phase = np.exp(2j * np.pi * np.outer(t, self.k) / P)
        return (phase @ Z).real / P

    def hidden(self, za, zb):
        x = np.concatenate([self._embed(self.Za, za), self._embed(self.Zb, zb)], axis=1)
        return np.maximum(x @ self.W1.T + self.b1, 0.0)

    def _build_templates(self):
        """Fourier templates of the hidden layer over sum class s.

        h_avg(s) from the integer lattice, DFT over s; rows at the model's
        dominant frequencies are complex neuron loadings U_k. A hidden state
        is then decoded to s by matched filter: argmax_s Σ_k Re(c_k e^{-iks}).
        """
        grid_a = np.repeat(np.arange(P), P).astype(float)
        grid_b = np.tile(np.arange(P), P).astype(float)
        h = self.hidden(grid_a, grid_b)
        sums = (grid_a + grid_b).astype(int) % P
        h_avg = np.zeros((P, h.shape[1]))
        for s in range(P):
            h_avg[s] = h[sums == s].mean(axis=0)
        dft = np.fft.fft(h_avg - h_avg.mean(axis=0), axis=0)
        power = (np.abs(dft[1:(P - 1) // 2 + 1]) ** 2).sum(axis=1)
        self.freqs = np.arange(1, (P - 1) // 2 + 1)  # all 48: sharpest peak
        self.top5 = np.sort(np.argsort(power)[::-1][:5] + 1)
        self.U = dft[self.freqs]                     # (48, H) complex templates
        self.s_grid = np.arange(0, P, 0.02)
        self.E = np.exp(-2j * np.pi * np.outer(self.freqs, self.s_grid) / P)

    def step_output(self, z):
        """Iterate through the class readout (softmax circular mean)."""
        logits = self.hidden(z, z) @ self.W2.T + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return (np.angle(probs @ self.class_phasor) * P / (2 * np.pi)) % P

    def decode_softmax(self, za, zb):
        logits = self.hidden(za, zb) @ self.W2.T + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return (np.angle(probs @ self.class_phasor) * P / (2 * np.pi)) % P

    def step_hidden(self, z):
        """Iterate at the representation level (matched-filter phase)."""
        c = self.hidden(z, z) @ self.U.conj().T      # (B, 5)
        score = (c @ self.E).real                    # (B, S)
        i = score.argmax(axis=1)
        # quadratic sub-grid refinement of the peak (circular neighbors)
        S = len(self.s_grid)
        ym = score[np.arange(len(i)), (i - 1) % S]
        y0 = score[np.arange(len(i)), i]
        yp = score[np.arange(len(i)), (i + 1) % S]
        denom = ym - 2 * y0 + yp
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (ym - yp) / denom, 0.0)
        return (self.s_grid[i] + np.clip(shift, -0.5, 0.5) * 0.02) % P


def circ_dist(u, v):
    d = np.abs(u - v) % P
    return np.minimum(d, P - d)


def main():
    rng = np.random.default_rng(0)
    sd = torch.load(CKPT_DIR / "ckpt_e7499.pt", map_location="cpu")
    net = ContinuousNet(sd)
    print(f"dominant frequencies: {list(net.top5)} (decoding with all 48)")

    # --- 1. The readout staircase ----------------------------------------
    z_fine = np.linspace(10.0, 13.0, 400)
    stair = net.decode_softmax(z_fine, z_fine)

    # --- 2. Separation growth: output level vs representation level -------
    N_PAIRS, N_ITER, DELTA0 = 400, 14, 1e-3
    ns = np.arange(N_ITER + 1)
    seps = {}
    for label, step in [("output", net.step_output), ("hidden", net.step_hidden)]:
        za = rng.uniform(0, P, N_PAIRS)
        zb = (za + DELTA0) % P
        sep = np.zeros((N_ITER + 1, N_PAIRS))
        sep[0] = circ_dist(za, zb)
        for n in range(1, N_ITER + 1):
            za, zb = step(za), step(zb)
            sep[n] = circ_dist(za, zb)
        seps[label] = sep

    floor = 1e-5   # display floor for merged orbits
    mean_log_hidden = np.log(np.maximum(seps["hidden"], floor)).mean(axis=1)
    mean_log_output = np.log(np.maximum(seps["output"], floor)).mean(axis=1)
    fit_mask = (mean_log_hidden > np.log(2 * DELTA0)) & (mean_log_hidden < np.log(3.0))
    fit_mask[0] = True
    lam = np.polyfit(ns[fit_mask], mean_log_hidden[fit_mask], 1)[0]

    # Period of the output-level integer dynamics (ord(2) mod 97)
    z, period = 1, 0
    while True:
        z, period = (2 * z) % P, period + 1
        if z == 1:
            break

    # --- 3. The fan (representation level) --------------------------------
    N_FAN, N_FAN_IT = 970, 8
    fan = np.zeros((N_FAN_IT + 1, N_FAN))
    fan[0] = np.linspace(0, P, N_FAN, endpoint=False)
    z = fan[0].copy()
    for n in range(1, N_FAN_IT + 1):
        z = net.step_hidden(z)
        fan[n] = z

    # --- Figure -----------------------------------------------------------
    fig = plt.figure(figsize=(12.8, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    ax.plot(z_fine, stair, color=ORANGE, linewidth=2, label="class readout")
    ax.plot(z_fine, (2 * z_fine) % P, color=MUTED, linewidth=1.5,
            linestyle="--", label="2z mod 97")
    ax.set_title("The readout is a quantizer\n(softmax decode of net(z, z))",
                 color=INK, loc="left")
    ax.set_xlabel("z")
    ax.set_ylabel("decoded output")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[1])
    style_axes(ax)
    ax.plot(ns, mean_log_output / LN2, color=ORANGE, linewidth=2,
            label=f"output level (→ period-{period} lattice orbit)")
    ax.plot(ns, mean_log_hidden / LN2, color=BLUE, linewidth=2,
            label="representation level")
    ax.plot(ns, (np.log(DELTA0) + LN2 * ns) / LN2, color=MUTED,
            linewidth=1.5, linestyle="--", label="slope ln 2 (exact chaos)")
    ax.axhline(np.log(P / 4) / LN2, color=GRID, linewidth=1)
    ax.set_title(f"Nearby orbits: λ = {lam:.2f} (ln 2 ≈ 0.69)",
                 color=INK, loc="left")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("log2 mean separation (classes)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[2])
    ax.imshow(fan, origin="upper", cmap=cm.get_cmap("twilight"), vmin=0, vmax=P,
              extent=[0, P, N_FAN_IT + 0.5, -0.5], aspect="auto",
              interpolation="nearest")
    ax.set_title("The fan: zₙ under representation-level iteration\n"
                 "(bands halve every row — the binary shift)",
                 color=INK, loc="left")
    ax.set_xlabel("z₀")
    ax.set_ylabel("iteration n")
    ax.set_yticks(range(0, N_FAN_IT + 1, 2))
    ax.tick_params(length=0)

    fig.suptitle("z ← net(z, z): chaos in the code, error correction at the readout",
                 color=INK, fontsize=12, x=0.02, y=1.06, ha="left")
    fig.savefig(CKPT_DIR / "fig_iterated_map.png", dpi=150, bbox_inches="tight")

    print(f"fitted Lyapunov exponent (representation level): {lam:.4f}  "
          f"(ln 2 = {LN2:.4f})")
    print(f"output-level integer orbit period: {period}")
    merged = (seps["output"][-1] <= floor).mean()
    print(f"output-level orbit pairs merged by n={N_ITER}: {merged:.0%}")
    print(f"figure written to {CKPT_DIR / 'fig_iterated_map.png'}")


if __name__ == "__main__":
    main()
