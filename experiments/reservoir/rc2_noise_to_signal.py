"""
RC-2: Noise-to-Signal Generalization via Behavioral Imprinting

Core question: if you train a reservoir exclusively on random I/O pairs sampled
from a target model, does it generalize to coherent inputs — ones the target was
designed for?

Setup:
  - Target: small GRU trained to near-perfect next-char prediction on a k-th order
    Markov chain over a V-char alphabet. Weights frozen. This is the function f.
  - Reservoir: Echo State Network (fixed random weights, ridge regression readout).
  - Three conditions trained independently:
      Random   — uniform random char sequences → f(input)
      Coherent — sequences sampled from the grammar → f(input)
      Mixed    — 80% random, 20% coherent
  - Evaluation: fixed held-out set of coherent sequences.

Measurements:
  - Cross-entropy on held-out coherent set vs N training sequences
  - KL divergence against ground truth Markov matrix (bypasses GRU approximation)
  - Learning curve: does generalization emerge gradually or as a phase transition?
  - Complexity probe: PCA on GRU output distributions → effective dimensionality

See experiment_noise_to_signal.md for full design rationale.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("PyTorch required.  conda run -n ro-framework pip install torch")
    raise

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

V = 10       # alphabet size
K = 3        # Markov order  (effective complexity ≈ V^K contexts, but output dim = V)
SEQ_LEN = 24 # chars per training sequence
SKIP = K     # skip first K positions (partial context)
PAIRS_PER_SEQ = SEQ_LEN - SKIP  # = 21 training pairs per sequence

RESERVOIR_SIZES = [64, 256, 1024]
SPECTRAL_RADIUS = 0.97
RIDGE_LAMBDA = 1e-4   # low regularisation — reservoir states ∈ [-1,1] so signal dominates

# Training sequences to sweep.  Pairs = N × PAIRS_PER_SEQ.
N_SWEEP = [25, 75, 250, 750, 2500, 7500]

N_TEST_SEQS = 500   # held-out coherent evaluation sequences
GRU_HIDDEN = 128    # large enough to memorise all V^K = 1000 contexts

SEED = 42


# ---------------------------------------------------------------------------
# Markov chain ground truth
# ---------------------------------------------------------------------------


class MarkovChain:
    """k-th order Markov chain over an alphabet of size V.

    Transition matrix T[ctx_idx, next_char] is independently Dirichlet-sampled
    for each context, giving a diverse and structured target function.
    """

    def __init__(self, V: int = 10, k: int = 3, concentration: float = 0.5, seed: int = 42):
        self.V = V
        self.k = k
        rng = np.random.default_rng(seed)
        n_ctx = V**k
        # concentration < 1 → peaked/sparse distributions (harder to learn from noise)
        self.T = np.array([rng.dirichlet(np.ones(V) * concentration) for _ in range(n_ctx)])
        self.powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)

    # --- context utilities ---

    def ctx_idx(self, context: np.ndarray) -> int:
        return int(np.dot(context[-self.k :], self.powers))

    def distribution(self, context: np.ndarray) -> np.ndarray:
        """P(next | last-k chars of context)."""
        return self.T[self.ctx_idx(context)]

    # --- sequence generation ---

    def generate_sequence(self, length: int, rng: np.random.Generator) -> np.ndarray:
        seq = list(rng.integers(0, self.V, size=self.k))
        while len(seq) < length:
            nxt = rng.choice(self.V, p=self.T[self.ctx_idx(np.array(seq[-self.k :]))])
            seq.append(int(nxt))
        return np.array(seq, dtype=np.int64)

    def generate_batch(self, B: int, length: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.stack([self.generate_sequence(length, rng) for _ in range(B)])

    # --- ground truth stats ---

    def entropy(self) -> float:
        """Expected per-step entropy under uniform context distribution."""
        return float(np.mean([-np.sum(row * np.log(row + 1e-12)) for row in self.T]))

    def ground_truth_dists(self, seqs: np.ndarray) -> np.ndarray:
        """Return Markov ground-truth distributions for positions [K:] in each sequence.

        seqs: (B, L)
        Returns: (B, L-K, V) — P(c_t | c_{t-K}...c_{t-1}) for t in K..L-1
        """
        B, L = seqs.shape
        out = np.zeros((B, L - self.k, self.V))
        for i in range(B):
            for t in range(self.k, L):
                out[i, t - self.k] = self.T[self.ctx_idx(seqs[i, t - self.k : t])]
        return out


# ---------------------------------------------------------------------------
# GRU target model
# ---------------------------------------------------------------------------


class MarkovGRU(nn.Module):
    """Small autoregressive GRU approximating a k-th order Markov chain.

    Convention: input to GRU at position t is one-hot(c_{t-1}), with a
    zero vector for t=0 (start-of-sequence token). Output at position t is
    P(c_t | c_0...c_{t-1}).
    """

    def __init__(self, V: int, hidden_dim: int = 64):
        super().__init__()
        self.V = V
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(V, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, V)

    def _make_input(self, chars_oh: torch.Tensor) -> torch.Tensor:
        """Shift: prepend SOS (zeros) and drop last char."""
        B, L, V = chars_oh.shape
        sos = torch.zeros(B, 1, V, device=chars_oh.device)
        return torch.cat([sos, chars_oh[:, :-1]], dim=1)  # (B, L, V)

    def forward(self, chars_oh: torch.Tensor) -> torch.Tensor:
        """chars_oh: (B, L, V)  →  logits: (B, L, V)"""
        out, _ = self.gru(self._make_input(chars_oh))
        return self.head(out)

    @torch.no_grad()
    def get_distributions_batch(self, seqs: np.ndarray) -> np.ndarray:
        """seqs: (B, L) int → (B, L, V) softmax dists, P(c_t | c_{<t})"""
        t = torch.tensor(seqs, dtype=torch.long, device=DEVICE)
        oh = F.one_hot(t, self.V).float()
        logits = self(oh)
        return F.softmax(logits, dim=-1).cpu().numpy()


def train_gru(gru: MarkovGRU, markov: MarkovChain, n_train: int = 10000,
              seq_len: int = 32, lr: float = 3e-3, max_epochs: int = 800,
              tol: float = 0.03, seed: int = 0) -> float:
    """Train GRU until validation CE ≤ ground-truth entropy + tol.

    Uses ReduceLROnPlateau to handle stagnation without causing divergence.
    Saves the best checkpoint and returns the best val CE achieved.
    """
    train_seqs = markov.generate_batch(n_train, seq_len, seed=seed)
    val_seqs = markov.generate_batch(1000, seq_len, seed=seed + 99999)
    target_ce = markov.entropy() + tol

    def to_tensors(seqs):
        t = torch.tensor(seqs, dtype=torch.long, device=DEVICE)
        oh = F.one_hot(t, markov.V).float()
        return oh, t

    x_val, y_val = to_tensors(val_seqs)
    opt = torch.optim.Adam(gru.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=30, min_lr=1e-5
    )

    rng = np.random.default_rng(seed)
    B = 256
    best_val_ce = float("inf")
    best_state = None

    for epoch in range(max_epochs):
        perm = rng.permutation(n_train)
        gru.train()
        for start in range(0, n_train, B):
            idx = perm[start : start + B]
            x, y = to_tensors(train_seqs[idx])
            logits = gru(x)
            loss = F.cross_entropy(logits.reshape(-1, markov.V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        gru.eval()
        with torch.no_grad():
            logits_v = gru(x_val)
            val_ce = F.cross_entropy(logits_v.reshape(-1, markov.V), y_val.reshape(-1)).item()
        scheduler.step(val_ce)

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_state = {k: v.clone() for k, v in gru.state_dict().items()}

        if epoch % 100 == 0 or epoch == max_epochs - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    GRU epoch {epoch:>4}: val_CE={val_ce:.4f}  best={best_val_ce:.4f}  target≤{target_ce:.4f}  lr={lr_now:.2e}")
            if best_val_ce <= target_ce:
                print(f"    Converged (best CE ≤ target).")
                break
            if lr_now <= 1.1e-5:
                print(f"    LR at floor, stopping.")
                break

    # Restore best checkpoint
    if best_state is not None:
        gru.load_state_dict(best_state)
    return best_val_ce


# ---------------------------------------------------------------------------
# Echo State Network for streaming character input
# ---------------------------------------------------------------------------


class SequenceESN:
    """Fixed-weight Echo State Network for sequential char-level prediction.

    Processes chars one at a time.  Only the linear readout is trained
    (via ridge regression on the Gram matrix — no autograd needed).
    """

    def __init__(self, V: int, N: int, spectral_radius: float = 0.97, seed: int = 0):
        self.V = V
        self.N = N  # reservoir size
        rng = np.random.default_rng(seed)

        # Input weights: (N, V)
        W_in = rng.standard_normal((N, V)) * 0.1
        self.W_in = W_in.astype(np.float32)

        # Recurrent weights: (N, N), rescaled to target spectral radius
        W_res = rng.standard_normal((N, N)).astype(np.float32)
        sr = float(np.max(np.abs(np.linalg.eigvals(W_res))))
        if sr > 1e-8:
            W_res *= spectral_radius / sr
        self.W_res = W_res

        self.bias = (rng.standard_normal(N) * 0.1).astype(np.float32)
        self.W_out: Optional[np.ndarray] = None  # (N+1, V) — set by fit_readout

    def process_batch(self, seqs: np.ndarray) -> np.ndarray:
        """Process B sequences through the fixed reservoir.

        seqs: (B, L) integer array
        Returns: (B, L, N) — reservoir state BEFORE each character.
                 state[:,t,:] encodes chars[:,0:t] and predicts chars[:,t].
        """
        B, L = seqs.shape
        H = np.zeros((B, self.N), dtype=np.float32)
        states = np.zeros((B, L, self.N), dtype=np.float32)

        for t in range(L):
            states[:, t, :] = H  # record state BEFORE processing c_t (to predict c_t)
            X = np.zeros((B, self.V), dtype=np.float32)
            X[np.arange(B), seqs[:, t]] = 1.0
            H = np.tanh(X @ self.W_in.T + H @ self.W_res.T + self.bias)

        return states  # (B, L, N)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Map reservoir states to distributions via learned readout.

        states: (..., N)
        Returns: (..., V) softmax probabilities
        """
        assert self.W_out is not None, "Call fit_readout first."
        flat = states.reshape(-1, self.N)
        bias_col = np.ones((len(flat), 1), dtype=np.float32)
        raw = np.concatenate([flat, bias_col], axis=1) @ self.W_out  # (..., V)
        # Stable softmax
        raw -= raw.max(axis=-1, keepdims=True)
        exp = np.exp(raw)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs.reshape(states.shape[:-1] + (self.V,))


# ---------------------------------------------------------------------------
# Ridge regression via Gram matrix (memory-efficient for large N)
# ---------------------------------------------------------------------------


def fit_readout(esn: SequenceESN, seqs: np.ndarray, targets: np.ndarray,
                lam: float = 1e-2, skip: int = K) -> None:
    """Fit ESN readout using ridge regression on the Gram matrix.

    seqs:    (B_total, L) — training sequences
    targets: (B_total, L, V) — target probability distributions (GRU softmax or Markov)

    IMPORTANT: We regress on log(target) and predict with softmax(W_out @ state).
    This is the correct approach because softmax is the natural link function for
    distributions. Regressing directly on probabilities and then applying softmax
    collapses small differences (e.g. [0.15, 0.09] → softmax → [0.11, 0.10]).
    Regressing on log-probs preserves the signal: softmax(log(p)) = p exactly.

    The Gram matrix G = X^T X accumulates in batches → O(N²) memory, not O(pairs×N).
    """
    N = esn.N
    dim = N + 1  # +1 for bias
    G = np.zeros((dim, dim), dtype=np.float64)
    C = np.zeros((dim, esn.V), dtype=np.float64)

    chunk = 256
    B_total = len(seqs)
    for start in range(0, B_total, chunk):
        s = seqs[start : start + chunk]
        t = targets[start : start + chunk]

        states = esn.process_batch(s)[:, skip:, :]  # (b, L-skip, N)
        tgt = t[:, skip:, :]                         # (b, L-skip, V)

        b, Ls, _ = states.shape
        X = states.reshape(b * Ls, N).astype(np.float64)

        # Key fix: regress on log-probabilities, not raw probabilities.
        # softmax(log(p)) = p exactly, so at prediction time softmax gives
        # the right distribution rather than near-uniform.
        log_tgt = np.log(np.maximum(tgt.reshape(b * Ls, esn.V), 1e-8)).astype(np.float64)

        Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)  # (n_pairs, N+1)
        G += Xb.T @ Xb
        C += Xb.T @ log_tgt

    # Solve: W_out = (G + λI)^{-1} C  — W_out maps state → log-prob space
    G[np.arange(dim), np.arange(dim)] += lam
    esn.W_out = np.linalg.solve(G, C).astype(np.float32)  # (N+1, V)


# ---------------------------------------------------------------------------
# Sequence generation for the three conditions
# ---------------------------------------------------------------------------


def _random_seqs(B: int, L: int, V: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, V, size=(B, L), dtype=np.int64)


def make_training_seqs(condition: str, N: int, markov: MarkovChain,
                       seq_len: int = SEQ_LEN, seed: int = 1) -> np.ndarray:
    """Return (N, seq_len) integer array of training sequences for a condition."""
    if condition in ("random", "oracle_random"):
        return _random_seqs(N, seq_len, markov.V, seed)
    elif condition in ("coherent", "oracle_coherent"):
        return markov.generate_batch(N, seq_len, seed=seed)
    elif condition == "mixed":
        n_random = int(N * 0.8)
        n_coherent = N - n_random
        rand_part = _random_seqs(n_random, seq_len, markov.V, seed)
        coh_part = markov.generate_batch(n_coherent, seq_len, seed=seed + 77777)
        return np.concatenate([rand_part, coh_part], axis=0)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


def get_targets(condition: str, seqs: np.ndarray, gru: MarkovGRU,
                markov: MarkovChain) -> np.ndarray:
    """Return (B, L, V) target distributions for training.

    For standard conditions: GRU softmax outputs (behavioral imprinting).
    For oracle conditions: ground truth Markov distributions (diagnostic —
      removes GRU approximation error as a confound).
    """
    if condition.startswith("oracle"):
        # Ground truth Markov: pad positions 0..K-1 with uniform (will be skipped)
        B, L = seqs.shape
        targets = np.full((B, L, markov.V), 1.0 / markov.V)
        for i in range(B):
            for t in range(markov.k, L):
                targets[i, t] = markov.distribution(seqs[i, t - markov.k : t])
        return targets
    else:
        return gru.get_distributions_batch(seqs)  # (B, L, V)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Mean per-position KL(p || q).  p and q are (..., V)."""
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return float(np.mean(np.sum(p * np.log(p / q), axis=-1)))


def _ce(probs: np.ndarray, chars: np.ndarray, eps: float = 1e-9) -> float:
    """Mean cross-entropy of predicted distributions on true chars."""
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_chars = chars.reshape(-1)
    selected = flat_probs[np.arange(len(flat_chars)), flat_chars]
    return float(-np.mean(np.log(np.clip(selected, eps, 1.0))))


def _accuracy(esn_probs: np.ndarray, markov_dists: np.ndarray) -> float:
    """Fraction of positions where argmax(ESN) == argmax(Markov ground truth)."""
    return float(np.mean(esn_probs.reshape(-1, esn_probs.shape[-1]).argmax(axis=1)
                         == markov_dists.reshape(-1, markov_dists.shape[-1]).argmax(axis=1)))


def evaluate(esn: SequenceESN, gru: MarkovGRU, markov: MarkovChain,
             test_seqs: np.ndarray) -> dict:
    """Compute all metrics on the held-out coherent test set.

    Returns dict with keys: ce, kl_gru, kl_markov, accuracy, ce_uniform, ce_gru
    """
    states = esn.process_batch(test_seqs)[:, SKIP:, :]   # (M, L-K, N)
    esn_probs = esn.predict(states)                        # (M, L-K, V)
    gru_dists = gru.get_distributions_batch(test_seqs)[:, SKIP:, :]  # (M, L-K, V)
    markov_dists = markov.ground_truth_dists(test_seqs)    # (M, L-K, V)
    chars = test_seqs[:, SKIP:]                            # (M, L-K)

    return {
        "ce":          _ce(esn_probs, chars),
        "kl_gru":      _kl(gru_dists, esn_probs),       # KL(GRU || ESN)
        "kl_markov":   _kl(markov_dists, esn_probs),    # KL(Markov || ESN) — primary
        "accuracy":    _accuracy(esn_probs, markov_dists),
        "ce_uniform":  np.log(markov.V),                 # baseline: uniform predictor
        "ce_gru":      _ce(gru_dists, chars),            # teacher CE (approximate ceiling)
    }


# ---------------------------------------------------------------------------
# Complexity probe — PCA on GRU output distributions
# ---------------------------------------------------------------------------


def complexity_probe(gru: MarkovGRU, markov: MarkovChain, n_seqs: int = 3000,
                     seq_len: int = SEQ_LEN) -> dict:
    """Estimate intrinsic dimensionality of the target model's I/O manifold.

    Samples output distributions from random inputs (uniform coverage of input
    space) and from coherent inputs (stationary distribution), then runs PCA
    to find effective dimensionality.

    The claim: effective dim ≈ V-1 = 9 (the simplex dimension of the output),
    regardless of V^K = 1000 nominal contexts.  If true, N_crit should scale
    with ~9 rather than 1000.
    """
    results = {}
    for label, seqs in [
        ("random",   _random_seqs(n_seqs, seq_len, markov.V, seed=SEED)),
        ("coherent", markov.generate_batch(n_seqs, seq_len, seed=SEED + 1)),
    ]:
        dists = gru.get_distributions_batch(seqs)[:, SKIP:, :]  # (B, L-K, V)
        flat = dists.reshape(-1, markov.V).astype(np.float64)    # (n_samples, V)

        centered = flat - flat.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var = s**2
        cumvar = np.cumsum(var) / (var.sum() + 1e-12)

        d_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        d_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        d_99 = int(np.searchsorted(cumvar, 0.99)) + 1

        results[label] = {
            "d_90": d_90, "d_95": d_95, "d_99": d_99,
            "singular_values": s.tolist(),
            "var_fractions": (var / var.sum()).tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_sweep(condition: str, esn_size: int, gru: MarkovGRU, markov: MarkovChain,
              test_seqs: np.ndarray, n_sweep: List[int] = N_SWEEP) -> List[dict]:
    """For each N in n_sweep: train ESN readout, evaluate, return metric list."""
    esn = SequenceESN(V=markov.V, N=esn_size, spectral_radius=SPECTRAL_RADIUS, seed=SEED)
    metrics_list = []

    for N in n_sweep:
        seqs = make_training_seqs(condition, N, markov, seq_len=SEQ_LEN, seed=SEED + 7)
        targets = get_targets(condition, seqs, gru, markov)  # (N, L, V)

        fit_readout(esn, seqs, targets, lam=RIDGE_LAMBDA, skip=SKIP)

        m = evaluate(esn, gru, markov, test_seqs)
        m["N"] = N
        m["pairs"] = N * PAIRS_PER_SEQ
        metrics_list.append(m)

    return metrics_list


# ---------------------------------------------------------------------------
# Results printing and plotting
# ---------------------------------------------------------------------------

CONDITIONS = ["random", "coherent", "mixed", "oracle_random", "oracle_coherent"]
COND_LABEL = {
    "random":          "Random (GRU)",
    "coherent":        "Coherent (GRU)",
    "mixed":           "Mixed 80/20 (GRU)",
    "oracle_random":   "Random (Oracle)",
    "oracle_coherent": "Coherent (Oracle)",
}
COLORS = {
    "random":          "#e74c3c",
    "coherent":        "#2ecc71",
    "mixed":           "#3498db",
    "oracle_random":   "#e67e22",
    "oracle_coherent": "#27ae60",
}


def print_table(results: dict, markov: MarkovChain) -> None:
    """Print a compact results table across all conditions and ESN sizes."""
    header = f"{'Condition':<14} {'N(ESN)':<10} {'N seqs':>7} | {'CE':>7} {'KL(Mkov)':>9} {'KL(GRU)':>8} {'Acc':>6}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for cond in CONDITIONS:
        for esn_size in RESERVOIR_SIZES:
            key = (cond, esn_size)
            if key not in results:
                continue
            for m in results[key]:
                label = f"{COND_LABEL[cond]:<14} {esn_size:<10}"
                row = (
                    f"{label} {m['N']:>7} | "
                    f"{m['ce']:>7.4f} "
                    f"{m['kl_markov']:>9.4f} "
                    f"{m['kl_gru']:>8.4f} "
                    f"{m['accuracy']:>6.3f}"
                )
                print(row)
        print()

    print(f"  Baselines — CE(uniform)={np.log(markov.V):.4f}  CE(GRU≈target)={results.get('ce_gru', '?')}")
    print(sep)


def plot_results(results: dict, markov: MarkovChain, save_path: str = "rc2_results.png") -> None:
    """Plot learning curves (KL vs Markov, CE) across conditions per ESN size."""
    if not HAS_PLT:
        return

    fig, axes = plt.subplots(len(RESERVOIR_SIZES), 2, figsize=(12, 4 * len(RESERVOIR_SIZES)))
    axes = np.array(axes).reshape(len(RESERVOIR_SIZES), 2)

    for ri, esn_size in enumerate(RESERVOIR_SIZES):
        ax_kl, ax_ce = axes[ri]

        for cond in CONDITIONS:
            key = (cond, esn_size)
            if key not in results:
                continue
            ms = results[key]
            xs = [m["N"] for m in ms]
            kl_vals = [m["kl_markov"] for m in ms]
            ce_vals = [m["ce"] for m in ms]

            kwargs = dict(label=COND_LABEL[cond], color=COLORS[cond], marker="o", linewidth=2)
            ax_kl.semilogx(xs, kl_vals, **kwargs)
            ax_ce.semilogx(xs, ce_vals, **kwargs)

        # Baselines
        for ax in (ax_kl, ax_ce):
            ax.axhline(np.log(markov.V), ls=":", color="gray", label="Uniform")
        ax_ce.axhline(results.get("ce_gru", 0), ls="--", color="black", alpha=0.5, label="GRU (target)")

        ax_kl.set_title(f"KL(Markov ‖ ESN) — N={esn_size}")
        ax_ce.set_title(f"Cross-entropy — N={esn_size}")
        for ax in (ax_kl, ax_ce):
            ax.set_xlabel("N training sequences")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"RC-2: Noise-to-Signal  (V={V}, k={K}, SR={SPECTRAL_RADIUS}, λ={RIDGE_LAMBDA})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")


def print_complexity_probe(probe: dict, markov: MarkovChain) -> None:
    print("=" * 60)
    print("Complexity Probe (PCA on GRU output distributions)")
    print("=" * 60)
    print(f"  Alphabet V={markov.V}, Markov order k={markov.k}")
    print(f"  Nominal input space: V^k = {markov.V**markov.k} contexts")
    print(f"  Expected output simplex dim: V-1 = {markov.V - 1}")
    print()
    for label in ("random", "coherent"):
        d = probe[label]
        vf = d["var_fractions"]
        top5 = "  ".join(f"{v:.1%}" for v in vf[:5])
        print(f"  {label.capitalize()} inputs:")
        print(f"    d_90={d['d_90']}  d_95={d['d_95']}  d_99={d['d_99']}")
        print(f"    Top-5 singular value fractions: {top5}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    print("=" * 60)
    print("RC-2: Noise-to-Signal Generalization via Behavioral Imprinting")
    print("=" * 60)
    print(f"  V={V}, k={K}, ESN sizes={RESERVOIR_SIZES}")
    print(f"  N sweep={N_SWEEP}")
    print(f"  Seq len={SEQ_LEN}, skip={SKIP} ({PAIRS_PER_SEQ} pairs/seq)")
    print(f"  λ={RIDGE_LAMBDA}, SR={SPECTRAL_RADIUS}")
    print(f"  Device: {DEVICE}")
    print()

    # 1. Build ground truth
    print("Building Markov chain...")
    markov = MarkovChain(V=V, k=K, concentration=0.5, seed=SEED)
    print(f"  Ground truth entropy: {markov.entropy():.4f} nats")
    print(f"  Uniform baseline:     {np.log(V):.4f} nats")
    print()

    # 2. Train GRU target model
    print("Training GRU target model...")
    gru = MarkovGRU(V=V, hidden_dim=GRU_HIDDEN).to(DEVICE)
    gru_ce = train_gru(gru, markov, n_train=8000, seq_len=32, max_epochs=600)
    gru.eval()
    print(f"  GRU final val CE: {gru_ce:.4f}  (ground truth: {markov.entropy():.4f})")
    print()

    # 3. Complexity probe
    print("Running complexity probe...")
    probe = complexity_probe(gru, markov)
    print_complexity_probe(probe, markov)

    # 4. Generate fixed test set (coherent sequences)
    print(f"Generating {N_TEST_SEQS} held-out coherent test sequences...")
    test_seqs = markov.generate_batch(N_TEST_SEQS, SEQ_LEN, seed=SEED + 999999)
    print()

    # Compute GRU CE on test set for baseline reference
    gru_test_dists = gru.get_distributions_batch(test_seqs)[:, SKIP:, :]
    test_chars = test_seqs[:, SKIP:]
    ce_gru_test = _ce(gru_test_dists, test_chars)

    # 5. Main sweep
    all_results: dict = {"ce_gru": ce_gru_test}

    for esn_size in RESERVOIR_SIZES:
        print(f"\n{'─'*60}")
        print(f"ESN reservoir size: {esn_size}")
        print(f"{'─'*60}")

        for cond in CONDITIONS:
            print(f"\n  Condition: {COND_LABEL[cond]}")
            t_cond = time.time()

            metrics_list = run_sweep(cond, esn_size, gru, markov, test_seqs)
            all_results[(cond, esn_size)] = metrics_list

            print(f"  {'N':>7} | {'CE':>7} {'KL(Mkov)':>9} {'KL(GRU)':>8} {'Acc':>6} {'Pairs':>8}")
            print(f"  {'─'*55}")
            for m in metrics_list:
                print(
                    f"  {m['N']:>7} | {m['ce']:>7.4f} {m['kl_markov']:>9.4f} "
                    f"{m['kl_gru']:>8.4f} {m['accuracy']:>6.3f} {m['pairs']:>8}"
                )

            print(f"  (elapsed: {time.time()-t_cond:.1f}s)")

    # 6. Print summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print_table(all_results, markov)

    # 7. Analysis
    print("\n--- Analysis ---")
    h = markov.entropy()
    print(f"  Ground truth Markov entropy: {h:.4f}")
    print(f"  GRU CE on coherent test:     {ce_gru_test:.4f}  (gap = {ce_gru_test - h:.4f})")
    print(f"  Uniform baseline CE:         {np.log(V):.4f}")
    print()

    best_esn = RESERVOIR_SIZES[-1]
    for cond in CONDITIONS:
        key = (cond, best_esn)
        if key not in all_results:
            continue
        ms = all_results[key]
        final = ms[-1]
        best_kl = min(m["kl_markov"] for m in ms)
        print(f"  {COND_LABEL[cond]:<26}: final_CE={final['ce']:.4f}  best_KL={best_kl:.4f}  final_acc={final['accuracy']:.3f}")

    print()

    # Core hypothesis test
    random_ms = all_results.get(("random", best_esn))
    coherent_ms = all_results.get(("coherent", best_esn))
    oracle_rand_ms = all_results.get(("oracle_random", best_esn))
    oracle_coh_ms = all_results.get(("oracle_coherent", best_esn))

    if random_ms and coherent_ms:
        rand_kl = min(m["kl_markov"] for m in random_ms)
        coh_kl = min(m["kl_markov"] for m in coherent_ms)
        ratio = rand_kl / (coh_kl + 1e-6)
        print(f"  KL ratio GRU(random / coherent):    {ratio:.2f}x")

    if oracle_rand_ms and oracle_coh_ms:
        orand_kl = min(m["kl_markov"] for m in oracle_rand_ms)
        ocoh_kl = min(m["kl_markov"] for m in oracle_coh_ms)
        o_ratio = orand_kl / (ocoh_kl + 1e-6)
        print(f"  KL ratio Oracle(random / coherent): {o_ratio:.2f}x")

    if oracle_rand_ms and random_ms:
        orand_kl = min(m["kl_markov"] for m in oracle_rand_ms)
        rand_kl = min(m["kl_markov"] for m in random_ms)
        teacher_ratio = rand_kl / (orand_kl + 1e-6)
        print(f"  KL ratio GRU_random / Oracle_random (teacher quality impact): {teacher_ratio:.2f}x")

    print()
    print("  Interpretation:")
    if random_ms and coherent_ms:
        r = min(m["kl_markov"] for m in random_ms) / (min(m["kl_markov"] for m in coherent_ms) + 1e-6)
        if r < 1.5:
            print("  → Random generalizes comparably to coherent (< 1.5x KL gap).")
            print("    SUPPORTS hypothesis: random I/O samples encode sufficient structure.")
        elif r < 3.0:
            print("  → Random is moderately worse than coherent (1.5x–3x KL gap).")
            print("    PARTIAL SUPPORT: structure is recoverable but less efficiently.")
        else:
            print("  → Random significantly worse than coherent (> 3x KL gap).")
            print("    DOES NOT SUPPORT hypothesis at this scale.")

    # 8. Plot
    import os
    plot_path = os.path.join(os.path.dirname(__file__), "rc2_results.png")
    plot_results(all_results, markov, save_path=plot_path)

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
