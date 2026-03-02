"""
RC-1: Echo State Network — Modular Addition

First reservoir computing experiment for the RO framework. Tests whether RC
provides a cleaner substrate for interpretability than MLPs, using the same
modular addition task (a + b mod 97) from Phase 8.

Key structural difference from the MLP experiment:
  - Reservoir weights (W_in, W_res, bias) are FIXED random — never trained
  - Only the readout head trains → the readout IS the knowledge
  - Inputs are processed as a short sequence: first a, then b
  - The reservoir gets several recurrent settle steps per symbol

This is now a true ESN-style setup rather than a feedforward random-feature
baseline. The recurrent update is:
  h_t = tanh(W_in @ x_t + W_res @ h_{t-1} + b)
with x_1 = one_hot(a) and x_2 = one_hot(b).

Parts:
  1. Ridge regression baseline — linear readout capacity check
  2. SGD readout training — nonlinear readout with K trajectory over epochs
  3. Recurrent spectral analysis

Requires: PyTorch (pip install torch)
Runtime:  ~40s on GPU, several minutes on CPU for the full p=97 run
"""

import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("This experiment requires PyTorch. Install with: pip install torch")
    sys.exit(1)

from ro_framework import Observer, PolarDoF, State
from ro_framework.knowledge.tracker import KnowledgeTracker
from ro_framework.observer.observer import ObservationPair

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Echo State Network
# ---------------------------------------------------------------------------


class EchoStateNetwork(nn.Module):
    """Echo State Network for modular addition.

    Fixed random weights (W_in, W_res, bias), trained nonlinear readout head.
    Inputs are processed sequentially: first a, then b.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        output_dim: int,
        spectral_radius: float = 0.99,
        input_scaling: float = 1.0,
        sparsity: float = 0.0,
        settle_steps: int = 5,
        readout_hidden_dim: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even: one half for a, one half for b")
        self.symbol_dim = input_dim // 2
        self.output_dim = output_dim
        self.settle_steps = max(1, settle_steps)
        self.readout_hidden_dim = readout_hidden_dim or reservoir_size

        rng = np.random.default_rng(seed)

        # Fixed input weights (buffer, not parameter)
        W_in = rng.standard_normal((reservoir_size, input_dim)) * input_scaling
        self.register_buffer("W_in", torch.tensor(W_in, dtype=torch.float32))

        # Fixed reservoir recurrent weights, scaled to spectral_radius
        W_res = rng.standard_normal((reservoir_size, reservoir_size))
        if sparsity > 0:
            mask = rng.random((reservoir_size, reservoir_size)) > sparsity
            W_res *= mask
        eigvals = np.linalg.eigvals(W_res)
        current_sr = np.max(np.abs(eigvals))
        if current_sr > 0:
            W_res = W_res * (spectral_radius / current_sr)
        self.register_buffer("W_res", torch.tensor(W_res, dtype=torch.float32))

        bias = rng.standard_normal(reservoir_size) * 0.1
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32))

        # Trainable readout head — the ONLY learned component
        self.readout_in = nn.Linear(reservoir_size, self.readout_hidden_dim, bias=True)
        self.readout_out = nn.Linear(self.readout_hidden_dim, output_dim, bias=True)

    def _step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Single ESN update."""
        return torch.tanh(x_t @ self.W_in[:, : x_t.shape[1]].T + h_prev @ self.W_res.T + self.bias)

    def _settle_symbol(self, x_t: torch.Tensor, h_prev: torch.Tensor, input_slice: slice) -> torch.Tensor:
        """Run several recurrent settle steps for one symbol input."""
        h = h_prev
        W_in_symbol = self.W_in[:, input_slice]
        for _ in range(self.settle_steps):
            h = torch.tanh(x_t @ W_in_symbol.T + h @ self.W_res.T + self.bias)
        return h

    def reservoir_state(self, x: torch.Tensor) -> torch.Tensor:
        """Compute final reservoir state after sequentially consuming a then b.

        Args:
            x: (batch, input_dim) one-hot concatenation of a and b.
        Returns:
            h: (batch, reservoir_size) reservoir activations.
        """
        x_a = x[:, : self.symbol_dim]
        x_b = x[:, self.symbol_dim :]
        h0 = torch.zeros(x.shape[0], self.reservoir_size, device=x.device, dtype=x.dtype)
        h1 = self._settle_symbol(x_a, h0, slice(0, self.symbol_dim))
        h2 = self._settle_symbol(x_b, h1, slice(self.symbol_dim, self.input_dim))
        return h2

    def readout_features_from_state(self, h: torch.Tensor) -> torch.Tensor:
        """Hidden features learned by the readout head."""
        return F.relu(self.readout_in(h))

    def logits_from_state(self, h: torch.Tensor) -> torch.Tensor:
        """Map reservoir states to output logits."""
        return self.readout_out(self.readout_features_from_state(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: reservoir state → readout logits."""
        return self.logits_from_state(self.reservoir_state(x))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_dataset(p: int, train_frac: float = 0.75, seed: int = 42):
    """Create train/test split with one-hot encoded inputs.

    Returns (train_x, train_y), (test_x, test_y) where:
      x: (N, 2*p) tensor — one_hot(a) concatenated with one_hot(b)
      y: (N,) long tensor — (a+b) mod p
    """
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)
    split = int(len(all_pairs) * train_frac)

    def to_tensors(pairs):
        a_idx = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        b_idx = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
        x_a = F.one_hot(a_idx, p).float()
        x_b = F.one_hot(b_idx, p).float()
        x = torch.cat([x_a, x_b], dim=1).to(DEVICE)
        y = ((a_idx + b_idx) % p).to(DEVICE)
        return x, y

    return to_tensors(all_pairs[:split]), to_tensors(all_pairs[split:])


# ---------------------------------------------------------------------------
# Sum-class averaging and Fourier analysis
# ---------------------------------------------------------------------------


def compute_sum_averaged_reservoir(esn: EchoStateNetwork, p: int) -> np.ndarray:
    """Reservoir states averaged by sum class s = (a+b) mod p.

    Returns: (p, reservoir_size) array — one row per sum class.
    """
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    x_a = F.one_hot(grid_a, p).float()
    x_b = F.one_hot(grid_b, p).float()
    x = torch.cat([x_a, x_b], dim=1)

    with torch.no_grad():
        h = esn.reservoir_state(x).cpu().numpy()
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    h_avg = np.zeros((p, esn.reservoir_size))
    for s in range(p):
        h_avg[s] = h[sums == s].mean(axis=0)
    return h_avg


def discover_top_frequencies(h_avg: np.ndarray, p: int, top_n: int = 5) -> list:
    """Discover which Fourier frequencies have most power in reservoir states.

    Returns list of (frequency_k, power_fraction) tuples, sorted by power.
    """
    dft = np.fft.fft(h_avg, axis=0)
    power = np.abs(dft) ** 2
    total_per_freq = power.sum(axis=1)
    max_k = (p - 1) // 2
    non_dc_total = total_per_freq[1 : max_k + 1].sum()

    results = []
    for k in range(1, max_k + 1):
        frac = total_per_freq[k] / non_dc_total if non_dc_total > 0 else 0
        results.append((k, frac))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Observer / KnowledgeTracker setup (same pattern as experiment 08)
# ---------------------------------------------------------------------------


def setup_observer_and_tracker(
    freq_indices: list,
    internal_size: int,
    max_features: int = 1,
    internal_prefix: str = "neuron",
    min_samples: int = 3,
):
    """Create Observer and KnowledgeTracker for Fourier feature tracking.

    External DoFs: Fourier features sin(2πk·s/p), cos(2πk·s/p).
    Internal DoFs: Reservoir neurons, logits, or other tracked units.
    """
    fourier_dofs = []
    for k in freq_indices:
        fourier_dofs.append(PolarDoF(name=f"sin_{k}", pole_negative=-1.0, pole_positive=1.0))
        fourier_dofs.append(PolarDoF(name=f"cos_{k}", pole_negative=-1.0, pole_positive=1.0))

    internal_dofs = [PolarDoF(name=f"{internal_prefix}_{j}") for j in range(internal_size)]

    class _DummyMapping:
        def __call__(self, state: State) -> State:
            return State(values={d: 0.0 for d in internal_dofs})

    observer = Observer(
        name="esn_tracker",
        internal_dofs=internal_dofs,
        external_dofs=fourier_dofs,
        world_model=_DummyMapping(),
        log_capacity=500,
    )

    tracker = KnowledgeTracker(
        observer,
        external_dofs=fourier_dofs,
        min_samples=min_samples,
        max_features=max_features,
    )
    return observer, tracker, fourier_dofs, internal_dofs


def populate_observations(observer, internal_avg, p, fourier_dofs, internal_dofs, freq_indices):
    """Populate observation log with sum-averaged (Fourier, internal) pairs."""
    observer.clear_memory()
    for s in range(p):
        ext_vals = {}
        for fi, k in enumerate(freq_indices):
            angle = 2 * np.pi * k * s / p
            ext_vals[fourier_dofs[2 * fi]] = float(np.sin(angle))
            ext_vals[fourier_dofs[2 * fi + 1]] = float(np.cos(angle))

        int_vals = {internal_dofs[j]: float(internal_avg[s, j]) for j in range(len(internal_dofs))}

        observer.observation_log.append(
            ObservationPair(
                external_state=State(values=ext_vals),
                internal_state=State(values=int_vals),
                timestamp=float(s),
            )
        )


# ---------------------------------------------------------------------------
# Linear baseline / capacity check
# ---------------------------------------------------------------------------


def ridge_regression(H_train, Y_train_onehot, H_test, Y_test, reg_lambda=1e-4):
    """Closed-form linear baseline on fixed reservoir states.

    W_out = (H^T H + λI)^{-1} H^T Y
    Returns (train_acc, test_acc, W_out).
    """
    N_res = H_train.shape[1]
    A = H_train.T @ H_train + reg_lambda * np.eye(N_res)
    W_out = np.linalg.solve(A, H_train.T @ Y_train_onehot)  # (N_res, p)

    train_pred = H_train @ W_out
    train_acc = (train_pred.argmax(axis=1) == Y_train_onehot.argmax(axis=1)).mean()

    test_pred = H_test @ W_out
    test_acc = (test_pred.argmax(axis=1) == Y_test.numpy()).mean()

    return train_acc, test_acc, W_out


def compute_sum_averaged_readout_features(esn: EchoStateNetwork, p: int) -> np.ndarray:
    """Average readout hidden features by sum class s = (a+b) mod p.

    This is the readout-dependent quantity that can change during training.
    Tracking K here is more informative than on final logits because the
    nonlinear readout head can express Fourier structure before the last
    class-composition layer makes it linearly separable.
    """
    grid_a = torch.arange(p, device=DEVICE).repeat_interleave(p)
    grid_b = torch.arange(p, device=DEVICE).repeat(p)
    x_a = F.one_hot(grid_a, p).float()
    x_b = F.one_hot(grid_b, p).float()
    x = torch.cat([x_a, x_b], dim=1)

    with torch.no_grad():
        features = esn.readout_features_from_state(esn.reservoir_state(x)).cpu().numpy()
    sums = ((grid_a + grid_b) % p).cpu().numpy()

    feature_avg = np.zeros((p, esn.readout_hidden_dim))
    for s in range(p):
        feature_avg[s] = features[sums == s].mean(axis=0)
    return feature_avg


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_header(freq_indices: list, tracked_label: str) -> None:
    print(f"{'Epoch':>6} | {'Train':>6} | {'Test':>6} | ", end="")
    for k in freq_indices:
        print(f"{'R(sin' + str(k) + ')':>8} {'R(cos' + str(k) + ')':>8} | ", end="")
    print(f"Type(sin_{freq_indices[0]}) [{tracked_label}]")
    print("-" * (36 + 20 * len(freq_indices)))


def print_row(epoch, train_acc, test_acc, tracker, fourier_dofs):
    print(f"{epoch:>6} | {train_acc:>5.0%} | {test_acc:>5.0%} | ", end="")
    for dof in fourier_dofs:
        latest = tracker.latest(dof)
        rho = latest.correlation if latest else 0.0
        print(f"{rho:>8.3f}", end=" ")
        if dof.name.startswith("cos_"):
            print("| ", end="")
    latest_first = tracker.latest(fourier_dofs[0])
    ktype = latest_first.knowledge_type if latest_first else "n/a"
    print(ktype)


def print_summary(tracker, fourier_dofs, tracked_label: str):
    print()
    print("=" * 70)
    print("Phase Transition Analysis")
    print("=" * 70)
    print()
    for dof in fourier_dofs:
        grok = tracker.detect_grokking(dof)
        resonance = tracker.detect_resonance(dof)
        forgetting = tracker.detect_forgetting(dof)

        print(f"  {dof.name}:")
        if grok is not None:
            print(f"    Grokking detected at epoch {grok}")
        else:
            print("    No grokking detected")
        if resonance:
            epochs_str = str(resonance[:5])
            print(f"    Resonance at epochs: {epochs_str}{'...' if len(resonance) > 5 else ''}")
        if forgetting:
            epochs_str = str(forgetting[:5])
            print(f"    Forgetting at epochs: {epochs_str}{'...' if len(forgetting) > 5 else ''}")

        traj = tracker.trajectory(dof)
        if traj:
            a = traj[-1].assessment
            best = a.best_internal_dof
            best_name = best.name if best else "none"
            contributing = len(a.contributing_dofs) if a.contributing_dofs else 0
            print(
                f"    Final: R={a.correlation:.3f}, ε={a.systematic_error:.3f}, "
                f"σ={a.random_error:.3f}, C={a.calibration:.3f}, type={a.knowledge_type}"
                f", features={contributing}, best_{tracked_label}={best_name}"
            )
        print()


# ---------------------------------------------------------------------------
# Part 1: Linear Readout Baseline
# ---------------------------------------------------------------------------


def run_ridge_baseline(p, reservoir_sizes, input_scalings, seed=42):
    """Sweep linear-readout capacity on fixed reservoir states."""
    print("=" * 70)
    print("Part 1: Linear Readout Baseline")
    print("=" * 70)
    print()

    (train_x, train_y), (test_x, test_y) = make_dataset(p, seed=seed)
    Y_train_onehot = F.one_hot(train_y, p).float().cpu().numpy()

    best_acc = 0.0
    best_config = None
    best_esn = None

    for N in reservoir_sizes:
        for iscale in input_scalings:
            esn = EchoStateNetwork(
                input_dim=2 * p,
                reservoir_size=N,
                output_dim=p,
                input_scaling=iscale,
                seed=seed,
            ).to(DEVICE)

            with torch.no_grad():
                H_train = esn.reservoir_state(train_x).cpu().numpy()
                H_test = esn.reservoir_state(test_x).cpu().numpy()

            train_acc, test_acc, W_out = ridge_regression(
                H_train, Y_train_onehot, H_test, test_y.cpu(), reg_lambda=1e-4
            )
            print(f"  N={N:>4}, input_scaling={iscale:.1f}: train={train_acc:>5.1%}  test={test_acc:>5.1%}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_config = (N, iscale)
                best_esn = esn

    print()
    print(f"  Best: N={best_config[0]}, input_scaling={best_config[1]:.1f}, test={best_acc:.1%}")
    if best_acc < 0.05:
        print("  Note: a linear readout still fails to infer the modular-addition")
        print("  rule under the current hyperparameters and random split.")
    print()

    return best_esn, best_config


def run_fourier_analysis(esn, p, num_freqs=5):
    """Fourier analysis on reservoir states and K(d_ext) assessment."""
    print("-" * 70)
    print("Fourier Analysis (sum-averaged reservoir states)")
    print("-" * 70)
    print()

    h_avg = compute_sum_averaged_reservoir(esn, p)
    top_freqs = discover_top_frequencies(h_avg, p, top_n=num_freqs)

    print("  Top frequencies by DFT power:")
    for k, frac in top_freqs:
        print(f"    k={k:>2}: {frac:.1%}")

    # Check power uniformity
    all_freqs = discover_top_frequencies(h_avg, p, top_n=(p - 1) // 2)
    fracs = [f for _, f in all_freqs]
    cv = np.std(fracs) / np.mean(fracs) if np.mean(fracs) > 0 else 0
    print(f"\n  Power spectrum CV = {cv:.2f} (0 = perfectly uniform, >1 = peaked)")
    if cv < 0.5:
        print("  → Reservoir has NO Fourier structure (power roughly uniform)")
    else:
        print("  → Reservoir shows frequency preference")

    # K(d_ext) assessment
    print()
    freq_indices = [k for k, _ in top_freqs[:num_freqs]]

    # Single-feature K
    observer1, tracker1, fdofs1, ndofs1 = setup_observer_and_tracker(
        freq_indices,
        esn.reservoir_size,
        max_features=1,
        internal_prefix="neuron",
        min_samples=min(10, p),
    )
    populate_observations(observer1, h_avg, p, fdofs1, ndofs1, freq_indices)
    tracker1.step(0)

    print("  K(d_ext) on reservoir (max_features=1, BEFORE readout training):")
    for dof in fdofs1:
        a = tracker1.latest(dof)
        if a:
            print(f"    {dof.name}: {a.knowledge_type} (ρ={a.correlation:.3f})")

    # Multi-feature K
    mf = min(50, esn.reservoir_size, p // 2)
    observer2, tracker2, fdofs2, ndofs2 = setup_observer_and_tracker(
        freq_indices,
        esn.reservoir_size,
        max_features=mf,
        internal_prefix="neuron",
        min_samples=min(10, p),
    )
    populate_observations(observer2, h_avg, p, fdofs2, ndofs2, freq_indices)
    tracker2.step(0)

    print(f"\n  K(d_ext) on reservoir (max_features={mf}, BEFORE readout training):")
    for dof in fdofs2:
        a = tracker2.latest(dof)
        if a:
            n_contrib = len(a.contributing_dofs) if a.contributing_dofs else 0
            print(f"    {dof.name}: {a.knowledge_type} (ρ={a.correlation:.3f}, features={n_contrib})")

    print()
    return freq_indices


# ---------------------------------------------------------------------------
# Part 2: SGD Readout Training with K Trajectory
# ---------------------------------------------------------------------------


def run_sgd_readout(
    p,
    reservoir_size,
    input_scaling,
    freq_indices,
    num_freqs=3,
    spectral_radius=0.99,
    settle_steps=5,
    lr=1e-3,
    weight_decay=0.1,
    num_epochs=2000,
    eval_interval=50,
    max_features=1,
    seed=42,
):
    """Train readout via SGD, tracking K trajectory."""
    print("=" * 70)
    print("Part 2: SGD Readout Training (K trajectory)")
    print("=" * 70)
    print(
        f"  reservoir_size={reservoir_size}, input_scaling={input_scaling}, "
        f"spectral_radius={spectral_radius}, settle_steps={settle_steps}"
    )
    print(f"  lr={lr}, weight_decay={weight_decay}, epochs={num_epochs}")
    print(f"  max_features={max_features}")
    print()

    esn = EchoStateNetwork(
        input_dim=2 * p,
        reservoir_size=reservoir_size,
        output_dim=p,
        spectral_radius=spectral_radius,
        input_scaling=input_scaling,
        settle_steps=settle_steps,
        seed=seed,
    ).to(DEVICE)

    (train_x, train_y), (test_x, test_y) = make_dataset(p, seed=seed)

    # Only train readout parameters
    optimizer = torch.optim.AdamW(
        list(esn.readout_in.parameters()) + list(esn.readout_out.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Pre-compute fixed reservoir states (they never change)
    with torch.no_grad():
        H_train = esn.reservoir_state(train_x)
        H_test = esn.reservoir_state(test_x)

    # Use top frequencies from Part 1 (or default)
    fi = freq_indices[:num_freqs] if len(freq_indices) >= num_freqs else list(range(1, num_freqs + 1))

    print(f"  Tracked frequencies: k={fi}")
    print("  Tracking K on sum-averaged readout hidden features.")
    print("  Reservoir-only K is constant by construction; these features show what")
    print("  the nonlinear readout head is actually extracting from the fixed reservoir.")
    print()
    print_header(fi, tracked_label="readout")

    observer, tracker, fourier_dofs, feature_dofs = setup_observer_and_tracker(
        fi,
        esn.readout_hidden_dim,
        max_features=max_features,
        internal_prefix="readout",
        min_samples=min(10, p),
    )

    for epoch in range(num_epochs):
        # Forward through pre-computed reservoir states
        logits = esn.logits_from_state(H_train)
        loss = criterion(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            esn.eval()
            with torch.no_grad():
                train_acc = (esn.logits_from_state(H_train).argmax(1) == train_y).float().mean().item()
                test_acc = (esn.logits_from_state(H_test).argmax(1) == test_y).float().mean().item()

            feature_avg = compute_sum_averaged_readout_features(esn, p)
            populate_observations(observer, feature_avg, p, fourier_dofs, feature_dofs, fi)
            tracker.step(epoch)
            print_row(epoch, train_acc, test_acc, tracker, fourier_dofs)

    print_summary(tracker, fourier_dofs, tracked_label="readout")

    return esn, tracker, fourier_dofs


# ---------------------------------------------------------------------------
# Part 3: Spectral Analysis
# ---------------------------------------------------------------------------


def run_spectral_analysis(esn, p, freq_indices):
    """Analyze recurrent reservoir structure and readout alignment."""
    print("=" * 70)
    print("Part 3: Spectral Analysis")
    print("=" * 70)
    print()

    W_in = esn.W_in.cpu().numpy()
    W_res = esn.W_res.cpu().numpy()
    W_out = esn.readout_out.weight.detach().cpu().numpy()  # (p, H_readout)

    eigvals = np.linalg.eigvals(W_res)
    radii = np.abs(eigvals)
    spectral_radius = float(radii.max())

    print(f"  W_in shape: {W_in.shape}")
    print(f"  W_res shape: {W_res.shape}")
    print(f"  Spectral radius: {spectral_radius:.3f}")
    print(f"  Mean eigenvalue radius: {radii.mean():.3f}")
    print()

    eigvecs = np.linalg.eig(W_res)[1]
    readout_input_weights = esn.readout_in.weight.detach().cpu().numpy()  # (H_readout, N)
    readout_mode_projection = np.abs(readout_input_weights @ eigvecs)
    mode_importance = np.linalg.norm(readout_mode_projection, axis=0)
    top_modes = np.argsort(mode_importance)[::-1][:10]

    print("  Readout mode importance (top 10 recurrent eigenmodes):")
    for idx in top_modes:
        lam = eigvals[idx]
        print(
            f"    Mode {idx} (|λ|={abs(lam):.3f}, angle={np.angle(lam):.3f}): "
            f"readout importance = {mode_importance[idx]:.3f}"
        )
    print()

    # Readout-projected DFT
    h_avg = compute_sum_averaged_reservoir(esn, p)
    with torch.no_grad():
        h_tensor = torch.tensor(h_avg, dtype=torch.float32, device=DEVICE)
        readout_features = esn.readout_features_from_state(h_tensor).cpu().numpy()
        h_projected = readout_features @ W_out.T  # (p, p) — sum class to logit space

    dft_readout = np.fft.fft(h_projected, axis=0)
    power_readout = np.abs(dft_readout) ** 2
    total_per_freq = power_readout.sum(axis=1)
    max_k = (p - 1) // 2
    non_dc = total_per_freq[1 : max_k + 1].sum()

    print("  Readout-projected DFT (top frequencies in logit space):")
    freq_power = [(k, total_per_freq[k] / non_dc) for k in range(1, max_k + 1)]
    freq_power.sort(key=lambda x: x[1], reverse=True)
    for k, frac in freq_power[:10]:
        marker = " <<<" if k in freq_indices else ""
        print(f"    k={k:>2}: {frac:.1%}{marker}")
    print()

    # Compare: which frequencies does the readout extract vs raw reservoir?
    raw_top = discover_top_frequencies(h_avg, p, top_n=5)
    print("  Raw reservoir top frequencies: ", end="")
    for k, frac in raw_top:
        print(f"k={k} ({frac:.1%})", end="  ")
    print()
    print("  Readout-projected top frequencies: ", end="")
    for k, frac in freq_power[:5]:
        print(f"k={k} ({frac:.1%})", end="  ")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    p: int = 97,
    num_freqs: int = 3,
    seed: int = 42,
):
    """Run the full RC-1 experiment."""
    t0 = time.time()

    print("=" * 70)
    print("RC-1: Echo State Network — Modular Addition")
    print("=" * 70)
    print(f"  p={p}, seed={seed}")
    print("  Mode: sequential ESN (a then b through recurrent state)")
    print(f"  Device: {DEVICE}")
    print()

    # Part 1: linear-readout sweep
    best_esn, (best_N, best_iscale) = run_ridge_baseline(
        p,
        reservoir_sizes=[512, 729],
        input_scalings=[0.1, 0.5],
        seed=seed,
    )

    tuned_esn = EchoStateNetwork(
        input_dim=2 * p,
        reservoir_size=729,
        output_dim=p,
        spectral_radius=0.99,
        input_scaling=0.1,
        settle_steps=5,
        seed=seed,
    ).to(DEVICE)
    freq_indices = run_fourier_analysis(tuned_esn, p, num_freqs=num_freqs)

    # Part 2 uses the known-good nonlinear-readout configuration.
    esn_sgd, tracker, fourier_dofs = run_sgd_readout(
        p,
        reservoir_size=729,
        input_scaling=0.1,
        freq_indices=freq_indices,
        num_freqs=num_freqs,
        spectral_radius=0.99,
        settle_steps=5,
        lr=5e-3,
        weight_decay=1.2,
        num_epochs=9000,
        eval_interval=250,
        max_features=min(9, p // 10),  # cap at n_samples//10 rule
        seed=seed,
    )

    # Part 3: Spectral analysis on trained ESN
    run_spectral_analysis(esn_sgd, p, freq_indices[:num_freqs])

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
